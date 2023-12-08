import os
import glob
import logging
import importlib
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from core.prefetch_dataloader import PrefetchDataLoader, CPUPrefetcher
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from torch.utils.tensorboard import SummaryWriter

from core.lr_scheduler import MultiStepRestartLR, CosineAnnealingRestartLR
from core.dataset import TrainDataset

from model.modules.flow_comp_raft import RAFT_bi, FlowLoss, EdgeLoss

# from skimage.feature import canny
from model.canny.canny_filter import Canny
from RAFT.utils.flow_viz_pt import flow_to_image


class Trainer:
    def __init__(self, config):
        self.config = config
        self.epoch = 0
        self.iteration = 0
        self.num_local_frames = config['train_data_loader']['num_local_frames']
        self.num_ref_frames = config['train_data_loader']['num_ref_frames']

        # setup data set and data loader
        self.train_dataset = TrainDataset(config['train_data_loader'])

        self.train_sampler = None
        self.train_args = config['trainer']
        if config['distributed']:
            self.train_sampler = DistributedSampler(
                self.train_dataset,
                num_replicas=config['world_size'],
                rank=config['global_rank'])

        dataloader_args = dict(
            dataset=self.train_dataset,
            batch_size=self.train_args['batch_size'] // config['world_size'],
            shuffle=(self.train_sampler is None),
            num_workers=self.train_args['num_workers'],
            sampler=self.train_sampler,
            drop_last=True)

        self.train_loader = PrefetchDataLoader(self.train_args['num_prefetch_queue'], **dataloader_args)
        self.prefetcher = CPUPrefetcher(self.train_loader)

        # set raft
        self.fix_raft = RAFT_bi(device = self.config['device'])
        self.flow_loss = FlowLoss()
        self.edge_loss = EdgeLoss()
        self.canny = Canny(sigma=(2,2), low_threshold=0.1, high_threshold=0.2)

        # setup models including generator and discriminator
        net = importlib.import_module('model.' + config['model']['net'])
        self.netG = net.RecurrentFlowCompleteNet()
        # print(self.netG)
        self.netG = self.netG.to(self.config['device'])

        # setup optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()
        self.load()

        if config['distributed']:
            self.netG = DDP(self.netG,
                            device_ids=[self.config['local_rank']],
                            output_device=self.config['local_rank'],
                            broadcast_buffers=True,
                            find_unused_parameters=True)

        # set summary writer
        self.dis_writer = None
        self.gen_writer = None
        self.summary = {}
        if self.config['global_rank'] == 0 or (not config['distributed']):
            self.gen_writer = SummaryWriter(
                os.path.join(config['save_dir'], 'gen'))

    def setup_optimizers(self):
        """Set up optimizers."""
        backbone_params = []
        for name, param in self.netG.named_parameters():
            if param.requires_grad:
                backbone_params.append(param)
            else:
                print(f'Params {name} will not be optimized.')
                
        optim_params = [
            {
                'params': backbone_params,
                'lr': self.config['trainer']['lr']
            },
        ]

        self.optimG = torch.optim.Adam(optim_params,
                                       betas=(self.config['trainer']['beta1'],
                                              self.config['trainer']['beta2']))


    def setup_schedulers(self):
        """Set up schedulers."""
        scheduler_opt = self.config['trainer']['scheduler']
        scheduler_type = scheduler_opt.pop('type')

        if scheduler_type in ['MultiStepLR', 'MultiStepRestartLR']:
            self.scheG = MultiStepRestartLR(
                self.optimG,
                milestones=scheduler_opt['milestones'],
                gamma=scheduler_opt['gamma'])
        elif scheduler_type == 'CosineAnnealingRestartLR':
            self.scheG = CosineAnnealingRestartLR(
                self.optimG,
                periods=scheduler_opt['periods'],
                restart_weights=scheduler_opt['restart_weights'])
        else:
            raise NotImplementedError(
                f'Scheduler {scheduler_type} is not implemented yet.')

    def update_learning_rate(self):
        """Update learning rate."""
        self.scheG.step()

    def get_lr(self):
        """Get current learning rate."""
        return self.optimG.param_groups[0]['lr']

    def add_summary(self, writer, name, val):
        """Add tensorboard summary."""
        if name not in self.summary:
            self.summary[name] = 0
        self.summary[name] += val
        n = self.train_args['log_freq']
        if writer is not None and self.iteration % n == 0:
            writer.add_scalar(name, self.summary[name] / n, self.iteration)
            self.summary[name] = 0

    def load(self):
        """Load netG."""
        # get the latest checkpoint
        model_path = self.config['save_dir']
        if os.path.isfile(os.path.join(model_path, 'latest.ckpt')):
            latest_epoch = open(os.path.join(model_path, 'latest.ckpt'),
                                'r').read().splitlines()[-1]
        else:
            ckpts = [
                os.path.basename(i).split('.pth')[0]
                for i in glob.glob(os.path.join(model_path, '*.pth'))
            ]
            ckpts.sort()
            latest_epoch = ckpts[-1][4:] if len(ckpts) > 0 else None

        if latest_epoch is not None:
            gen_path = os.path.join(model_path, f'gen_{int(latest_epoch):06d}.pth')
            opt_path = os.path.join(model_path,f'opt_{int(latest_epoch):06d}.pth')

            if self.config['global_rank'] == 0:
                print(f'Loading model from {gen_path}...')
            dataG = torch.load(gen_path, map_location=self.config['device'])
            self.netG.load_state_dict(dataG)


            data_opt = torch.load(opt_path, map_location=self.config['device'])
            self.optimG.load_state_dict(data_opt['optimG'])
            self.scheG.load_state_dict(data_opt['scheG'])

            self.epoch = data_opt['epoch']
            self.iteration = data_opt['iteration']

        else:
            if self.config['global_rank'] == 0:
                print('Warnning: There is no trained model found.'
                      'An initialized model will be used.')

    def save(self, it):
        """Save parameters every eval_epoch"""
        if self.config['global_rank'] == 0:
            # configure path
            gen_path = os.path.join(self.config['save_dir'],
                                    f'gen_{it:06d}.pth')
            opt_path = os.path.join(self.config['save_dir'],
                                    f'opt_{it:06d}.pth')
            print(f'\nsaving model to {gen_path} ...')

            # remove .module for saving
            if isinstance(self.netG, torch.nn.DataParallel) or isinstance(self.netG, DDP):
                netG = self.netG.module
            else:
                netG = self.netG

            # save checkpoints
            torch.save(netG.state_dict(), gen_path)
            torch.save(
                {
                    'epoch': self.epoch,
                    'iteration': self.iteration,
                    'optimG': self.optimG.state_dict(),
                    'scheG': self.scheG.state_dict()
                }, opt_path)

            latest_path = os.path.join(self.config['save_dir'], 'latest.ckpt')
            os.system(f"echo {it:06d} > {latest_path}")

    def train(self):
        """training entry"""
        pbar = range(int(self.train_args['iterations']))
        if self.config['global_rank'] == 0:
            pbar = tqdm(pbar,
                        initial=self.iteration,
                        dynamic_ncols=True,
                        smoothing=0.01)

        os.makedirs('logs', exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(filename)s[line:%(lineno)d]"
            "%(levelname)s %(message)s",
            datefmt="%a, %d %b %Y %H:%M:%S",
            filename=f"logs/{self.config['save_dir'].split('/')[-1]}.log",
            filemode='w')

        while True:
            self.epoch += 1
            self.prefetcher.reset()
            if self.config['distributed']:
                self.train_sampler.set_epoch(self.epoch)
            self._train_epoch(pbar)
            if self.iteration > self.train_args['iterations']:
                break
        print('\nEnd training....')

    # def get_edges(self, flows): # fgvc
    #     # (b, t, 2, H, W)
    #     b, t, _, h, w = flows.shape
    #     flows = flows.view(-1, 2, h, w)
    #     flows_list = flows.permute(0, 2, 3, 1).cpu().numpy()
    #     edges = []
    #     for f in list(flows_list):
    #         flows_gray = (f[:, :, 0] ** 2 + f[:, :, 1] ** 2) ** 0.5
    #         if flows_gray.max() < 1:
    #             flows_gray = flows_gray*0
    #         else:
    #             flows_gray = flows_gray / flows_gray.max()
            
    #         edge = canny(flows_gray, sigma=2, low_threshold=0.1, high_threshold=0.2) # fgvc
    #         edge = torch.from_numpy(edge).view(1, 1, h, w).float()
    #         edges.append(edge)
    #     edges = torch.stack(edges, dim=0).to(self.config['device'])
    #     edges = edges.view(b, t, 1, h, w)
    #     return edges

    def get_edges(self, flows): 
        # (b, t, 2, H, W)
        b, t, _, h, w = flows.shape
        flows = flows.view(-1, 2, h, w)
        flows_gray = (flows[:, 0, None] ** 2 + flows[:, 1, None] ** 2) ** 0.5
        if flows_gray.max() < 1:
            flows_gray = flows_gray*0
        else:
            flows_gray = flows_gray / flows_gray.max()
            
        magnitude, edges = self.canny(flows_gray.float())
        edges = edges.view(b, t, 1, h, w)
        return edges
        
    def _train_epoch(self, pbar):
        """Process input and calculate loss every training epoch"""
        device = self.config['device']
        train_data = self.prefetcher.next()
        while train_data is not None:
            self.iteration += 1
            frames, masks, flows_f, flows_b, _ = train_data
            frames, masks = frames.to(device), masks.to(device)
            masks = masks.float()

            l_t = self.num_local_frames
            b, t, c, h, w = frames.size()
            gt_local_frames = frames[:, :l_t, ...]
            local_masks = masks[:, :l_t, ...].contiguous()

            # get gt optical flow
            if flows_f[0] == 'None' or flows_b[0] == 'None':
                gt_flows_bi = self.fix_raft(gt_local_frames)
            else:
                gt_flows_bi = (flows_f.to(device), flows_b.to(device))

            # get gt edge
            gt_edges_forward = self.get_edges(gt_flows_bi[0])
            gt_edges_backward = self.get_edges(gt_flows_bi[1])
            gt_edges_bi = [gt_edges_forward, gt_edges_backward]

            # complete flow
            pred_flows_bi, pred_edges_bi = self.netG.module.forward_bidirect_flow(gt_flows_bi, local_masks)

            # optimize net_g
            self.optimG.zero_grad()

            # compulte flow_loss
            flow_loss, warp_loss = self.flow_loss(pred_flows_bi, gt_flows_bi, local_masks, gt_local_frames)
            flow_loss = flow_loss * self.config['losses']['flow_weight']
            warp_loss = warp_loss * 0.01
            self.add_summary(self.gen_writer, 'loss/flow_loss', flow_loss.item())
            self.add_summary(self.gen_writer, 'loss/warp_loss', warp_loss.item())

            # compute edge loss
            edge_loss = self.edge_loss(pred_edges_bi, gt_edges_bi, local_masks)
            edge_loss = edge_loss*1.0
            self.add_summary(self.gen_writer, 'loss/edge_loss', edge_loss.item())

            loss = flow_loss + warp_loss + edge_loss
            loss.backward()
            self.optimG.step()
            self.update_learning_rate()

            # write image to tensorboard
            # if self.iteration % 200 == 0:             
            if self.iteration % 200 == 0 and self.gen_writer is not None:        
                t = 5     
                # forward to cpu
                gt_flows_forward_cpu = flow_to_image(gt_flows_bi[0][0]).cpu()
                masked_flows_forward_cpu = (gt_flows_forward_cpu[t] * (1-local_masks[0][t].cpu())).to(gt_flows_forward_cpu)
                pred_flows_forward_cpu = flow_to_image(pred_flows_bi[0][0]).cpu()

                flow_results = torch.cat([gt_flows_forward_cpu[t], masked_flows_forward_cpu, pred_flows_forward_cpu[t]], 1)
                self.gen_writer.add_image('img/flow-f:gt-pred', flow_results, self.iteration)

                # backward to cpu
                gt_flows_backward_cpu = flow_to_image(gt_flows_bi[1][0]).cpu()
                masked_flows_backward_cpu = (gt_flows_backward_cpu[t] * (1-local_masks[0][t+1].cpu())).to(gt_flows_backward_cpu)
                pred_flows_backward_cpu = flow_to_image(pred_flows_bi[1][0]).cpu()

                flow_results = torch.cat([gt_flows_backward_cpu[t], masked_flows_backward_cpu, pred_flows_backward_cpu[t]], 1)
                self.gen_writer.add_image('img/flow-b:gt-pred', flow_results, self.iteration)

                # TODO: show edge
                # forward
                gt_edges_forward_cpu = gt_edges_bi[0][0].cpu()
                masked_edges_forward_cpu = (gt_edges_forward_cpu[t] * (1-local_masks[0][t].cpu())).to(gt_edges_forward_cpu)
                pred_edges_forward_cpu = pred_edges_bi[0][0].cpu()

                edge_results = torch.cat([gt_edges_forward_cpu[t], masked_edges_forward_cpu, pred_edges_forward_cpu[t]], 1)
                self.gen_writer.add_image('img/edge-f:gt-pred', edge_results, self.iteration)
                # backward
                gt_edges_backward_cpu = gt_edges_bi[1][0].cpu()
                masked_edges_backward_cpu = (gt_edges_backward_cpu[t] * (1-local_masks[0][t+1].cpu())).to(gt_edges_backward_cpu)
                pred_edges_backward_cpu = pred_edges_bi[1][0].cpu()

                edge_results = torch.cat([gt_edges_backward_cpu[t], masked_edges_backward_cpu, pred_edges_backward_cpu[t]], 1)
                self.gen_writer.add_image('img/edge-b:gt-pred', edge_results, self.iteration)
                
            # console logs
            if self.config['global_rank'] == 0:
                pbar.update(1)
                pbar.set_description((f"flow: {flow_loss.item():.3f}; "
                                      f"warp: {warp_loss.item():.3f}; "
                                      f"edge: {edge_loss.item():.3f}; "
                                      f"lr: {self.get_lr()}"))

                if self.iteration % self.train_args['log_freq'] == 0:
                    logging.info(f"[Iter {self.iteration}] "
                                 f"flow: {flow_loss.item():.4f}; "
                                 f"warp: {warp_loss.item():.4f}")

            # saving models
            if self.iteration % self.train_args['save_freq'] == 0:
                self.save(int(self.iteration))

            if self.iteration > self.train_args['iterations']:
                break

            train_data = self.prefetcher.next()