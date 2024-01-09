import os
import glob
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from tensorboardX import SummaryWriter

from backend.inpaint.sttn.auto_sttn import Discriminator
from backend.inpaint.sttn.auto_sttn import InpaintGenerator
from backend.tools.train.dataset_sttn import Dataset
from backend.tools.train.loss_sttn import AdversarialLoss


class Trainer:
    def __init__(self, config, debug=False):
        # 训练器初始化
        self.config = config  # 保存配置信息
        self.epoch = 0  # 当前训练所处的epoch
        self.iteration = 0  # 当前训练迭代次数
        if debug:
            # 如果是调试模式，设置更频繁的保存和验证频率
            self.config['trainer']['save_freq'] = 5
            self.config['trainer']['valid_freq'] = 5
            self.config['trainer']['iterations'] = 5

        # 设置数据集和数据加载器
        self.train_dataset = Dataset(config['data_loader'], split='train', debug=debug)  # 创建训练集对象
        self.train_sampler = None  # 初始化训练集采样器为None
        self.train_args = config['trainer']  # 训练过程参数
        if config['distributed']:
            # 如果是分布式训练，则初始化分布式采样器
            self.train_sampler = DistributedSampler(
                self.train_dataset,
                num_replicas=config['world_size'],
                rank=config['global_rank']
            )
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.train_args['batch_size'] // config['world_size'],
            shuffle=(self.train_sampler is None),  # 如果没有采样器则进行打乱
            num_workers=self.train_args['num_workers'],
            sampler=self.train_sampler
        )

        # 设置损失函数
        self.adversarial_loss = AdversarialLoss(type=self.config['losses']['GAN_LOSS'])  # 对抗性损失
        self.adversarial_loss = self.adversarial_loss.to(self.config['device'])  # 将损失函数转移到相应设备
        self.l1_loss = nn.L1Loss()  # L1损失

        # 初始化生成器和判别器模型
        self.netG = InpaintGenerator()  # 生成网络
        self.netG = self.netG.to(self.config['device'])  # 转移到设备
        self.netD = Discriminator(
            in_channels=3, use_sigmoid=config['losses']['GAN_LOSS'] != 'hinge'
        )
        self.netD = self.netD.to(self.config['device'])  # 判别网络
        # 初始化优化器
        self.optimG = torch.optim.Adam(
            self.netG.parameters(),  # 生成器参数
            lr=config['trainer']['lr'],  # 学习率
            betas=(self.config['trainer']['beta1'], self.config['trainer']['beta2'])
        )
        self.optimD = torch.optim.Adam(
            self.netD.parameters(),  # 判别器参数
            lr=config['trainer']['lr'],  # 学习率
            betas=(self.config['trainer']['beta1'], self.config['trainer']['beta2'])
        )
        self.load()  # 加载模型

        if config['distributed']:
            # 如果是分布式训练，则使用分布式数据并行包装器
            self.netG = DDP(
                self.netG,
                device_ids=[self.config['local_rank']],
                output_device=self.config['local_rank'],
                broadcast_buffers=True,
                find_unused_parameters=False
            )
            self.netD = DDP(
                self.netD,
                device_ids=[self.config['local_rank']],
                output_device=self.config['local_rank'],
                broadcast_buffers=True,
                find_unused_parameters=False
            )

        # 设置日志记录器
        self.dis_writer = None  # 判别器写入器
        self.gen_writer = None  # 生成器写入器
        self.summary = {}  # 存放摘要统计
        if self.config['global_rank'] == 0 or (not config['distributed']):
            # 如果不是分布式训练或者为分布式训练的主节点
            self.dis_writer = SummaryWriter(
                os.path.join(config['save_dir'], 'dis')
            )
            self.gen_writer = SummaryWriter(
                os.path.join(config['save_dir'], 'gen')
            )

    # 获取当前学习率
    def get_lr(self):
        return self.optimG.param_groups[0]['lr']

    # 调整学习率
    def adjust_learning_rate(self):
        # 计算衰减的学习率
        decay = 0.1 ** (min(self.iteration, self.config['trainer']['niter_steady']) // self.config['trainer']['niter'])
        new_lr = self.config['trainer']['lr'] * decay
        # 如果新的学习率和当前学习率不同，则更新优化器中的学习率
        if new_lr != self.get_lr():
            for param_group in self.optimG.param_groups:
                param_group['lr'] = new_lr
            for param_group in self.optimD.param_groups:
                param_group['lr'] = new_lr

    # 添加摘要信息
    def add_summary(self, writer, name, val):
        # 添加并更新统计信息，每次迭代都累加
        if name not in self.summary:
            self.summary[name] = 0
        self.summary[name] += val
        # 每100次迭代记录一次
        if writer is not None and self.iteration % 100 == 0:
            writer.add_scalar(name, self.summary[name] / 100, self.iteration)
            self.summary[name] = 0

    # 加载模型netG and netD
    def load(self):
        model_path = self.config['save_dir']  # 模型的保存路径
        # 检测是否存在最近的模型检查点
        if os.path.isfile(os.path.join(model_path, 'latest.ckpt')):
            # 读取最后一个epoch的编号
            latest_epoch = open(os.path.join(
                model_path, 'latest.ckpt'), 'r').read().splitlines()[-1]
        else:
            # 如果不存在latest.ckpt，尝试读取存储好的模型文件列表，获取最近的一个
            ckpts = [os.path.basename(i).split('.pth')[0] for i in glob.glob(
                os.path.join(model_path, '*.pth'))]
            ckpts.sort()  # 排序模型文件，以获取最近的一个
            latest_epoch = ckpts[-1] if len(ckpts) > 0 else None  # 获取最近的epoch值
        if latest_epoch is not None:
            # 拼接得到生成器和判别器的模型文件路径
            gen_path = os.path.join(
                model_path, 'gen_{}.pth'.format(str(latest_epoch).zfill(5)))
            dis_path = os.path.join(
                model_path, 'dis_{}.pth'.format(str(latest_epoch).zfill(5)))
            opt_path = os.path.join(
                model_path, 'opt_{}.pth'.format(str(latest_epoch).zfill(5)))
            # 如果是主节点，输出加载模型的信息
            if self.config['global_rank'] == 0:
                print('Loading model from {}...'.format(gen_path))
            # 加载生成器模型
            data = torch.load(gen_path, map_location=self.config['device'])
            self.netG.load_state_dict(data['netG'])
            # 加载判别器模型
            data = torch.load(dis_path, map_location=self.config['device'])
            self.netD.load_state_dict(data['netD'])
            # 加载优化器状态
            data = torch.load(opt_path, map_location=self.config['device'])
            self.optimG.load_state_dict(data['optimG'])
            self.optimD.load_state_dict(data['optimD'])
            # 更新当前epoch和迭代次数
            self.epoch = data['epoch']
            self.iteration = data['iteration']
        else:
            # 如果没有找到模型文件，则输出警告信息
            if self.config['global_rank'] == 0:
                print('Warning: There is no trained model found. An initialized model will be used.')

    # 保存模型参数，每次评估周期 (eval_epoch) 调用一次
    def save(self, it):
        # 只在全局排名为0的进程上执行保存操作，通常代表主节点
        if self.config['global_rank'] == 0:
            # 生成保存生成器模型状态字典的文件路径
            gen_path = os.path.join(
                self.config['save_dir'], 'gen_{}.pth'.format(str(it).zfill(5)))
            # 生成保存判别器模型状态字典的文件路径
            dis_path = os.path.join(
                self.config['save_dir'], 'dis_{}.pth'.format(str(it).zfill(5)))
            # 生成保存优化器状态字典的文件路径
            opt_path = os.path.join(
                self.config['save_dir'], 'opt_{}.pth'.format(str(it).zfill(5)))

            # 打印消息表示模型正在保存
            print('\nsaving model to {} ...'.format(gen_path))

            # 判断模型是否是经过DataParallel或DDP包装的，若是则获取原始的模型
            if isinstance(self.netG, torch.nn.DataParallel) or isinstance(self.netG, DDP):
                netG = self.netG.module
                netD = self.netD.module
            else:
                netG = self.netG
                netD = self.netD

            # 保存生成器和判别器的模型参数
            torch.save({'netG': netG.state_dict()}, gen_path)
            torch.save({'netD': netD.state_dict()}, dis_path)
            # 保存当前的epoch、迭代次数和优化器的状态
            torch.save({
                'epoch': self.epoch,
                'iteration': self.iteration,
                'optimG': self.optimG.state_dict(),
                'optimD': self.optimD.state_dict()
            }, opt_path)

            # 写入最新的迭代次数到"latest.ckpt"文件
            os.system('echo {} > {}'.format(str(it).zfill(5),
                                            os.path.join(self.config['save_dir'], 'latest.ckpt')))

        # 训练入口

    def train(self):
        # 初始化进度条范围
        pbar = range(int(self.train_args['iterations']))
        # 如果是全局rank 0的进程，则设置显示进度条
        if self.config['global_rank'] == 0:
            pbar = tqdm(pbar, initial=self.iteration, dynamic_ncols=True, smoothing=0.01)

        # 开始训练循环
        while True:
            self.epoch += 1  # epoch计数增加
            if self.config['distributed']:
                # 如果是分布式训练，则对采样器进行设置，保证每个进程获取的数据不同
                self.train_sampler.set_epoch(self.epoch)

            # 调用训练一个epoch的函数
            self._train_epoch(pbar)
            # 如果迭代次数超过配置中的迭代上限，则退出循环
            if self.iteration > self.train_args['iterations']:
                break
        # 训练结束输出
        print('\nEnd training....')

        # 每个训练周期处理输入并计算损失

    def _train_epoch(self, pbar):
        device = self.config['device']  # 获取设备信息

        # 遍历数据加载器中的数据
        for frames, masks in self.train_loader:
            # 调整学习率
            self.adjust_learning_rate()
            # 迭代次数+1
            self.iteration += 1

            # 将frames和masks转移到设备上
            frames, masks = frames.to(device), masks.to(device)
            b, t, c, h, w = frames.size()  # 获取帧和蒙版的尺寸
            masked_frame = (frames * (1 - masks).float())  # 应用蒙版到图像
            pred_img = self.netG(masked_frame, masks)  # 使用生成器生成填充图像
            # 调整frames和masks的维度以符合网络的输入要求
            frames = frames.view(b * t, c, h, w)
            masks = masks.view(b * t, 1, h, w)
            comp_img = frames * (1. - masks) + masks * pred_img  # 生成最终的组合图像

            gen_loss = 0  # 初始化生成器损失
            dis_loss = 0  # 初始化判别器损失

            # 判别器对抗性损失
            real_vid_feat = self.netD(frames)  # 判别器对真实图像判别
            fake_vid_feat = self.netD(comp_img.detach())  # 判别器对生成图像判别，注意detach是为了不计算梯度
            dis_real_loss = self.adversarial_loss(real_vid_feat, True, True)  # 真实图像的损失
            dis_fake_loss = self.adversarial_loss(fake_vid_feat, False, True)  # 生成图像的损失
            dis_loss += (dis_real_loss + dis_fake_loss) / 2  # 求平均的判别器损失
            # 添加判别器损失到摘要
            self.add_summary(self.dis_writer, 'loss/dis_vid_fake', dis_fake_loss.item())
            self.add_summary(self.dis_writer, 'loss/dis_vid_real', dis_real_loss.item())
            # 优化判别器
            self.optimD.zero_grad()
            dis_loss.backward()
            self.optimD.step()

            # 生成器对抗性损失
            gen_vid_feat = self.netD(comp_img)
            gan_loss = self.adversarial_loss(gen_vid_feat, True, False)  # 生成器的对抗损失
            gan_loss = gan_loss * self.config['losses']['adversarial_weight']  # 权重放大
            gen_loss += gan_loss  # 累加到生成器损失
            # 添加生成器对抗性损失到摘要
            self.add_summary(self.gen_writer, 'loss/gan_loss', gan_loss.item())

            # 生成器L1损失
            hole_loss = self.l1_loss(pred_img * masks, frames * masks)  # 只计算有蒙版区域的损失
            # 考虑蒙版的平均值，乘以配置中的hole_weight
            hole_loss = hole_loss / torch.mean(masks) * self.config['losses']['hole_weight']
            gen_loss += hole_loss  # 累加到生成器损失
            # 添加hole_loss到摘要
            self.add_summary(self.gen_writer, 'loss/hole_loss', hole_loss.item())

            # 计算蒙版外区域的L1损失
            valid_loss = self.l1_loss(pred_img * (1 - masks), frames * (1 - masks))
            # 考虑非蒙版区的平均值，乘以配置中的valid_weight
            valid_loss = valid_loss / torch.mean(1 - masks) * self.config['losses']['valid_weight']
            gen_loss += valid_loss  # 累加到生成器损失
            # 添加valid_loss到摘要
            self.add_summary(self.gen_writer, 'loss/valid_loss', valid_loss.item())

            # 生成器优化
            self.optimG.zero_grad()
            gen_loss.backward()
            self.optimG.step()

            # 控制台日志输出
            if self.config['global_rank'] == 0:
                pbar.update(1)  # 进度条更新
                pbar.set_description((  # 设置进度条描述
                    f"d: {dis_loss.item():.3f}; g: {gan_loss.item():.3f};"  # 打印损失数值
                    f"hole: {hole_loss.item():.3f}; valid: {valid_loss.item():.3f}")
                )

            # 模型保存
            if self.iteration % self.train_args['save_freq'] == 0:
                self.save(int(self.iteration // self.train_args['save_freq']))
            # 迭代次数终止判断
            if self.iteration > self.train_args['iterations']:
                break

