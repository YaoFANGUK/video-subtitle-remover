import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

from backend.inpaint.video.raft import RAFT
from backend.inpaint.video.model.modules.flow_loss_utils import flow_warp, ternary_loss2


def initialize_RAFT(model_path='weights/raft-things.pth', device='cuda'):
    """Initializes the RAFT model.
    """
    args = argparse.ArgumentParser()
    args.raft_model = model_path
    args.small = False
    args.mixed_precision = False
    args.alternate_corr = False
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.raft_model, map_location='cpu'))
    model = model.module

    model.to(device)

    return model


class RAFT_bi(nn.Module):
    """Flow completion loss"""
    def __init__(self, model_path='weights/raft-things.pth', device='cuda'):
        super().__init__()
        self.fix_raft = initialize_RAFT(model_path, device=device)

        for p in self.fix_raft.parameters():
            p.requires_grad = False

        self.l1_criterion = nn.L1Loss()
        self.eval()

    def forward(self, gt_local_frames, iters=20):
        b, l_t, c, h, w = gt_local_frames.size()
        # print(gt_local_frames.shape)

        with torch.no_grad():
            gtlf_1 = gt_local_frames[:, :-1, :, :, :].reshape(-1, c, h, w)
            gtlf_2 = gt_local_frames[:, 1:, :, :, :].reshape(-1, c, h, w)
            # print(gtlf_1.shape)

            _, gt_flows_forward = self.fix_raft(gtlf_1, gtlf_2, iters=iters, test_mode=True)
            _, gt_flows_backward = self.fix_raft(gtlf_2, gtlf_1, iters=iters, test_mode=True)

        
        gt_flows_forward = gt_flows_forward.view(b, l_t-1, 2, h, w)
        gt_flows_backward = gt_flows_backward.view(b, l_t-1, 2, h, w)

        return gt_flows_forward, gt_flows_backward


##################################################################################
def smoothness_loss(flow, cmask):
    delta_u, delta_v, mask = smoothness_deltas(flow)
    loss_u = charbonnier_loss(delta_u, cmask)
    loss_v = charbonnier_loss(delta_v, cmask)
    return loss_u + loss_v


def smoothness_deltas(flow):
    """
    flow: [b, c, h, w]
    """
    mask_x = create_mask(flow, [[0, 0], [0, 1]])
    mask_y = create_mask(flow, [[0, 1], [0, 0]])
    mask = torch.cat((mask_x, mask_y), dim=1)
    mask = mask.to(flow.device)
    filter_x = torch.tensor([[0, 0, 0.], [0, 1, -1], [0, 0, 0]])
    filter_y = torch.tensor([[0, 0, 0.], [0, 1, 0], [0, -1, 0]])
    weights = torch.ones([2, 1, 3, 3])
    weights[0, 0] = filter_x
    weights[1, 0] = filter_y
    weights = weights.to(flow.device)

    flow_u, flow_v = torch.split(flow, split_size_or_sections=1, dim=1)
    delta_u = F.conv2d(flow_u, weights, stride=1, padding=1)
    delta_v = F.conv2d(flow_v, weights, stride=1, padding=1)
    return delta_u, delta_v, mask


def second_order_loss(flow, cmask):
    delta_u, delta_v, mask = second_order_deltas(flow)
    loss_u = charbonnier_loss(delta_u, cmask)
    loss_v = charbonnier_loss(delta_v, cmask)
    return loss_u + loss_v


def charbonnier_loss(x, mask=None, truncate=None, alpha=0.45, beta=1.0, epsilon=0.001):
    """
    Compute the generalized charbonnier loss of the difference tensor x
    All positions where mask == 0 are not taken into account
    x: a tensor of shape [b, c, h, w]
    mask: a mask of shape [b, mc, h, w], where mask channels must be either 1 or the same as
    the number of channels of x. Entries should be 0 or 1
    return: loss
    """
    b, c, h, w = x.shape
    norm = b * c * h * w
    error = torch.pow(torch.square(x * beta) + torch.square(torch.tensor(epsilon)), alpha)
    if mask is not None:
        error = mask * error
    if truncate is not None:
        error = torch.min(error, truncate)
    return torch.sum(error) / norm


def second_order_deltas(flow):
    """
    consider the single flow first
    flow shape: [b, c, h, w]
    """
    # create mask
    mask_x = create_mask(flow, [[0, 0], [1, 1]])
    mask_y = create_mask(flow, [[1, 1], [0, 0]])
    mask_diag = create_mask(flow, [[1, 1], [1, 1]])
    mask = torch.cat((mask_x, mask_y, mask_diag, mask_diag), dim=1)
    mask = mask.to(flow.device)

    filter_x = torch.tensor([[0, 0, 0.], [1, -2, 1], [0, 0, 0]])
    filter_y = torch.tensor([[0, 1, 0.], [0, -2, 0], [0, 1, 0]])
    filter_diag1 = torch.tensor([[1, 0, 0.], [0, -2, 0], [0, 0, 1]])
    filter_diag2 = torch.tensor([[0, 0, 1.], [0, -2, 0], [1, 0, 0]])
    weights = torch.ones([4, 1, 3, 3])
    weights[0] = filter_x
    weights[1] = filter_y
    weights[2] = filter_diag1
    weights[3] = filter_diag2
    weights = weights.to(flow.device)

    # split the flow into flow_u and flow_v, conv them with the weights
    flow_u, flow_v = torch.split(flow, split_size_or_sections=1, dim=1)
    delta_u = F.conv2d(flow_u, weights, stride=1, padding=1)
    delta_v = F.conv2d(flow_v, weights, stride=1, padding=1)
    return delta_u, delta_v, mask

def create_mask(tensor, paddings):
    """
    tensor shape: [b, c, h, w]
    paddings: [2 x 2] shape list, the first row indicates up and down paddings
    the second row indicates left and right paddings
    |            |
    |       x    |
    |     x * x  |
    |       x    |
    |            |
    """
    shape = tensor.shape
    inner_height = shape[2] - (paddings[0][0] + paddings[0][1])
    inner_width = shape[3] - (paddings[1][0] + paddings[1][1])
    inner = torch.ones([inner_height, inner_width])
    torch_paddings = [paddings[1][0], paddings[1][1], paddings[0][0], paddings[0][1]]  # left, right, up and down
    mask2d = F.pad(inner, pad=torch_paddings)
    mask3d = mask2d.unsqueeze(0).repeat(shape[0], 1, 1)
    mask4d = mask3d.unsqueeze(1)
    return mask4d.detach()

def ternary_loss(flow_comp, flow_gt, mask, current_frame, shift_frame, scale_factor=1):
    if scale_factor != 1:
        current_frame = F.interpolate(current_frame, scale_factor=1 / scale_factor, mode='bilinear')
        shift_frame = F.interpolate(shift_frame, scale_factor=1 / scale_factor, mode='bilinear')
    warped_sc = flow_warp(shift_frame, flow_gt.permute(0, 2, 3, 1))
    noc_mask = torch.exp(-50. * torch.sum(torch.abs(current_frame - warped_sc), dim=1).pow(2)).unsqueeze(1)
    warped_comp_sc = flow_warp(shift_frame, flow_comp.permute(0, 2, 3, 1))
    loss = ternary_loss2(current_frame, warped_comp_sc, noc_mask, mask)
    return loss

class FlowLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1_criterion = nn.L1Loss()

    def forward(self, pred_flows, gt_flows, masks, frames):
        # pred_flows: b t-1 2 h w
        loss = 0
        warp_loss = 0
        h, w = pred_flows[0].shape[-2:]
        masks = [masks[:,:-1,...].contiguous(), masks[:, 1:, ...].contiguous()]
        frames0 = frames[:,:-1,...]
        frames1 = frames[:,1:,...]
        current_frames = [frames0, frames1]
        next_frames = [frames1, frames0]
        for i in range(len(pred_flows)):
            # print(pred_flows[i].shape)
            combined_flow = pred_flows[i] * masks[i] + gt_flows[i] * (1-masks[i])
            l1_loss = self.l1_criterion(pred_flows[i] * masks[i], gt_flows[i] * masks[i]) / torch.mean(masks[i])
            l1_loss += self.l1_criterion(pred_flows[i] * (1-masks[i]), gt_flows[i] * (1-masks[i])) / torch.mean((1-masks[i]))

            smooth_loss = smoothness_loss(combined_flow.reshape(-1,2,h,w), masks[i].reshape(-1,1,h,w))
            smooth_loss2 = second_order_loss(combined_flow.reshape(-1,2,h,w), masks[i].reshape(-1,1,h,w))
            
            warp_loss_i = ternary_loss(combined_flow.reshape(-1,2,h,w), gt_flows[i].reshape(-1,2,h,w), 
                            masks[i].reshape(-1,1,h,w), current_frames[i].reshape(-1,3,h,w), next_frames[i].reshape(-1,3,h,w)) 

            loss += l1_loss + smooth_loss + smooth_loss2

            warp_loss += warp_loss_i
            
        return loss, warp_loss


def edgeLoss(preds_edges, edges):
    """

    Args:
        preds_edges: with shape [b, c, h , w]
        edges: with shape [b, c, h, w]

    Returns: Edge losses

    """
    mask = (edges > 0.5).float()
    b, c, h, w = mask.shape
    num_pos = torch.sum(mask, dim=[1, 2, 3]).float() # Shape: [b,].
    num_neg = c * h * w - num_pos # Shape: [b,].
    neg_weights = (num_neg / (num_pos + num_neg)).unsqueeze(1).unsqueeze(2).unsqueeze(3)
    pos_weights = (num_pos / (num_pos + num_neg)).unsqueeze(1).unsqueeze(2).unsqueeze(3)
    weight = neg_weights * mask + pos_weights * (1 - mask)  # weight for debug
    losses = F.binary_cross_entropy_with_logits(preds_edges.float(), edges.float(), weight=weight, reduction='none')
    loss = torch.mean(losses)
    return loss

class EdgeLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred_edges, gt_edges, masks):
        # pred_flows: b t-1 1 h w
        loss = 0
        h, w = pred_edges[0].shape[-2:]
        masks = [masks[:,:-1,...].contiguous(), masks[:, 1:, ...].contiguous()]
        for i in range(len(pred_edges)):
            # print(f'edges_{i}',  torch.sum(gt_edges[i])) # debug
            combined_edge = pred_edges[i] * masks[i] + gt_edges[i] * (1-masks[i])
            edge_loss = (edgeLoss(pred_edges[i].reshape(-1,1,h,w), gt_edges[i].reshape(-1,1,h,w)) \
                        + 5 * edgeLoss(combined_edge.reshape(-1,1,h,w), gt_edges[i].reshape(-1,1,h,w)))
            loss += edge_loss 

        return loss


class FlowSimpleLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1_criterion = nn.L1Loss()

    def forward(self, pred_flows, gt_flows):
        # pred_flows: b t-1 2 h w
        loss = 0
        h, w = pred_flows[0].shape[-2:]
        h_orig, w_orig = gt_flows[0].shape[-2:]
        pred_flows = [f.view(-1, 2, h, w) for f in pred_flows]
        gt_flows = [f.view(-1, 2, h_orig, w_orig) for f in gt_flows]

        ds_factor = 1.0*h/h_orig
        gt_flows = [F.interpolate(f, scale_factor=ds_factor, mode='area') * ds_factor for f in gt_flows]
        for i in range(len(pred_flows)):
            loss += self.l1_criterion(pred_flows[i], gt_flows[i])

        return loss