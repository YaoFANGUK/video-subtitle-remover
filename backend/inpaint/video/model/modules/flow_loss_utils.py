import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def flow_warp(x,
              flow,
              interpolation='bilinear',
              padding_mode='zeros',
              align_corners=True):
    """Warp an image or a feature map with optical flow.
    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, h, w, 2). The last dimension is
            a two-channel, denoting the width and height relative offsets.
            Note that the values are not normalized to [-1, 1].
        interpolation (str): Interpolation mode: 'nearest' or 'bilinear'.
            Default: 'bilinear'.
        padding_mode (str): Padding mode: 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Whether align corners. Default: True.
    Returns:
        Tensor: Warped image or feature map.
    """
    if x.size()[-2:] != flow.size()[1:3]:
        raise ValueError(f'The spatial sizes of input ({x.size()[-2:]}) and '
                         f'flow ({flow.size()[1:3]}) are not the same.')
    _, _, h, w = x.size()
    # create mesh grid
    device = flow.device
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h, device=device), torch.arange(0, w, device=device))
    grid = torch.stack((grid_x, grid_y), 2).type_as(x)  # (w, h, 2)
    grid.requires_grad = False

    grid_flow = grid + flow
    # scale grid_flow to [-1,1]
    grid_flow_x = 2.0 * grid_flow[:, :, :, 0] / max(w - 1, 1) - 1.0
    grid_flow_y = 2.0 * grid_flow[:, :, :, 1] / max(h - 1, 1) - 1.0
    grid_flow = torch.stack((grid_flow_x, grid_flow_y), dim=3)
    output = F.grid_sample(x,
                           grid_flow,
                           mode=interpolation,
                           padding_mode=padding_mode,
                           align_corners=align_corners)
    return output


# def image_warp(image, flow):
#     b, c, h, w = image.size()
#     device = image.device
#     flow = torch.cat([flow[:, 0:1, :, :] / ((w - 1.0) / 2.0), flow[:, 1:2, :, :] / ((h - 1.0) / 2.0)], dim=1) # normalize to [-1~1](from upper left to lower right
#     flow = flow.permute(0, 2, 3, 1) # if you wanna use grid_sample function, the channel(band) shape of show must be in the last dimension
#     x = np.linspace(-1, 1, w)
#     y = np.linspace(-1, 1, h)
#     X, Y = np.meshgrid(x, y)
#     grid = torch.cat((torch.from_numpy(X.astype('float32')).unsqueeze(0).unsqueeze(3),
#                       torch.from_numpy(Y.astype('float32')).unsqueeze(0).unsqueeze(3)), 3).to(device)
#     output = torch.nn.functional.grid_sample(image, grid + flow, mode='bilinear', padding_mode='zeros')
#     return output


def length_sq(x):
    return torch.sum(torch.square(x), dim=1, keepdim=True)


def fbConsistencyCheck(flow_fw, flow_bw, alpha1=0.01, alpha2=0.5):
    flow_bw_warped = flow_warp(flow_bw, flow_fw.permute(0, 2, 3, 1))  # wb(wf(x))
    flow_fw_warped = flow_warp(flow_fw, flow_bw.permute(0, 2, 3, 1))  # wf(wb(x))
    flow_diff_fw = flow_fw + flow_bw_warped  # wf + wb(wf(x))
    flow_diff_bw = flow_bw + flow_fw_warped  # wb + wf(wb(x))

    mag_sq_fw = length_sq(flow_fw) + length_sq(flow_bw_warped)  # |wf| + |wb(wf(x))|
    mag_sq_bw = length_sq(flow_bw) + length_sq(flow_fw_warped)  # |wb| + |wf(wb(x))|
    occ_thresh_fw = alpha1 * mag_sq_fw + alpha2
    occ_thresh_bw = alpha1 * mag_sq_bw + alpha2

    fb_occ_fw = (length_sq(flow_diff_fw) > occ_thresh_fw).float()
    fb_occ_bw = (length_sq(flow_diff_bw) > occ_thresh_bw).float()

    return fb_occ_fw, fb_occ_bw  # fb_occ_fw -> frame2 area occluded by frame1, fb_occ_bw -> frame1 area occluded by frame2


def rgb2gray(image):
    gray_image = image[:, 0] * 0.299 + image[:, 1] * 0.587 + 0.110 * image[:, 2]
    gray_image = gray_image.unsqueeze(1)
    return gray_image


def ternary_transform(image, max_distance=1):
    device = image.device
    patch_size = 2 * max_distance + 1
    intensities = rgb2gray(image) * 255
    out_channels = patch_size * patch_size
    w = np.eye(out_channels).reshape(out_channels, 1, patch_size, patch_size)
    weights = torch.from_numpy(w).float().to(device)
    patches = F.conv2d(intensities, weights, stride=1, padding=1)
    transf = patches - intensities
    transf_norm = transf / torch.sqrt(0.81 + torch.square(transf))
    return transf_norm


def hamming_distance(t1, t2):
    dist = torch.square(t1 - t2)
    dist_norm = dist / (0.1 + dist)
    dist_sum = torch.sum(dist_norm, dim=1, keepdim=True)
    return dist_sum


def create_mask(mask, paddings):
    """
    padding: [[top, bottom], [left, right]]
    """
    shape = mask.shape
    inner_height = shape[2] - (paddings[0][0] + paddings[0][1])
    inner_width = shape[3] - (paddings[1][0] + paddings[1][1])
    inner = torch.ones([inner_height, inner_width])

    mask2d = F.pad(inner, pad=[paddings[1][0], paddings[1][1], paddings[0][0], paddings[0][1]]) 
    mask3d = mask2d.unsqueeze(0)
    mask4d = mask3d.unsqueeze(0).repeat(shape[0], 1, 1, 1)
    return mask4d.detach()


def ternary_loss2(frame1, warp_frame21, confMask, masks, max_distance=1):
    """

    Args:
        frame1: torch tensor, with shape [b * t, c, h, w]
        warp_frame21: torch tensor, with shape [b * t, c, h, w]
        confMask: confidence mask, with shape [b * t, c, h, w]
        masks: torch tensor, with shape [b * t, c, h, w]
        max_distance: maximum distance.

    Returns: ternary loss

    """
    t1 = ternary_transform(frame1)
    t21 = ternary_transform(warp_frame21)
    dist = hamming_distance(t1, t21) 
    loss = torch.mean(dist * confMask * masks) / torch.mean(masks)
    return loss

