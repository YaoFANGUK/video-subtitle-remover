# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import scipy.ndimage
from PIL import Image

import torch
import torchvision

from backend import config
from backend.inpaint.video.model.modules.flow_comp_raft import RAFT_bi
from backend.inpaint.video.model.recurrent_flow_completion import RecurrentFlowCompleteNet
from backend.inpaint.video.model.propainter import InpaintGenerator
from backend.inpaint.video.core.utils import to_tensors
from backend.inpaint.video.model.misc import get_device

import warnings

warnings.filterwarnings("ignore")


def binary_mask(mask, th=0.1):
    mask[mask > th] = 1
    mask[mask <= th] = 0
    return mask


# read frame-wise masks
def read_mask(mpath, length, size, flow_mask_dilates=8, mask_dilates=5):
    masks_img = []
    masks_dilated = []
    flow_masks = []
    # 如果传入的直接为numpy array
    if isinstance(mpath, np.ndarray):
        masks_img = [Image.fromarray(mpath)]
    # input single img path
    else:
        if isinstance(mpath, str):
            if mpath.endswith(('jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG')):
                masks_img = [Image.open(mpath)]
        else:
            mnames = sorted(os.listdir(mpath))
            for mp in mnames:
                masks_img.append(Image.open(os.path.join(mpath, mp)))

    for mask_img in masks_img:
        mask_img = np.array(mask_img.convert('L'))

        # Dilate 8 pixel so that all known pixel is trustworthy
        if flow_mask_dilates > 0:
            flow_mask_img = scipy.ndimage.binary_dilation(mask_img, iterations=flow_mask_dilates).astype(np.uint8)
        else:
            flow_mask_img = binary_mask(mask_img).astype(np.uint8)
        # Close the small holes inside the foreground objects
        # flow_mask_img = cv2.morphologyEx(flow_mask_img, cv2.MORPH_CLOSE, np.ones((21, 21),np.uint8)).astype(bool)
        # flow_mask_img = scipy.ndimage.binary_fill_holes(flow_mask_img).astype(np.uint8)
        flow_masks.append(Image.fromarray(flow_mask_img * 255))

        if mask_dilates > 0:
            mask_img = scipy.ndimage.binary_dilation(mask_img, iterations=mask_dilates).astype(np.uint8)
        else:
            mask_img = binary_mask(mask_img).astype(np.uint8)
        masks_dilated.append(Image.fromarray(mask_img * 255))

    if len(masks_img) == 1:
        flow_masks = flow_masks * length
        masks_dilated = masks_dilated * length

    return flow_masks, masks_dilated


def extrapolation(video_ori, scale):
    """Prepares the data for video outpainting.
    """
    nFrame = len(video_ori)
    imgW, imgH = video_ori[0].size

    # Defines new FOV.
    imgH_extr = int(scale[0] * imgH)
    imgW_extr = int(scale[1] * imgW)
    imgH_extr = imgH_extr - imgH_extr % 8
    imgW_extr = imgW_extr - imgW_extr % 8
    H_start = int((imgH_extr - imgH) / 2)
    W_start = int((imgW_extr - imgW) / 2)

    # Extrapolates the FOV for video.
    frames = []
    for v in video_ori:
        frame = np.zeros((imgH_extr, imgW_extr, 3), dtype=np.uint8)
        frame[H_start: H_start + imgH, W_start: W_start + imgW, :] = v
        frames.append(Image.fromarray(frame))

    # Generates the mask for missing region.
    masks_dilated = []
    flow_masks = []

    dilate_h = 4 if H_start > 10 else 0
    dilate_w = 4 if W_start > 10 else 0
    mask = np.ones(((imgH_extr, imgW_extr)), dtype=np.uint8)

    mask[H_start + dilate_h: H_start + imgH - dilate_h,
    W_start + dilate_w: W_start + imgW - dilate_w] = 0
    flow_masks.append(Image.fromarray(mask * 255))

    mask[H_start: H_start + imgH, W_start: W_start + imgW] = 0
    masks_dilated.append(Image.fromarray(mask * 255))

    flow_masks = flow_masks * nFrame
    masks_dilated = masks_dilated * nFrame

    return frames, flow_masks, masks_dilated, (imgW_extr, imgH_extr)


def get_ref_index(mid_neighbor_id, neighbor_ids, length, ref_stride=10, ref_num=-1):
    ref_index = []
    if ref_num == -1:
        for i in range(0, length, ref_stride):
            if i not in neighbor_ids:
                ref_index.append(i)
    else:
        start_idx = max(0, mid_neighbor_id - ref_stride * (ref_num // 2))
        end_idx = min(length, mid_neighbor_id + ref_stride * (ref_num // 2))
        for i in range(start_idx, end_idx, ref_stride):
            if i not in neighbor_ids:
                if len(ref_index) > ref_num:
                    break
                ref_index.append(i)
    return ref_index


class VideoInpaint:
    def __init__(self, sub_video_length=config.MAX_PROCESS_NUM, use_fp16=True):
        self.device = get_device()
        self.use_fp16 = use_fp16
        self.use_half = True if self.use_fp16 else False
        if self.device == torch.device('cpu'):
            self.use_half = False
        # Length of sub-video for long video inference.
        self.sub_video_length = sub_video_length
        # Length of local neighboring frames.'
        self.neighbor_length = 10
        # Mask dilation for video and flow masking
        self.mask_dilation = 4
        # Stride of global reference frames
        self.ref_stride = 10
        # Iterations for RAFT inference
        self.raft_iter = 20
        # Stride of global reference frames
        self.ref_stride = 10
        # 设置raft模型
        self.fix_raft = self.init_raft_model()
        # 设置fix_flow模型
        self.fix_flow_complete = self.init_fix_flow_model()
        # 设置inpaint模型
        self.model = self.init_inpaint_model()

    def init_raft_model(self):
        # set up RAFT and flow competition model
        return RAFT_bi(os.path.join(config.VIDEO_INPAINT_MODEL_PATH, 'raft-things.pth'), self.device)

    def init_fix_flow_model(self):
        fix_flow_complete_model = RecurrentFlowCompleteNet(
            os.path.join(config.VIDEO_INPAINT_MODEL_PATH, 'recurrent_flow_completion.pth'))
        for p in fix_flow_complete_model.parameters():
            p.requires_grad = False
        fix_flow_complete_model.to(self.device)
        fix_flow_complete_model.eval()
        return fix_flow_complete_model

    def init_inpaint_model(self):
        # set up ProPainter model
        return InpaintGenerator(model_path=os.path.join(config.VIDEO_INPAINT_MODEL_PATH, 'ProPainter.pth')).to(
            self.device).eval()

    def inpaint(self, frames, mask):
        if isinstance(frames[0], np.ndarray):
            frames = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]
        size = frames[0].size
        frames_len = len(frames)
        flow_masks, masks_dilated = read_mask(mask, frames_len, size,
                                              flow_mask_dilates=self.mask_dilation,
                                              mask_dilates=self.mask_dilation)
        w, h = size
        # for saving the masked frames or video
        masked_frame_for_save = []
        for i in range(len(frames)):
            mask_ = np.expand_dims(np.array(masks_dilated[i]), 2).repeat(3, axis=2) / 255.
            img = np.array(frames[i])
            green = np.zeros([h, w, 3])
            green[:, :, 1] = 255
            alpha = 0.6
            # alpha = 1.0
            fuse_img = (1 - alpha) * img + alpha * green
            fuse_img = mask_ * fuse_img + (1 - mask_) * img
            masked_frame_for_save.append(fuse_img.astype(np.uint8))

        frames_inp = [np.array(f).astype(np.uint8) for f in frames]
        frames = to_tensors()(frames).unsqueeze(0) * 2 - 1
        flow_masks = to_tensors()(flow_masks).unsqueeze(0)
        masks_dilated = to_tensors()(masks_dilated).unsqueeze(0)
        frames, flow_masks, masks_dilated = frames.to(self.device), flow_masks.to(self.device), masks_dilated.to(
            self.device)
        video_length = frames.size(1)
        with torch.no_grad():
            # ---- compute flow ----
            if frames.size(-1) <= 640:
                short_clip_len = 12
            elif frames.size(-1) <= 720:
                short_clip_len = 8
            elif frames.size(-1) <= 1280:
                short_clip_len = 4
            else:
                short_clip_len = 2

            # use fp32 for RAFT
            if frames.size(1) > short_clip_len:
                gt_flows_f_list, gt_flows_b_list = [], []
                for f in range(0, video_length, short_clip_len):
                    end_f = min(video_length, f + short_clip_len)
                    if f == 0:
                        flows_f, flows_b = self.fix_raft(frames[:, f:end_f], iters=self.raft_iter)
                    else:
                        flows_f, flows_b = self.fix_raft(frames[:, f - 1:end_f], iters=self.raft_iter)
                    gt_flows_f_list.append(flows_f)
                    gt_flows_b_list.append(flows_b)
                    torch.cuda.empty_cache()
                gt_flows_f = torch.cat(gt_flows_f_list, dim=1)
                gt_flows_b = torch.cat(gt_flows_b_list, dim=1)
                gt_flows_bi = (gt_flows_f, gt_flows_b)
            else:
                gt_flows_bi = self.fix_raft(frames, iters=self.raft_iter)
                torch.cuda.empty_cache()

            if self.use_half:
                frames, flow_masks, masks_dilated = frames.half(), flow_masks.half(), masks_dilated.half()
                gt_flows_bi = (gt_flows_bi[0].half(), gt_flows_bi[1].half())
                fix_flow_complete = self.fix_flow_complete.half()
                self.model = self.model.half()

            # ---- complete flow ----
            flow_length = gt_flows_bi[0].size(1)
            if flow_length > self.sub_video_length:
                pred_flows_f, pred_flows_b = [], []
                pad_len = 5
                for f in range(0, flow_length, self.sub_video_length):
                    s_f = max(0, f - pad_len)
                    e_f = min(flow_length, f + self.sub_video_length + pad_len)
                    pad_len_s = max(0, f) - s_f
                    pad_len_e = e_f - min(flow_length, f + self.sub_video_length)
                    pred_flows_bi_sub, _ = fix_flow_complete.forward_bidirect_flow(
                        (gt_flows_bi[0][:, s_f:e_f], gt_flows_bi[1][:, s_f:e_f]),
                        flow_masks[:, s_f:e_f + 1])
                    pred_flows_bi_sub = fix_flow_complete.combine_flow(
                        (gt_flows_bi[0][:, s_f:e_f], gt_flows_bi[1][:, s_f:e_f]),
                        pred_flows_bi_sub,
                        flow_masks[:, s_f:e_f + 1])

                    pred_flows_f.append(pred_flows_bi_sub[0][:, pad_len_s:e_f - s_f - pad_len_e])
                    pred_flows_b.append(pred_flows_bi_sub[1][:, pad_len_s:e_f - s_f - pad_len_e])
                    torch.cuda.empty_cache()

                pred_flows_f = torch.cat(pred_flows_f, dim=1)
                pred_flows_b = torch.cat(pred_flows_b, dim=1)
                pred_flows_bi = (pred_flows_f, pred_flows_b)
            else:
                pred_flows_bi, _ = fix_flow_complete.forward_bidirect_flow(gt_flows_bi, flow_masks)
                pred_flows_bi = fix_flow_complete.combine_flow(gt_flows_bi, pred_flows_bi, flow_masks)
                torch.cuda.empty_cache()

            # ---- image propagation ----
            masked_frames = frames * (1 - masks_dilated)
            # ensure a minimum of 100 frames for image propagation
            subvideo_length_img_prop = min(100, self.sub_video_length)
            if video_length > subvideo_length_img_prop:
                updated_frames, updated_masks = [], []
                pad_len = 10
                for f in range(0, video_length, subvideo_length_img_prop):
                    s_f = max(0, f - pad_len)
                    e_f = min(video_length, f + subvideo_length_img_prop + pad_len)
                    pad_len_s = max(0, f) - s_f
                    pad_len_e = e_f - min(video_length, f + subvideo_length_img_prop)

                    b, t, _, _, _ = masks_dilated[:, s_f:e_f].size()
                    pred_flows_bi_sub = (pred_flows_bi[0][:, s_f:e_f - 1], pred_flows_bi[1][:, s_f:e_f - 1])
                    prop_imgs_sub, updated_local_masks_sub = self.model.img_propagation(masked_frames[:, s_f:e_f],
                                                                                        pred_flows_bi_sub,
                                                                                        masks_dilated[:, s_f:e_f],
                                                                                        'nearest')
                    updated_frames_sub = frames[:, s_f:e_f] * (1 - masks_dilated[:, s_f:e_f]) + prop_imgs_sub.view(b, t, 3, h, w) * masks_dilated[:, s_f:e_f]
                    updated_masks_sub = updated_local_masks_sub.view(b, t, 1, h, w)
                    updated_frames.append(updated_frames_sub[:, pad_len_s:e_f - s_f - pad_len_e])
                    updated_masks.append(updated_masks_sub[:, pad_len_s:e_f - s_f - pad_len_e])
                    torch.cuda.empty_cache()

                updated_frames = torch.cat(updated_frames, dim=1)
                updated_masks = torch.cat(updated_masks, dim=1)
            else:
                b, t, _, _, _ = masks_dilated.size()
                prop_imgs, updated_local_masks = self.model.img_propagation(masked_frames, pred_flows_bi, masks_dilated,
                                                                       'nearest')
                updated_frames = frames * (1 - masks_dilated) + prop_imgs.view(b, t, 3, h, w) * masks_dilated
                updated_masks = updated_local_masks.view(b, t, 1, h, w)
                torch.cuda.empty_cache()

        ori_frames = frames_inp
        comp_frames = [None] * video_length

        neighbor_stride = self.neighbor_length // 2
        if video_length > self.sub_video_length:
            ref_num = self.sub_video_length // self.ref_stride
        else:
            ref_num = -1

        # ---- feature propagation + transformer ----
        for f in range(0, video_length, neighbor_stride):
            neighbor_ids = [
                i for i in range(max(0, f - neighbor_stride),
                                 min(video_length, f + neighbor_stride + 1))
            ]
            ref_ids = get_ref_index(f, neighbor_ids, video_length, self.ref_stride, ref_num)
            selected_imgs = updated_frames[:, neighbor_ids + ref_ids, :, :, :]
            selected_masks = masks_dilated[:, neighbor_ids + ref_ids, :, :, :]
            selected_update_masks = updated_masks[:, neighbor_ids + ref_ids, :, :, :]
            selected_pred_flows_bi = (
                pred_flows_bi[0][:, neighbor_ids[:-1], :, :, :], pred_flows_bi[1][:, neighbor_ids[:-1], :, :, :])

            with torch.no_grad():
                # 1.0 indicates mask
                l_t = len(neighbor_ids)
                pred_img = self.model(selected_imgs, selected_pred_flows_bi, selected_masks, selected_update_masks, l_t)
                pred_img = pred_img.view(-1, 3, h, w)
                pred_img = (pred_img + 1) / 2
                pred_img = pred_img.cpu().permute(0, 2, 3, 1).numpy() * 255
                binary_masks = masks_dilated[0, neighbor_ids, :, :, :].cpu().permute(
                    0, 2, 3, 1).numpy().astype(np.uint8)
                for i in range(len(neighbor_ids)):
                    idx = neighbor_ids[i]
                    img = np.array(pred_img[i]).astype(np.uint8) * binary_masks[i] \
                          + ori_frames[idx] * (1 - binary_masks[i])
                    if comp_frames[idx] is None:
                        comp_frames[idx] = img
                    else:
                        comp_frames[idx] = comp_frames[idx].astype(np.float32) * 0.5 + img.astype(np.float32) * 0.5
                    comp_frames[idx] = comp_frames[idx].astype(np.uint8)
            torch.cuda.empty_cache()
        # save videos frame
        comp_frames = [cv2.cvtColor(i, cv2.COLOR_RGB2BGR) for i in comp_frames]
        return comp_frames


def read_frames(v_path):
    video_cap = cv2.VideoCapture(v_path)
    video_frames = []
    while True:
        ret, frame = video_cap.read()
        if not ret:
            break
        video_frames.append(frame)
    video_frames = [Image.fromarray(f) for f in video_frames]
    return video_frames


if __name__ == '__main__':
    # VideoInpaint
    video_inpaint = VideoInpaint(sub_video_length=80)
    frames = read_frames('/home/yao/Documents/Project/video-subtitle-remover/local_test/test1.mp4')
    mask = cv2.imread('/home/yao/Documents/Project/video-subtitle-remover/local_test/test1_mask.png')
    inpainted_frames = video_inpaint.inpaint(frames, mask)
    save_root = '/home/yao/Documents/Project/video-subtitle-remover/local_test/'
    video_out_path = os.path.join(save_root, 'inpaint_out.mp4')
    print("size: ", inpainted_frames[0].shape)
    video_writer = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'mp4v'), 24, (640, 360))
    for comp_frame in inpainted_frames:
        video_writer.write(comp_frame)
    video_writer.release()
    print(f'\nAll results are saved in {save_root}')

