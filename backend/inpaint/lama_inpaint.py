import os
import gc
from typing import Union, List
import torch
import numpy as np
from PIL import Image
from backend.inpaint.utils.lama_util import prepare_img_and_mask, get_image, pad_img_to_modulo
from backend import config
from backend.tools.inpaint_tools import get_inpaint_area_by_mask

class LamaInpaint:
    def __init__(self, device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), model_path='big-lama.pt') -> None:
        self.model = torch.jit.load(model_path, map_location=device)
        self.model.eval()
        self.device = device

    def inpaint(self, image: Union[Image.Image, np.ndarray], mask: Union[Image.Image, np.ndarray]):
        if isinstance(image, np.ndarray):
            orig_height, orig_width = image.shape[:2]
        else:
            orig_height, orig_width = np.array(image).shape[:2]
        image, mask = prepare_img_and_mask(image, mask, self.device)
        with torch.inference_mode():
            inpainted = self.model(image, mask)
            cur_res = inpainted[0].permute(1, 2, 0).detach().cpu().numpy()
            cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
            cur_res = cur_res[:orig_height, :orig_width]
            return cur_res

    def _inpaint_batch(self, images: List[np.ndarray], masks: List[np.ndarray]):
        """批量推理：将多帧分小批次送入 GPU，避免单次推理过大导致卡死"""
        if len(images) == 1:
            return [self.inpaint(images[0], masks[0])]

        orig_height, orig_width = images[0].shape[:2]
        # 分小批次推理，每批最多 4 帧
        mini_batch_size = 4
        results = [None] * len(images)
        for start in range(0, len(images), mini_batch_size):
            end = min(start + mini_batch_size, len(images))
            batch_imgs = []
            batch_masks = []
            for i in range(start, end):
                batch_imgs.append(get_image(images[i]))
                batch_masks.append(get_image(masks[i]))

            padded_imgs = np.stack([pad_img_to_modulo(img, 8) for img in batch_imgs])
            padded_masks = np.stack([pad_img_to_modulo(m, 8) for m in batch_masks])

            img_tensor = torch.from_numpy(padded_imgs).to(self.device)
            mask_tensor = torch.from_numpy(padded_masks).to(self.device)
            mask_tensor = (mask_tensor > 0) * 1

            with torch.inference_mode():
                inpainted = self.model(img_tensor, mask_tensor)
                batch_results = inpainted.permute(0, 2, 3, 1).detach().cpu().numpy()
                batch_results = np.clip(batch_results * 255, 0, 255).astype('uint8')

            for i in range(end - start):
                results[start + i] = batch_results[i][:orig_height, :orig_width]

            del img_tensor, mask_tensor, padded_imgs, padded_masks
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return results

    def __call__(self, input_frames: List[np.ndarray], input_mask: np.ndarray):
        """
        :param input_frames: 原视频帧
        :param input_mask: 字幕区域mask
        """
        mask = input_mask[:, :, None]
        H_ori, W_ori = mask.shape[:2]
        H_ori = int(H_ori + 0.5)
        W_ori = int(W_ori + 0.5)
        # 确定去字幕的垂直高度部分
        split_h = int(W_ori * 3 / 16)
        inpaint_area = get_inpaint_area_by_mask(W_ori, H_ori, split_h, mask)
        # 高分辨率帧存储列表
        frames_hr = [f.copy() for f in input_frames]
        comps = {}  # 存放补全后帧的字典
        # 存储最终的视频帧
        inpainted_frames = []

        for k in range(len(inpaint_area)):
            # 收集该区域的所有裁剪帧和遮罩
            cropped_frames = []
            cropped_masks = []
            for j in range(len(frames_hr)):
                image_crop = frames_hr[j][inpaint_area[k][0]:inpaint_area[k][1], :, :]
                mask_crop = mask[inpaint_area[k][0]:inpaint_area[k][1], :, :]
                cropped_frames.append(image_crop)
                cropped_masks.append(mask_crop)

            # 批量推理
            comps[k] = self._inpaint_batch(cropped_frames, cropped_masks)
            del cropped_frames, cropped_masks
            gc.collect()

        # 如果存在去除部分
        if inpaint_area:
            for j in range(len(frames_hr)):
                frame = frames_hr[j]
                for k in range(len(inpaint_area)):
                    frame[inpaint_area[k][0]:inpaint_area[k][1], :, :] = comps[k][j]
                inpainted_frames.append(frame)
        else:
            # 无需处理的区域，返回原始帧
            inpainted_frames = frames_hr

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return inpainted_frames


