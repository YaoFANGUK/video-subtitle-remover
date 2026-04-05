import os
import copy
from typing import Union, List
import torch
import numpy as np
from PIL import Image
from backend.inpaint.utils.lama_util import prepare_img_and_mask
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
        # 初始化帧存储变量
        # 高分辨率帧存储列表
        frames_hr = copy.deepcopy(input_frames)
        frames_scaled = {}  # 存放缩放后帧的字典
        masks_scaled = {}  # 存放缩放后遮罩的字典
        comps = {}  # 存放补全后帧的字典
        # 存储最终的视频帧
        inpainted_frames = []
        for k in range(len(inpaint_area)):
            frames_scaled[k] = []  # 为每个去除部分初始化一个列表
            masks_scaled[k] = []  # 为每个去除部分初始化一个列表

        # 读取并缩放帧
        for j in range(len(frames_hr)):
            image = frames_hr[j]
            # 对每个去除部分进行切割和缩放
            for k in range(len(inpaint_area)):
                image_crop = image[inpaint_area[k][0]:inpaint_area[k][1], :, :]  # 切割
                mask_crop = mask[inpaint_area[k][0]:inpaint_area[k][1], :, :]  # 切割
                frames_scaled[k].append(image_crop)  # 将切割后的帧添加到对应列表
                masks_scaled[k].append(mask_crop)  # 将切割后的遮罩添加到对应列表

        # 处理每一个去除部分
        for k in range(len(inpaint_area)):
            # 调用inpaint函数逐帧处理
            comps[k] = []
            for i in range(len(frames_scaled[k])):
                inpainted_frame = self.inpaint(frames_scaled[k][i], masks_scaled[k][i])
                comps[k].append(inpainted_frame)

        # 如果存在去除部分
        if inpaint_area:
            for j in range(len(frames_hr)):
                frame = frames_hr[j]  # 取出原始帧
                # 对于模式中的每一个段落
                for k in range(len(inpaint_area)):
                    comp = comps[k][j]  # 获取补全后的帧
                    # 实现遮罩区域内的图像融合
                    frame[inpaint_area[k][0]:inpaint_area[k][1], :, :] = comp
                # 将最终帧添加到列表
                inpainted_frames.append(frame)
                # print(f'processing frame, {len(frames_hr) - j} left')
        return inpainted_frames


