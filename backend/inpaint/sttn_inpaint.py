import copy
import cv2
import numpy as np
import torch
from torchvision import transforms
from typing import List

from backend import config
from backend.inpaint.sttn.auto_sttn import InpaintGenerator
from backend.inpaint.utils.sttn_utils import Stack, ToTorchFormatTensor

# 定义图像预处理方式
_to_tensors = transforms.Compose([
    Stack(),  # 将图像堆叠为序列
    ToTorchFormatTensor()  # 将堆叠的图像转化为PyTorch张量
])


class STTNInpaint:
    def __init__(self):
        self.device = config.device
        # 1. 创建InpaintGenerator模型实例并装载到选择的设备上
        self.model = InpaintGenerator().to(self.device)
        # 2. 载入预训练模型的权重，转载模型的状态字典
        self.model.load_state_dict(torch.load(config.STTN_MODEL_PATH, map_location=self.device)['netG'])
        # 3. # 将模型设置为评估模式
        self.model.eval()
        # 模型输入用的宽和高
        self.model_input_width, self.model_input_height = 640, 120
        # 2. 设置相连帧数
        self.neighbor_stride = 5
        self.ref_length = 5

    def __call__(self, frames: List[np.ndarray], mask: np.ndarray):
        """
        :param frames: 原视频帧
        :param mask: 字幕区域mask
        """
        H_ori, W_ori = mask.shape[:2]
        H_ori = int(H_ori + 0.5)
        W_ori = int(W_ori + 0.5)
        # 确定去字幕的垂直高度部分
        split_h = int(W_ori * 3 / 16)
        inpaint_area = self.get_inpaint_area_by_mask(H_ori, split_h, mask)
        # 初始化帧存储变量
        # 高分辨率帧存储列表
        frames_hr = copy.deepcopy(frames)
        frames_scaled = {}  # 存放缩放后帧的字典
        comps = {}  # 存放补全后帧的字典
        # 存储最终的视频帧
        inpainted_frames = []
        for k in range(len(inpaint_area)):
            frames_scaled[k] = []  # 为每个去除部分初始化一个列表

        # 读取并缩放帧
        for frame_hr in frames_hr:
            # 对每个去除部分进行切割和缩放
            for k in range(len(inpaint_area)):
                image_crop = frame_hr[inpaint_area[k][0]:inpaint_area[k][1], :, :]  # 切割
                image_resize = cv2.resize(image_crop, (self.model_input_width, self.model_input_height))  # 缩放
                frames_scaled[k].append(image_resize)  # 将缩放后的帧添加到对应列表

        # 处理每一个去除部分
        for k in range(len(inpaint_area)):
            # 调用inpaint函数进行处理
            comps[k] = self.inpaint(frames_scaled[k])

        # 如果存在去除部分
        if inpaint_area:
            for j in range(len(frames_hr)):
                frame = frames_hr[j]  # 取出原始帧
                # 对于模式中的每一个段落
                for k in range(len(inpaint_area)):
                    comp = cv2.resize(comps[k][j], (W_ori, split_h))  # 将补全帧缩放回原大小
                    comp = cv2.cvtColor(np.array(comp).astype(np.uint8), cv2.COLOR_BGR2RGB)  # 转换颜色空间
                    # 获取遮罩区域并进行图像合成
                    mask_area = mask[inpaint_area[k][0]:inpaint_area[k][1], :]  # 取出遮罩区域
                    # 实现遮罩区域内的图像融合
                    frame[inpaint_area[k][0]:inpaint_area[k][1], :, :] = mask_area * comp + \
                                                                         (1 - mask_area) * frame[
                                                                                           inpaint_area[k][0]:
                                                                                           inpaint_area[k][1], :, :]
                # 将最终帧添加到列表
                print(f'processing frame, {len(frames_hr) - j} left')
                inpainted_frames.append(frame)
        return inpainted_frames

    @staticmethod
    def read_mask(path):
        img = cv2.imread(path, 0)
        ret, img = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)
        img = img[:, :, None]
        return img

    def get_ref_index(self, neighbor_ids, length):
        """
        采样整个视频的参考帧
        """
        # 初始化参考帧的索引列表
        ref_index = []
        # 在视频长度范围内根据ref_length逐步迭代
        for i in range(0, length, self.ref_length):
            # 如果当前帧不在近邻帧中
            if i not in neighbor_ids:
                # 将它添加到参考帧列表
                ref_index.append(i)
        # 返回参考帧索引列表
        return ref_index

    def inpaint(self, frames: List[np.ndarray]):
        """
        使用STTN完成空洞填充（空洞即被遮罩的区域）
        """
        frame_length = len(frames)
        # 对帧进行预处理转换为张量，并进行归一化
        feats = _to_tensors(frames).unsqueeze(0) * 2 - 1
        # 把特征张量转移到指定的设备（CPU或GPU）
        feats = feats.to(self.device)
        # 初始化一个与视频长度相同的列表，用于存储处理完成的帧
        comp_frames = [None] * frame_length
        # 关闭梯度计算，用于推理阶段节省内存并加速
        with torch.no_grad():
            # 将处理好的帧通过编码器，产生特征表示
            feats = self.model.encoder(feats.view(frame_length, 3, self.model_input_height, self.model_input_width))
            # 获取特征维度信息
            _, c, feat_h, feat_w = feats.size()
            # 调整特征形状以匹配模型的期望输入
            feats = feats.view(1, frame_length, c, feat_h, feat_w)
        # 获取重绘区域
        # 在设定的邻居帧步幅内循环处理视频
        for f in range(0, frame_length, self.neighbor_stride):
            # 计算邻近帧的ID
            neighbor_ids = [i for i in range(max(0, f - self.neighbor_stride), min(frame_length, f + self.neighbor_stride + 1))]
            # 获取参考帧的索引
            ref_ids = self.get_ref_index(neighbor_ids, frame_length)
            # 同样关闭梯度计算
            with torch.no_grad():
                # 通过模型推断特征并传递给解码器以生成完成的帧
                pred_feat = self.model.infer(feats[0, neighbor_ids + ref_ids, :, :, :])
                # 将预测的特征通过解码器生成图片，并应用激活函数tanh，然后分离出张量
                pred_img = torch.tanh(self.model.decoder(pred_feat[:len(neighbor_ids), :, :, :])).detach()
                # 将结果张量重新缩放到0到255的范围内（图像像素值）
                pred_img = (pred_img + 1) / 2
                # 将张量移动回CPU并转为NumPy数组
                pred_img = pred_img.cpu().permute(0, 2, 3, 1).numpy() * 255
                # 遍历邻近帧
                for i in range(len(neighbor_ids)):
                    idx = neighbor_ids[i]
                    # 将预测的图片转换为无符号8位整数格式
                    img = np.array(pred_img[i]).astype(np.uint8)
                    if comp_frames[idx] is None:
                        # 如果该位置为空，则赋值为新计算出的图片
                        comp_frames[idx] = img
                    else:
                        # 如果此位置之前已有图片，则将新旧图片混合以提高质量
                        comp_frames[idx] = comp_frames[idx].astype(np.float32) * 0.5 + img.astype(np.float32) * 0.5
        # 返回处理完成的帧序列
        return comp_frames

    @staticmethod
    def get_inpaint_area_by_mask(H, h, mask):
        """
        获取字幕去除区域，根据mask来确定需要填补的区域和高度
        """
        # 存储绘画区域的列表
        inpaint_area = []
        # 从视频底部的字幕位置开始，假设字幕通常位于底部
        to_H = from_H = H
        # 从底部向上遍历遮罩
        while from_H != 0:
            if to_H - h < 0:
                # 如果下一段会超出顶端，则从顶端开始
                from_H = 0
                to_H = h
            else:
                # 确定段的上边界
                from_H = to_H - h
            # 检查当前段落是否包含遮罩像素
            if not np.all(mask[from_H:to_H, :] == 0) and np.sum(mask[from_H:to_H, :]) > 10:
                # 如果不是第一个段落，向下移动以确保没遗漏遮罩区域
                if to_H != H:
                    move = 0
                    while to_H + move < H and not np.all(mask[to_H + move, :] == 0):
                        move += 1
                    # 确保没有越过底部
                    if to_H + move < H and move < h:
                        to_H += move
                        from_H += move
                # 将该段落添加到列表中
                inpaint_area.append((from_H, to_H))
            # 移动到下一个段落
            to_H -= h
        return inpaint_area  # 返回绘画区域列表


if __name__ == '__main__':
    sttn_inpaint = STTNInpaint()
    video_path = '/home/yao/Documents/Project/video-subtitle-remover/local_test/english1.mp4'
    mask_path = '/home/yao/Documents/Project/video-subtitle-remover/local_test/english1_mask.png'
    video_cap = cv2.VideoCapture(video_path)
    mask = sttn_inpaint.read_mask(mask_path)
    input_frames = []
    index = 0
    print('读取视频帧')
    while True:
        ret, frame = video_cap.read()
        if not ret:
            break
        if index == 200:
            break
        index += 1
        input_frames.append(frame)
    print('开始填充')
    inpainted_frames = sttn_inpaint(input_frames, mask)
    for i,frame in enumerate(inpainted_frames):
        cv2.imwrite(f"/home/yao/Documents/Project/video-subtitle-remover/local_test/res/{i}.png", frame)

