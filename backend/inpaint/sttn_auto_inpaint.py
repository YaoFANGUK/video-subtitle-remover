import os
import copy
import time
import sys
from typing import List

import cv2
import torch
import numpy as np
from tqdm import tqdm
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from backend.config import config
from backend.inpaint.sttn.auto_sttn import InpaintGenerator
from backend.inpaint.utils.sttn_utils import Stack, ToTorchFormatTensor
from backend.tools.inpaint_tools import get_inpaint_area_by_mask, is_frame_number_in_ab_sections

# 定义图像预处理方式
_to_tensors = transforms.Compose([
    Stack(),  # 将图像堆叠为序列
    ToTorchFormatTensor()  # 将堆叠的图像转化为PyTorch张量
])

class STTNInpaint:
    def __init__(self, device, model_path):
        self.device = device
        # 1. 创建InpaintGenerator模型实例并装载到选择的设备上
        self.model = InpaintGenerator().to(self.device)
        # 2. 载入预训练模型的权重，转载模型的状态字典
        self.model.load_state_dict(torch.load(model_path, map_location='cpu')['netG'])
        # 3. # 将模型设置为评估模式
        self.model.eval()
        # 模型输入用的宽和高
        self.model_input_width, self.model_input_height = 640, 120
        # 2. 设置相连帧数
        self.neighbor_stride = config.sttnNeighborStride.value
        self.ref_length = config.sttnReferenceLength.value

    def __call__(self, input_frames: List[np.ndarray], input_mask: np.ndarray):
        """
        :param input_frames: 原视频帧
        :param mask: 字幕区域mask
        """
        _, mask = cv2.threshold(input_mask, 127, 1, cv2.THRESH_BINARY)
        mask = mask[:, :, None]
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
        comps = {}  # 存放补全后帧的字典
        # 存储最终的视频帧
        inpainted_frames = []
        for k in range(len(inpaint_area)):
            frames_scaled[k] = []  # 为每个去除部分初始化一个列表

        # 读取并缩放帧
        for j in range(len(frames_hr)):
            image = frames_hr[j]
            # 对每个去除部分进行切割和缩放
            for k in range(len(inpaint_area)):
                image_crop = image[inpaint_area[k][0]:inpaint_area[k][1], :, :]  # 切割
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
                    frame[inpaint_area[k][0]:inpaint_area[k][1], :, :] = mask_area * comp + (1 - mask_area) * frame[inpaint_area[k][0]:inpaint_area[k][1], :, :]
                # 将最终帧添加到列表
                inpainted_frames.append(frame)
                # print(f'processing frame, {len(frames_hr) - j} left')
        return inpainted_frames

    @staticmethod
    def read_mask(path):
        img = cv2.imread(path, 0)
        # 转为binary mask
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


class STTNAutoInpaint:

    def read_frame_info_from_video(self):
        # 使用opencv读取视频
        reader = cv2.VideoCapture(self.video_path)
        # 获取视频的宽度, 高度, 帧率和帧数信息并存储在frame_info字典中
        frame_info = {
            'W_ori': int(reader.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5),  # 视频的原始宽度
            'H_ori': int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5),  # 视频的原始高度
            'fps': reader.get(cv2.CAP_PROP_FPS),  # 视频的帧率
            'len': int(reader.get(cv2.CAP_PROP_FRAME_COUNT) + 0.5)  # 视频的总帧数
        }
        # 返回视频读取对象、帧信息和视频写入对象
        return reader, frame_info

    def __init__(self, device, model_path, video_path, mask_path=None, clip_gap=None):
        # STTNInpaint视频修复实例初始化
        self.sttn_inpaint = STTNInpaint(device, model_path)
        # 视频和掩码路径
        self.video_path = video_path
        self.mask_path = mask_path
        # 设置输出视频文件的路径
        self.video_out_path = os.path.join(
            os.path.dirname(os.path.abspath(self.video_path)),
            f"{os.path.basename(self.video_path).rsplit('.', 1)[0]}_no_sub.mp4"
        )
        # 配置可在一次处理中加载的最大帧数
        if clip_gap is None:
            self.clip_gap = config.getSttnMaxLoadNum()
        else:
            self.clip_gap = clip_gap

    def __call__(self, input_mask=None, input_sub_remover=None, tbar=None):
        reader = None
        writer = None
        try:
            # 读取视频帧信息
            reader, frame_info = self.read_frame_info_from_video()
            if input_sub_remover is not None:
                ab_sections = input_sub_remover.ab_sections
                
                writer = input_sub_remover.video_writer
            else:
                ab_sections = None
                # 创建视频写入对象，用于输出修复后的视频
                writer = cv2.VideoWriter(self.video_out_path, cv2.VideoWriter_fourcc(*"mp4v"), frame_info['fps'], (frame_info['W_ori'], frame_info['H_ori']))
            
            # 计算需要迭代修复视频的次数
            rec_time = frame_info['len'] // self.clip_gap if frame_info['len'] % self.clip_gap == 0 else frame_info['len'] // self.clip_gap + 1
            # 计算分割高度，用于确定修复区域的大小
            split_h = int(frame_info['W_ori'] * 3 / 16)
            
            if input_mask is None:
                # 读取掩码
                mask = self.sttn_inpaint.read_mask(self.mask_path)
            else:
                _, mask = cv2.threshold(input_mask, 127, 1, cv2.THRESH_BINARY)
                mask = mask[:, :, None]
                
            # 得到修复区域位置
            inpaint_area = get_inpaint_area_by_mask(frame_info['W_ori'], frame_info['H_ori'], split_h, mask)
            # 遍历每一次的迭代次数
            for i in range(rec_time):
                start_f = i * self.clip_gap  # 起始帧位置
                end_f = min((i + 1) * self.clip_gap, frame_info['len'])  # 结束帧位置
                tqdm.write(f'Processing: {start_f + 1} - {end_f} / Total: {frame_info['len']}')
                
                frames_hr = []  # 高分辨率帧列表
                frames = {}  # 帧字典，用于存储裁剪后的图像
                comps = {}  # 组合字典，用于存储修复后的图像
                
                # 初始化帧字典
                for k in range(len(inpaint_area)):
                    frames[k] = []
                    
                # 读取和修复高分辨率帧
                valid_frames_count = 0
                for j in range(start_f, end_f):
                    success, image = reader.read()
                    if not success:
                        print(f"Warning: Failed to read frame {j}.")
                        break
                    
                    frames_hr.append(image)
                    valid_frames_count += 1
                    
                    if is_frame_number_in_ab_sections(j, ab_sections):
                        for k in range(len(inpaint_area)):
                            # 裁剪、缩放并添加到帧字典
                            image_crop = image[inpaint_area[k][0]:inpaint_area[k][1], :, :]
                            image_resize = cv2.resize(image_crop, (self.sttn_inpaint.model_input_width, self.sttn_inpaint.model_input_height))
                            frames[k].append(image_resize)
                
                # 如果没有读取到有效帧，则跳过当前迭代
                if valid_frames_count == 0:
                    print(f"Warning: No valid frames found in range {start_f+1}-{end_f}. Skipping this segment.")
                    continue
                    
                # 对每个修复区域运行修复
                for k in range(len(inpaint_area)):
                    if len(frames[k]) > 0:  # 确保有帧可以处理
                        comps[k] = self.sttn_inpaint.inpaint(frames[k])
                    else:
                        comps[k] = []
                
                # 如果有要修复的区域
                if inpaint_area and valid_frames_count > 0:
                    # 创建一个映射，记录哪些帧被处理了以及它们在frames[k]中的索引
                    processed_frames_map = {}
                    processed_idx = 0
                    
                    # 构建映射关系
                    for j in range(start_f, end_f):
                        if j - start_f < valid_frames_count and is_frame_number_in_ab_sections(j, ab_sections):
                            processed_frames_map[j - start_f] = processed_idx
                            processed_idx += 1
                    
                    # 应用修复结果
                    for j in range(valid_frames_count):
                        if input_sub_remover is not None and input_sub_remover.gui_mode:
                            original_frame = copy.deepcopy(frames_hr[j])
                        else:
                            original_frame = None
                            
                        frame = frames_hr[j]
                        
                        # 只有被处理过的帧才应用修复结果
                        if j in processed_frames_map:
                            comp_idx = processed_frames_map[j]
                            for k in range(len(inpaint_area)):
                                if comp_idx < len(comps[k]):  # 确保索引有效
                                    # 将修复的图像重新扩展到原始分辨率，并融合到原始帧
                                    comp = cv2.resize(comps[k][comp_idx], (frame_info['W_ori'], split_h))
                                    comp = cv2.cvtColor(np.array(comp).astype(np.uint8), cv2.COLOR_BGR2RGB)
                                    mask_area = mask[inpaint_area[k][0]:inpaint_area[k][1], :]
                                    frame[inpaint_area[k][0]:inpaint_area[k][1], :, :] = mask_area * comp + (1 - mask_area) * frame[inpaint_area[k][0]:inpaint_area[k][1], :, :]
                        
                        writer.write(frame)
                        
                        if input_sub_remover is not None:
                            if tbar is not None:
                                input_sub_remover.update_progress(tbar, increment=1)
                            if original_frame is not None and input_sub_remover.gui_mode:
                                input_sub_remover.update_preview_with_comp(original_frame, frame)
        except Exception as e:
            print(f"Error during video processing: {str(e)}")
            # 不抛出异常，允许程序继续执行
        finally:
            if writer:
                writer.release()


if __name__ == '__main__':
    mask_path = '../../test/test.png'
    video_path = '../../test/test.mp4'
    # 记录开始时间
    start = time.time()
    sttn_video_inpaint = STTNAutoInpaint(video_path, mask_path, clip_gap=config.getSttnMaxLoadNum())
    sttn_video_inpaint()
    print(f'video generated at {sttn_video_inpaint.video_out_path}')
    print(f'time cost: {time.time() - start}')
