import os
import json
import random
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from backend.tools.train.utils_sttn import ZipReader, create_random_shape_with_random_motion
from backend.tools.train.utils_sttn import Stack, ToTorchFormatTensor, GroupRandomHorizontalFlip


# 自定义的数据集
class Dataset(torch.utils.data.Dataset):
    def __init__(self, args: dict, split='train', debug=False):
        # 初始化函数，传入配置参数字典，数据集划分类型，默认为'train'
        self.args = args
        self.split = split
        self.sample_length = args['sample_length']  # 样本长度参数
        self.size = self.w, self.h = (args['w'], args['h'])  # 设置图像的目标宽高

        # 打开存放数据相关信息的json文件
        with open(os.path.join(args['data_root'], args['name'], split+'.json'), 'r') as f:
            self.video_dict = json.load(f)  # 加载json文件内容
        self.video_names = list(self.video_dict.keys())  # 获取视频的名称列表
        if debug or split != 'train':  # 如果是调试模式或者不是训练集，只取前100个视频
            self.video_names = self.video_names[:100]

        # 定义数据的转换操作，转换成堆叠的张量
        self._to_tensors = transforms.Compose([
            Stack(),
            ToTorchFormatTensor(),  # 便于在PyTorch中使用的张量格式
        ])

    def __len__(self):
        # 返回数据集中视频的数量
        return len(self.video_names)

    def __getitem__(self, index):
        # 获取一个样本项
        try:
            item = self.load_item(index)  # 尝试加载指定索引的数据项
        except:
            print('Loading error in video {}'.format(self.video_names[index]))  # 如果加载出错，打印出错信息
            item = self.load_item(0)  # 加载第一个项目作为兜底
        return item

    def load_item(self, index):
        # 加载数据项的具体实现
        video_name = self.video_names[index]  # 根据索引获取视频名称
        # 为所有视频帧生成帧文件名列表
        all_frames = [f"{str(i).zfill(5)}.jpg" for i in range(self.video_dict[video_name])]
        # 生成随机运动的随机形状的遮罩
        all_masks = create_random_shape_with_random_motion(
            len(all_frames), imageHeight=self.h, imageWidth=self.w)
        # 获取参考帧的索引
        ref_index = get_ref_index(len(all_frames), self.sample_length)
        # 读取视频帧
        frames = []
        masks = []
        for idx in ref_index:
            # 读取图片，转化为RGB，调整大小并添加到列表中
            img = ZipReader.imread('{}/{}/JPEGImages/{}.zip'.format(
                self.args['data_root'], self.args['name'], video_name), all_frames[idx]).convert('RGB')
            img = img.resize(self.size)
            frames.append(img)
            masks.append(all_masks[idx])
        if self.split == 'train':
            # 如果是训练集，随机水平翻转图像
            frames = GroupRandomHorizontalFlip()(frames)
        # 转换成张量形式
        frame_tensors = self._to_tensors(frames)*2.0 - 1.0  # 归一化处理
        mask_tensors = self._to_tensors(masks)  # 将遮罩转换成张量
        return frame_tensors, mask_tensors  # 返回图像和遮罩的张量


def get_ref_index(length, sample_length):
    # 获取参考帧索引的实现
    if random.uniform(0, 1) > 0.5:
        # 有一半的概率随机选择帧
        ref_index = random.sample(range(length), sample_length)
        ref_index.sort()  # 排序保证顺序
    else:
        # 另一半概率选择连续的帧
        pivot = random.randint(0, length-sample_length)
        ref_index = [pivot+i for i in range(sample_length)]
    return ref_index
