import os
import sys
from functools import cached_property

import cv2
from tqdm import tqdm

from .model_config import ModelConfig
from .hardware_accelerator import HardwareAccelerator
from .common_tools import get_readable_path
from .ocr import get_coordinates
from backend.config import config, tr
from backend.scenedetect import scene_detect
from backend.scenedetect.detectors import ContentDetector
from backend.tools.inpaint_tools import is_frame_number_in_ab_sections

class SubtitleDetect:
    """
    文本框检测类，用于检测视频帧中是否存在文本框
    """

    def __init__(self, video_path, sub_areas=[]):
        self.video_path = video_path
        self.sub_areas = sub_areas

    @cached_property
    def text_detector(self):
        import paddle
        paddle.disable_signal_handler()
        from paddleocr.tools.infer import utility
        from paddleocr.tools.infer.predict_det import TextDetector
        hardware_accelerator = HardwareAccelerator.instance()
        onnx_providers = hardware_accelerator.onnx_providers
        model_config = ModelConfig()
        parser = utility.init_args()
        args = parser.parse_args([])
        args.det_algorithm = 'DB'
        args.det_model_dir = os.path.join(model_config.DET_MODEL_DIR, 'inference.onnx') if len(onnx_providers) > 0 else model_config.DET_MODEL_DIR
        args.use_gpu=False
        args.use_onnx=len(onnx_providers) > 0
        args.onnx_providers=onnx_providers
        return TextDetector(args)

    def detect_subtitle(self, img):
        temp_list = []
        dt_boxes, elapse = self.text_detector(img)
        coordinate_list = get_coordinates(dt_boxes.tolist())
        if coordinate_list:
            for coordinate in coordinate_list:
                xmin, xmax, ymin, ymax = coordinate
                if self.sub_areas is not None and len(self.sub_areas) > 0:
                    for sub_area in self.sub_areas:
                        s_ymin, s_ymax, s_xmin, s_xmax = sub_area
                        if (s_xmin <= xmin and xmax <= s_xmax
                                and s_ymin <= ymin
                                and ymax <= s_ymax):
                            temp_list.append((xmin, xmax, ymin, ymax))
                else:
                    temp_list.append((xmin, xmax, ymin, ymax))
        return temp_list

    def find_subtitle_frame_no(self, sub_remover=None):
        video_cap = cv2.VideoCapture(get_readable_path(self.video_path))
        frame_count = video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
        tbar = tqdm(total=int(frame_count), unit='frame', position=0, file=sys.__stdout__, desc='Subtitle Finding')
        current_frame_no = 0
        subtitle_frame_no_box_dict = {}
        if sub_remover:
            sub_remover.append_output(tr['Main']['ProcessingStartFindingSubtitles'])
        while video_cap.isOpened():
            ret, frame = video_cap.read()
            # 如果读取视频帧失败（视频读到最后一帧）
            if not ret:
                break
            # 读取视频帧成功
            current_frame_no += 1
            if not is_frame_number_in_ab_sections(current_frame_no - 1, sub_remover.ab_sections):
                tbar.update(1)
                continue
            temp_list = self.detect_subtitle(frame)
            if len(temp_list) > 0:
                subtitle_frame_no_box_dict[current_frame_no] = temp_list
            tbar.update(1)
            if sub_remover:
                sub_remover.progress_total = (100 * float(current_frame_no) / float(frame_count)) // 2
        subtitle_frame_no_box_dict = self.unify_regions(subtitle_frame_no_box_dict)
        if sub_remover:
            sub_remover.append_output(tr['Main']['FinishedFindingSubtitles'])
        new_subtitle_frame_no_box_dict = dict()
        for key in subtitle_frame_no_box_dict.keys():
            if len(subtitle_frame_no_box_dict[key]) > 0:
                new_subtitle_frame_no_box_dict[key] = subtitle_frame_no_box_dict[key]
        return new_subtitle_frame_no_box_dict

    @staticmethod
    def split_range_by_scene(intervals, points):
        # 确保离散值列表是有序的
        points.sort()
        # 用于存储结果区间的列表
        result_intervals = []
        # 遍历区间
        for start, end in intervals:
            # 在当前区间内的点
            current_points = [p for p in points if start <= p <= end]

            # 遍历当前区间内的离散点
            for p in current_points:
                # 如果当前离散点不是区间的起始点，添加从区间开始到离散点前一个数字的区间
                if start < p:
                    result_intervals.append((start, p - 1))
                # 更新区间开始为当前离散点
                start = p
            # 添加从最后一个离散点或区间开始到区间结束的区间
            result_intervals.append((start, end))
        # 输出结果
        return result_intervals

    @staticmethod
    def get_scene_div_frame_no(v_path):
        """
        获取发生场景切换的帧号
        """
        scene_div_frame_no_list = []
        scene_list = scene_detect(v_path, ContentDetector())
        for scene in scene_list:
            start, end = scene
            if start.frame_num == 0:
                pass
            else:
                scene_div_frame_no_list.append(start.frame_num + 1)
        return scene_div_frame_no_list

    @staticmethod
    def are_similar(region1, region2):
        """判断两个区域是否相似。"""
        xmin1, xmax1, ymin1, ymax1 = region1
        xmin2, xmax2, ymin2, ymax2 = region2

        return abs(xmin1 - xmin2) <= config.subtitleAreaPixelToleranceXPixel.value and abs(xmax1 - xmax2) <= config.subtitleAreaPixelToleranceXPixel.value and \
            abs(ymin1 - ymin2) <= config.subtitleAreaPixelToleranceYPixel.value and abs(ymax1 - ymax2) <= config.subtitleAreaPixelToleranceYPixel.value

    def unify_regions(self, raw_regions):
        """将连续相似的区域统一，保持列表结构。"""
        if len(raw_regions) > 0:
            keys = sorted(raw_regions.keys())  # 对键进行排序以确保它们是连续的
            unified_regions = {}

            # 初始化
            last_key = keys[0]
            unify_value_map = {last_key: raw_regions[last_key]}

            for key in keys[1:]:
                current_regions = raw_regions[key]

                # 新增一个列表来存放匹配过的标准区间
                new_unify_values = []

                for idx, region in enumerate(current_regions):
                    last_standard_region = unify_value_map[last_key][idx] if idx < len(unify_value_map[last_key]) else None

                    # 如果当前的区间与前一个键的对应区间相似，我们统一它们
                    if last_standard_region and self.are_similar(region, last_standard_region):
                        new_unify_values.append(last_standard_region)
                    else:
                        new_unify_values.append(region)

                # 更新unify_value_map为最新的区间值
                unify_value_map[key] = new_unify_values
                last_key = key

            # 将最终统一后的结果传递给unified_regions
            for key in keys:
                unified_regions[key] = unify_value_map[key]
            return unified_regions
        else:
            return raw_regions

    @staticmethod
    def find_continuous_ranges(subtitle_frame_no_box_dict):
        """
        获取字幕出现的起始帧号与结束帧号
        """
        numbers = sorted(list(subtitle_frame_no_box_dict.keys()))
        ranges = []
        start = numbers[0]  # 初始区间开始值

        for i in range(1, len(numbers)):
            # 如果当前数字与前一个数字间隔超过1，
            # 则上一个区间结束，记录当前区间的开始与结束
            if numbers[i] - numbers[i - 1] != 1:
                end = numbers[i - 1]  # 则该数字是当前连续区间的终点
                ranges.append((start, end))
                start = numbers[i]  # 开始下一个连续区间
        # 添加最后一个区间
        ranges.append((start, numbers[-1]))
        return ranges

    @staticmethod
    def find_continuous_ranges_with_same_mask(subtitle_frame_no_box_dict):
        numbers = sorted(list(subtitle_frame_no_box_dict.keys()))
        ranges = []
        start = numbers[0]  # 初始区间开始值
        for i in range(1, len(numbers)):
            # 如果当前帧号与前一个帧号间隔超过1，
            # 则上一个区间结束，记录当前区间的开始与结束
            if numbers[i] - numbers[i - 1] != 1:
                end = numbers[i - 1]  # 则该数字是当前连续区间的终点
                ranges.append((start, end))
                start = numbers[i]  # 开始下一个连续区间
            # 如果当前帧号与前一个帧号间隔为1，且当前帧号对应的坐标点与上一帧号对应的坐标点不一致
            # 记录当前区间的开始与结束
            if numbers[i] - numbers[i - 1] == 1:
                if subtitle_frame_no_box_dict[numbers[i]] != subtitle_frame_no_box_dict[numbers[i - 1]]:
                    end = numbers[i - 1]  # 则该数字是当前连续区间的终点
                    ranges.append((start, end))
                    start = numbers[i]  # 开始下一个连续区间
        # 添加最后一个区间
        ranges.append((start, numbers[-1]))
        return ranges

    @staticmethod
    def filter_and_merge_intervals(intervals, target_length):
        """
        合并传入的字幕起始区间，确保区间大小最低为STTN_REFERENCE_LENGTH
        """
        expanded = []
        # 首先单独处理单点区间以扩展它们
        for start, end in intervals:
            if start == end:  # 单点区间
                # 扩展到接近的目标长度，但保证前后不重叠
                prev_end = expanded[-1][1] if expanded else float('-inf')
                next_start = float('inf')
                # 查找下一个区间的起始点
                for ns, ne in intervals:
                    if ns > end:
                        next_start = ns
                        break
                # 确定新的扩展起点和终点
                new_start = max(start - (target_length - 1) // 2, prev_end + 1)
                new_end = min(start + (target_length - 1) // 2, next_start - 1)
                # 如果新的扩展终点在起点前面，说明没有足够空间来进行扩展
                if new_end < new_start:
                    new_start, new_end = start, start  # 保持原样
                expanded.append((new_start, new_end))
            else:
                # 非单点区间直接保留，稍后处理任何可能的重叠
                expanded.append((start, end))
        # 排序以合并那些因扩展导致重叠的区间
        expanded.sort(key=lambda x: x[0])
        # 合并重叠的区间，但仅当它们之间真正重叠且小于目标长度时
        merged = [expanded[0]]
        for start, end in expanded[1:]:
            last_start, last_end = merged[-1]
            # 检查是否重叠
            if start <= last_end and (end - last_start + 1 < target_length or last_end - last_start + 1 < target_length):
                # 需要合并
                merged[-1] = (last_start, max(last_end, end))  # 合并区间
            elif start == last_end + 1 and (end - last_start + 1 < target_length or last_end - last_start + 1 < target_length):
                # 相邻区间也需要合并的场景
                merged[-1] = (last_start, end)
            else:
                # 如果没有重叠且都大于目标长度，则直接保留
                merged.append((start, end))
        return merged
