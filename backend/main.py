import shutil
import subprocess
import random
import config
import os
from pathlib import Path
import threading
import cv2
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import importlib
import numpy as np
import tempfile
from tqdm import tqdm
from tools.infer import utility
from tools.infer.predict_det import TextDetector
from inpaint.lama_inpaint import inpaint_img_with_lama


class SubtitleDetect:
    """
    文本框检测类，用于检测视频帧中是否存在文本框
    """

    def __init__(self, video_path, sub_area=None):
        # 获取参数对象
        importlib.reload(config)
        args = utility.parse_args()
        args.det_algorithm = 'DB'
        args.det_model_dir = config.DET_MODEL_PATH
        self.text_detector = TextDetector(args)
        self.video_path = video_path
        self.sub_area = sub_area

    def detect_subtitle(self, img):
        dt_boxes, elapse = self.text_detector(img)
        return dt_boxes, elapse

    @staticmethod
    def get_coordinates(dt_box):
        """
        从返回的检测框中获取坐标
        :param dt_box 检测框返回结果
        :return list 坐标点列表
        """
        coordinate_list = list()
        if isinstance(dt_box, list):
            for i in dt_box:
                i = list(i)
                (x1, y1) = int(i[0][0]), int(i[0][1])
                (x2, y2) = int(i[1][0]), int(i[1][1])
                (x3, y3) = int(i[2][0]), int(i[2][1])
                (x4, y4) = int(i[3][0]), int(i[3][1])
                xmin = max(x1, x4)
                xmax = min(x2, x3)
                ymin = max(y1, y2)
                ymax = min(y3, y4)
                coordinate_list.append((xmin, xmax, ymin, ymax))
        return coordinate_list

    def find_subtitle_frame_no(self):
        video_cap = cv2.VideoCapture(self.video_path)
        frame_count = video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
        tbar = tqdm(total=int(frame_count), unit='f', position=0, file=sys.__stdout__, desc='字幕查找')
        current_frame_no = 0
        subtitle_frame_no_list = {}

        while video_cap.isOpened():
            ret, frame = video_cap.read()
            # 如果读取视频帧失败（视频读到最后一帧）
            if not ret:
                break
            # 读取视频帧成功
            current_frame_no += 1
            dt_boxes, elapse = self.detect_subtitle(frame)
            coordinate_list = self.get_coordinates(dt_boxes.tolist())
            if coordinate_list:
                temp_list = []
                for coordinate in coordinate_list:
                    xmin, xmax, ymin, ymax = coordinate
                    if self.sub_area is not None:
                        s_ymin, s_ymax, s_xmin, s_xmax = self.sub_area
                        if (s_xmin <= xmin and xmax <= s_xmax
                            and s_ymin <= ymin
                            and ymax <= s_ymax):
                            temp_list.append((xmin, xmax, ymin, ymax))
                    else:
                        temp_list.append((xmin, xmax, ymin, ymax))
                subtitle_frame_no_list[current_frame_no] = temp_list
            tbar.update(1)
        return subtitle_frame_no_list


class SubtitleRemover:
    def __init__(self, vd_path, sub_area=None):
        importlib.reload(config)
        # 线程锁
        self.lock = threading.RLock()
        uln = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
        # 用户指定的字幕区域位置
        self.sub_area = sub_area
        # 视频路径
        self.video_path = vd_path
        self.video_cap = cv2.VideoCapture(vd_path)
        # 通过视频路径获取视频名称
        self.vd_name = Path(self.video_path).stem
        # 视频帧总数
        self.frame_count = self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
        # 视频帧率
        self.fps = self.video_cap.get(cv2.CAP_PROP_FPS)
        # 视频尺寸
        self.size = (int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.frame_height = int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_width = int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        # 创建字幕检测对象
        self.sub_detector = SubtitleDetect(self.video_path, self.sub_area)
        # 创建视频写对象
        self.video_temp_out_name = os.path.join(os.path.dirname(self.video_path), f'{self.vd_name}_{"".join(random.sample(uln, 8))}.mp4')
        self.video_writer = cv2.VideoWriter(self.video_temp_out_name, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, self.size)
        self.video_out_name = os.path.join(os.path.dirname(self.video_path), f'{self.vd_name}_no_sub.mp4')


    @staticmethod
    def get_coordinates(dt_box):
        """
        从返回的检测框中获取坐标
        :param dt_box 检测框返回结果
        :return list 坐标点列表
        """
        coordinate_list = list()
        if isinstance(dt_box, list):
            for i in dt_box:
                i = list(i)
                (x1, y1) = int(i[0][0]), int(i[0][1])
                (x2, y2) = int(i[1][0]), int(i[1][1])
                (x3, y3) = int(i[2][0]), int(i[2][1])
                (x4, y4) = int(i[3][0]), int(i[3][1])
                xmin = max(x1, x4)
                xmax = min(x2, x3)
                ymin = max(y1, y2)
                ymax = min(y3, y4)
                coordinate_list.append((xmin, xmax, ymin, ymax))
        return coordinate_list

    def run(self):
        # 寻找字幕帧
        sub_list = self.sub_detector.find_subtitle_frame_no()
        index = 0
        tbar = tqdm(total=int(self.frame_count), unit='f', position=0, file=sys.__stdout__, desc='字幕去除')
        while True:
            ret, frame = self.video_cap.read()
            if not ret:
                break
            index += 1
            if index in sub_list:
                masks = self.create_mask(frame, sub_list[index])
                frame = self.inpaint_frame(frame, masks)
            self.video_writer.write(frame)
            tbar.update(1)
        self.video_cap.release()
        self.video_writer.release()
        # 将原音频合并到新生成的视频文件中
        self.merge_audio_to_video()
        print(f"视频生字幕去除成功，文件路径：{self.video_out_name}")

    @staticmethod
    def inpaint( img, mask):
        img_inpainted = inpaint_img_with_lama(img, mask, config.LAMA_CONFIG, config.LAMA_MODEL_PATH, device=config.device)
        return img_inpainted

    def inpaint_frame(self, censored_img, mask_list):
        inpainted_frame = censored_img
        if mask_list:
            for mask in mask_list:
                inpainted_frame = self.inpaint(inpainted_frame, mask)
        return inpainted_frame

    @staticmethod
    def create_mask(input_img, coords_list):
        masks = []
        if coords_list:
            for coords in coords_list:
                mask = np.zeros(input_img.shape[0:2], dtype="uint8")
                xmin, xmax, ymin, ymax = coords
                # 为了避免框过小，放大10个像素
                cv2.rectangle(mask, (xmin - 10, ymin - 10), (xmax + 10, ymax + 10), (255, 255, 255), thickness=-1)
                masks.append(mask)
        return masks

    def merge_audio_to_video(self):
        temp = tempfile.NamedTemporaryFile(suffix='.aac', delete=False)
        audio_extract_command = [config.FFMPEG_PATH,
                                 "-y", "-i", self.video_path,
                                 "-acodec", "copy",
                                 "-vn", "-loglevel", "error", temp.name]
        use_shell = True if os.name == "nt" else False
        subprocess.check_output(audio_extract_command, stdin=open(os.devnull), shell=use_shell)
        if os.path.exists(self.video_temp_out_name):
            audio_merge_command = [config.FFMPEG_PATH,
                                   "-y", "-i", self.video_temp_out_name,
                                   "-i", temp.name,
                                   "-vcodec", "copy",
                                   "-acodec", "copy",
                                   "-loglevel", "error", self.video_out_name]
            subprocess.check_output(audio_merge_command, stdin=open(os.devnull), shell=use_shell)
        if os.path.exists(self.video_temp_out_name):
            os.remove(self.video_temp_out_name)
        temp.close()


if __name__ == '__main__':
    # 提示用户输入视频路径
    video_path = input(f"请输入视频文件路径: ").strip()
    # 新建字幕提取对象
    sd = SubtitleRemover(video_path)
    sd.run()
