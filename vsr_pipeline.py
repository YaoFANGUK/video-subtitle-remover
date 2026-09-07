# -*- coding: utf-8 -*-
"""VSR 生产流水线:实测验证的去字幕最佳流程,作为对外服务的处理核心。

流程:
  1. PaddleOCR 按检测反馈调整采样间隔，物化独立的逐帧字幕轨迹
  2. 可选 DashScope 贴纸定位，经时间/空间关联后生成独立遮罩
  3. LAMA 单帧或 ProPainter 分段修复，受限白字残留复核
  4. 合回源音频；最终画质仍需关键帧验收

与 backend/main.py 的区别:
  - 无 GUI/进度条/临时文件包袱,模型常驻(worker 进程 import 一次可处理多条视频)
  - 白字自检内建于流水线(实测中 OCR 漏检的低对比度字幕由它兜底)
  - 差分验收的判据固化在代码里(白字判据经正反例校准,详见 docs/02-use/04)

用法:
  CLI:  python vsr_pipeline.py -i in.mp4 -o out.mp4 --inpaint-mode propainter
  库:   Pipeline(...).process_video(input_path, output_path)
"""
import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from collections import deque
from fractions import Fraction

import av
import cv2
import numpy as np
import torch

from backend.subtitle_templates import SubtitleTemplates
from backend.subtitle_tracking import (
    associate_sticker_hits,
    group_sticker_boxes as _group_sticker_boxes,
    materialize_tracks,
    merge_residual_runs,
    plan_vlm_frames,
    sticker_match_score as _sticker_match_score,
    track_text_boxes,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 默认禁止 PaddleOCR 启动时联网检查模型源(服务器离线场景/加快启动);
# 需要联网检查时显式设 PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=False
os.environ.setdefault('PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK', 'True')
# 减少 PyTorch 显存碎片(reserved but unallocated 可达数 GB,是 OOM 常因)。
# 注意不能用 expandable_segments:它依赖 CUDA 虚拟内存 API,在虚拟化/
# 容器 GPU 环境会报 "operation not supported"(实测),用老式碎片控制
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:128')

# ---------- 可调参数(验证记录见 docs/02-use/04) ----------
DEFAULT_DET_MODEL_DIR = os.path.join(BASE_DIR, 'backend', 'models', 'V5', 'ch_det_fast')
DEFAULT_DET_MODEL_NAME = 'PP-OCRv5_mobile_det'
LAMA_PT = os.path.join(BASE_DIR, 'backend', 'models', 'big-lama', 'big-lama.pt')

MASK_PAD = 4             # OCR 框外扩像素:mask 比字形宽的环带是 ProPainter
                         # 传播距离最远、质量最差的区域(白雾残影所在),
                         # 收紧外扩(4px 盖住字形抗锯齿边缘)可显著缩小环带
STICKER_MASK_PAD = 12    # VLM 贴纸框独立外扩:模型框边界通常比 OCR 框更松,
                         # 4px 会在 emoji 边缘留下橙色残片;贴纸区域小,
                         # 增加到 12px 不扩大字幕的擦除范围
MASK_EXPAND_DOWN = 0     # mask 向下扩展:实测下扩 55px 会把字幕正下方的画面
                         # (鞋子等)罩进 mask 擦掉,且逐帧开关造成内容闪现。
                         # emoji/贴纸的擦除改由检测扩展或后处理承担,不走盲下扩
GLYPH_DILATE = 21        # 字形 mask 膨胀核(约 10px,盖住笔画边缘)
GLYPH_NEIGHBORHOOD = 60  # 字形自检的邻域:仅限 OCR 框向外扩该像素的范围
                         # (漏擦的字总是紧挨着被检出的字行;远处白色物体不进 mask,防误伤)
PROP_TEXT_MIN_GLYPH_PIXELS = 80
                         # 框内至少有这么多白色字形像素才启用精确遮罩;
                         # 抗压缩噪声或亮色物体不会触发
WHITE_ORIG_TH = 228      # 原帧白字判据:三通道下限(经 f165 残留/f180 干净校准)
WHITE_FIXED_TH = 210     # 修复帧"仍白"判据:放宽以抗重编码灰度漂移
WHITE_EDGE_TH = 180
WHITE_EDGE_RADIUS = 3
WHITE_EDGE_DILATE = 3
GLYPH_OUTLINE_RADIUS = 2  # 在亮字形外再覆盖窄暗描边，不填充整行背景
WHITE_RB_MAX = 25        # |R-B| 上限:排除蓝裤腿等彩色亮物
MIN_BOX_ASPECT = 1.8     # 检出框最小宽高比(w/h):字幕行是水平长条(实测≥2.7),
                         # 近方形框是动物/物体误检(实测狗被检出 1.1:1 的框),
                         # 贴纸通过独立 VLM 定位，不依赖 OCR 框下扩
RESID_MIN_PX = 50        # 帧内残留像素超过该值才触发补擦(抗压缩噪声)
OCR_STRIDE = 5          # 稳定时的最大间隔；有变化立即回到逐帧检测
OCR_REFINE_RADIUS = 15  # 变化时向前补查的最大帧数
OCR_STABLE_HITS = 2     # 连续稳定两次后增大间隔
SCENE_THUMB_SIZE = 32
SCENE_MEAN_DIFF = 18.0
# ProPainter 在 24GB 卡上的显存安全窗口。段输出与模型内部子窗口保持
# 一致，避免服务器上还存在 CUDA/驱动非 PyTorch 占用时，80 帧窗口 OOM。
PROPAINTER_SEG_LEN = 40
PROPAINTER_OVERLAP = 20
PROPAINTER_SUB_VIDEO_LENGTH = PROPAINTER_SEG_LEN + PROPAINTER_OVERLAP
# ffmpeg:优先用系统 PATH 里的(服务器/Linux 场景),否则回退仓库自带的平台二进制
FFMPEG = shutil.which('ffmpeg') or os.path.join(BASE_DIR, 'backend', 'ffmpeg', 'macos', 'ffmpeg')


# ---------- LAMA 引擎(自包含,不依赖 backend 包/任何 GUI 栈) ----------
class LamaEngine:
    """big-lama TorchScript 推理封装。

    语义与 backend/inpaint/lama_inpaint + lama_util 完全一致:
    输入 RGB uint8 (H,W,3) + mask uint8 (H,W),输出修复后的 RGB uint8 (H,W,3)。
    """

    def __init__(self, model_path=LAMA_PT, device='auto'):
        """device: 'auto'(有 CUDA 用 GPU,否则 CPU)/ 'cuda' / 'cpu'。"""
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        if self.device.type == 'cuda':
            # cudnn 卷积默认开 TF32(10 位尾数),对生成像素任务会累积误差
            # 表现为修复区发雾/涂抹;关掉强制 FP32,与 CPU 输出对齐
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.matmul.allow_tf32 = False
        if not os.path.exists(model_path):
            # git clone 后只有分片文件(完整 .pt 不入库),首次运行自动合并
            shard_dir = os.path.dirname(model_path)
            manifest = os.path.join(shard_dir, 'fs_manifest.csv')
            if os.path.exists(manifest):
                print(f'[init] 合并模型分片: {shard_dir}')
                from fsplit.filesplit import Filesplit
                Filesplit().merge(input_dir=shard_dir)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f'LAMA 模型缺失: {model_path}')
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()

    @staticmethod
    def _to_tensor(img, modulo=8):
        """CHW(或 HW)数组 symmetric padding 到 modulo 的倍数并转 float tensor(0~1)。

        与原 lama_util 的 get_image/pad_img_to_modulo(np.pad mode='symmetric')一致;
        2D 输入(mask)自动升维为 (1,H,W)。
        """
        if img.ndim == 2:
            img = img[np.newaxis, ...]
        c, h, w = img.shape
        oh = (h // modulo + 1) * modulo if h % modulo else h
        ow = (w // modulo + 1) * modulo if w % modulo else w
        padded = np.pad(img.astype(np.float32),
                        ((0, 0), (0, oh - h), (0, ow - w)), mode='symmetric')
        return torch.from_numpy(np.ascontiguousarray(padded)) / 255

    @torch.inference_mode()
    def inpaint(self, image_rgb, mask):
        h, w = mask.shape[:2]
        img = image_rgb.transpose(2, 0, 1)                      # HWC→CHW
        img_t = self._to_tensor(img).unsqueeze(0).to(self.device)            # (1,3,H,W) 0~1
        mask_t = self._to_tensor((mask > 0).astype('float32')).unsqueeze(0)  # (1,1,H,W)
        mask_t = (mask_t > 0) * 1
        mask_t = mask_t.to(self.device)
        out = self.model(img_t, mask_t)                          # (1,3,H,W) 0~1
        out = out[0].permute(1, 2, 0).float().cpu().numpy()
        out = np.clip(out * 255, 0, 255).astype('uint8')[:h, :w]
        return out


# ---------- VLM 贴纸/emoji 定位(可选,需 DashScope API Key) ----------
def _dashscope_key():
    """DashScope API Key:环境变量 DASHSCOPE_API_KEY 优先,
    其次 config/config.json 的 Service.DashscopeApiKey(config.json 已被
    gitignore,key 不会入库)。都没有则返回 None。"""
    key = os.environ.get('DASHSCOPE_API_KEY')
    if key:
        return key
    try:
        cfg_path = os.path.join(BASE_DIR, 'config', 'config.json')
        with open(cfg_path, encoding='utf-8') as f:
            data = json.load(f)
        return (data.get('Service') or {}).get('DashscopeApiKey')
    except Exception:
        return None


def _sticker_box_from_vlm(bbox_2d, region, pad=STICKER_MASK_PAD):
    """将 VLM 的 0–1000 坐标框换算为全帧框，并使用贴纸专用外扩。"""
    x1, y1, x2, y2 = (float(value) for value in bbox_2d)
    if not all(math.isfinite(value) for value in (x1, y1, x2, y2)) or x1 >= x2 or y1 >= y2:
        raise ValueError('invalid sticker coordinates')
    ymin, ymax, xmin, xmax = region
    width, height = xmax - xmin, ymax - ymin
    return (max(ymin, int(y1 * height / 1000) + ymin - pad),
            min(ymax, int(y2 * height / 1000) + ymin + pad),
            max(xmin, int(x1 * width / 1000) + xmin - pad),
            min(xmax, int(x2 * width / 1000) + xmin + pad))


# ---------- VLM 贴纸/emoji 定位(可选,需 DashScope API Key) ----------
def locate_stickers_vlm(video_path, region, sample_frames=None, samples=20,
                        model='qwen3.7-plus', max_calls=32, timeout=120):
    """采样帧调 VLM 定位贴纸原始框，关联和外扩由轨迹层完成。

    emoji 是图像贴纸不是文字,OCR 按设计不检测;VLM 语义定位是通用方案
    (盲下扩会误擦字幕正下方的鞋子等画面,已实测)。
    返回 {帧号: [(ymin,ymax,xmin,xmax), ...]}(0-1000 归一化坐标已换算)。
    """
    import base64
    from io import BytesIO
    import requests
    key = _dashscope_key()
    if not key:
        print('[sticker-vlm] 未配置 DASHSCOPE_API_KEY(环境变量或 config/config.json),'
              '跳过贴纸定位,emoji 将保留')
        return {}
    base = os.environ.get('DASHSCOPE_BASE_URL', 'https://dashscope.aliyuncs.com/compatible-mode/v1')
    ymin, ymax, xmin, xmax = region
    prompt = ('这是视频的一帧。请找出画面中所有 emoji 表情图标、贴纸、图案水印'
              '(不是文字,不是真实物体)。相邻的多个图标必须分别输出独立框，'
              '不要把一排图标合并成一个框；即使内容相同也分别输出。输出 JSON 数组,每项 '
              '{"label": "内容", "bbox_2d": [x1, y1, x2, y2]}(0-1000 归一化坐标)。'
              '没有则输出 []。只输出 JSON。')

    max_calls = max(0, int(max_calls))
    hits, calls = {}, 0
    with av.open(video_path) as src:
        stream = src.streams.video[0]
        total = stream.frames or (int(stream.duration * stream.time_base * stream.average_rate)
                                  if stream.duration else 0)
        if sample_frames is None:
            sample_frames = range(0, total, max(1, total // max(1, samples)))
        # 先按优先级截断，再按解码顺序请求；失败也占用预算。
        sample_set = set(list(dict.fromkeys(i for i in sample_frames if i >= 0))[:max_calls])
        if sample_set:
            for n, frame in enumerate(src.decode(video=0)):
                if calls >= max_calls or n > max(sample_set):
                    break
                if n not in sample_set:
                    continue
                calls += 1
                try:
                    img = frame.to_image().crop((xmin, ymin, xmax, ymax))
                    buf = BytesIO()
                    img.save(buf, format='PNG')
                    b64 = base64.b64encode(buf.getvalue()).decode()
                    resp = requests.post(
                        f'{base}/chat/completions',
                        headers={'Authorization': f'Bearer {key}'},
                        json={'model': model, 'messages': [{'role': 'user', 'content': [
                            {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{b64}'}},
                            {'type': 'text', 'text': prompt}]}]}, timeout=timeout)
                    resp.raise_for_status()
                    content = resp.json()['choices'][0]['message']['content'].strip()
                    if content.startswith('```'):
                        content = content.split('```')[1]
                        if content.startswith('json'):
                            content = content[4:]
                    parsed = json.loads(content)
                    if not isinstance(parsed, list):
                        raise ValueError('expected sticker list')
                    boxes = [_sticker_box_from_vlm(item['bbox_2d'], region, pad=0) for item in parsed]
                    hits[n] = list(dict.fromkeys(box for box in boxes
                                                if box[0] < box[1] and box[2] < box[3]))
                except Exception as exc:
                    print(f'[sticker-vlm] f{n} 定位失败: {type(exc).__name__}')
    print(f'[sticker-vlm] calls={calls}/{max_calls} hits={sum(map(len, hits.values()))}')
    return hits


# ---------- 模型单例(worker 进程内 import 一次,处理多条视频复用) ----------
class Pipeline:
    """持有常驻模型,提供单视频处理入口。"""

    def __init__(self, det_model_dir=DEFAULT_DET_MODEL_DIR,
                 det_model_name=DEFAULT_DET_MODEL_NAME,
                 lama_pt=LAMA_PT, threads=None, device='auto', inpaint_mode='lama'):
        if threads:
            torch.set_num_threads(threads)
        self.inpaint_mode = inpaint_mode
        print(f'[init] 加载 OCR 检测模型: {det_model_dir}')
        from paddleocr import TextDetection
        # OCR 固定 CPU:占比小(~13%),不值得为它装 paddle-gpu
        self.ocr = TextDetection(
            model_name=det_model_name,
            model_dir=det_model_dir,
            device='cpu',
            enable_hpi=False,
        )
        if inpaint_mode == 'lama':
            print(f'[init] 加载 LAMA: {lama_pt}')
            self.inpainter = LamaEngine(lama_pt, device=device)
            print(f'[init] 模型就绪(LAMA device: {self.inpainter.device})')
        elif inpaint_mode == 'propainter':
            # ProPainter 时序修复:被字幕遮挡的真实像素可从相邻帧沿光流传播
            # 回来,重建质量远超单帧 LAMA(对照 kaipai 目标效果);显存大,
            # 必须 GPU,首次用到时才加载
            self._pp_device = torch.device('cuda' if (device == 'auto' and torch.cuda.is_available()) or device == 'cuda' else 'cpu')
            self.inpainter = None
            print(f'[init] ProPainter 模式(引擎将在首次修复时加载,device: {self._pp_device})')
        else:
            raise ValueError(f'未知 inpaint_mode: {inpaint_mode}')

    def _ensure_propainter(self):
        """ProPainter 惰性加载(首次修复时)。"""
        if self.inpainter is None:
            from backend.inpaint.propainter_inpaint import PropainterInpaint
            from backend.tools.model_config import ModelConfig
            self.inpainter = PropainterInpaint(
                device=self._pp_device,
                model_dir=ModelConfig().PROPAINTER_MODEL_DIR,
                sub_video_length=PROPAINTER_SUB_VIDEO_LENGTH,
                use_fp16=self._pp_device.type == 'cuda',
            )
            print('[init] ProPainter 已加载')

    # ---- OCR 检测:返回该帧在 region 内的文字框列表 [(ymin,ymax,xmin,xmax), ...] ----
    def detect(self, frame_rgb, region):
        ymin, ymax, xmin, xmax = region
        # region 裁剪送检:局部图不触发 det 的整帧缩放,框更贴合字形(实测 y 范围约紧一半);
        # 全屏时等价于原行为。PaddleX 惯例吃 BGR
        crop = frame_rgb[ymin:ymax, xmin:xmax]
        results = self.ocr.predict(cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
        boxes = []
        for res in results:
            polys = res.get('dt_polys')
            if polys is None or len(polys) == 0:
                continue
            for poly in polys:
                x1, y1 = poly[:, 0].min(), poly[:, 1].min()
                x2, y2 = poly[:, 0].max(), poly[:, 1].max()
                bw, bh = x2 - x1, y2 - y1
                # 宽高比过滤:字幕行是水平长条;近方形框是动物/物体误检
                if bw < MIN_BOX_ASPECT * bh:
                    continue
                # 坐标平移回全帧,外扩后输出
                box = (max(ymin, int(y1) + ymin - MASK_PAD),
                       min(ymax, int(y2) + ymin + MASK_PAD),
                       max(xmin, int(x1) + xmin - MASK_PAD),
                       min(xmax, int(x2) + xmin + MASK_PAD))
                if box[0] < box[1] and box[2] < box[3]:
                    boxes.append(box)
        return boxes

    @staticmethod
    def _detection_stable(previous, current, previous_img, current_img):
        if len(previous) != len(current):
            return False
        unmatched = list(current)
        for old in previous:
            match = next((box for box in unmatched
                          if max(abs(a - b) for a, b in zip(old, box)) <= 4), None)
            if match is None:
                return False
            unmatched.remove(match)
            y1, y2, x1, x2 = match
            before = cv2.resize(previous_img[y1:y2, x1:x2], (32, 8))
            after = cv2.resize(current_img[y1:y2, x1:x2], (32, 8))
            if np.abs(before.astype(np.float32) - after).mean() > SCENE_MEAN_DIFF:
                return False
        return True

    def _detect_timeline(self, input_path, region, ocr_stride=OCR_STRIDE,
                         ocr_refine_radius=OCR_REFINE_RADIUS, progress=None):
        """按检测反馈调整间隔；只缓存最近跳过的帧用于变化后的回查。"""
        stride = max(1, int(ocr_stride))
        radius = max(1, int(ocr_refine_radius))
        pending = deque(maxlen=min(stride, radius))
        sampled, sample_frames, scene_changes = {}, [], []
        previous_boxes, previous_img, previous_thumb = None, None, None
        interval, stable_hits, next_frame, refined = 1, 0, 0, 0

        def observe(frame_no, img, scene_change=False):
            nonlocal previous_boxes, previous_img, interval, stable_hits, next_frame, refined
            boxes = self.detect(img, region)
            sample_frames.append(frame_no)
            changed = (scene_change or previous_boxes is None
                       or not self._detection_stable(previous_boxes, boxes, previous_img, img))
            if changed:
                for skipped_no, skipped_img in pending:
                    sampled[skipped_no] = self.detect(skipped_img, region)
                    refined += 1
                interval, stable_hits = 1, 0
            else:
                stable_hits += 1
                if stable_hits >= OCR_STABLE_HITS:
                    interval = min(stride, interval * 2)
                    stable_hits = 0
            sampled[frame_no] = boxes
            pending.clear()
            previous_boxes, previous_img = boxes, img
            next_frame = frame_no + interval

        total, last_img = 0, None
        with av.open(input_path) as src:
            stream = src.streams.video[0]
            estimate = stream.frames or 0
            for frame_no, frame in enumerate(src.decode(video=0)):
                img = frame.to_ndarray(format='rgb24')
                y1, y2, x1, x2 = region
                thumb = cv2.resize(img[y1:y2, x1:x2], (SCENE_THUMB_SIZE, SCENE_THUMB_SIZE))
                cut = (previous_thumb is not None
                       and np.abs(thumb.astype(np.float32) - previous_thumb).mean() > SCENE_MEAN_DIFF)
                if cut:
                    scene_changes.append(frame_no)
                if frame_no >= next_frame or cut:
                    observe(frame_no, img, cut)
                else:
                    pending.append((frame_no, img))
                previous_thumb, last_img, total = thumb, img, frame_no + 1
                if progress and (total % 30 == 0 or total == estimate):
                    progress(total, estimate, f'检测中 OCR {len(sampled)}')
        if total and total - 1 not in sampled:
            # 最后一帧必须检测，但不能把它再作为跳过帧重复回查。
            pending.pop()
            observe(total - 1, last_img)
        max_gap = max(10, 2 * stride)
        tracks = track_text_boxes(sampled, total, max_gap=max_gap,
                                  scene_change_frames=scene_changes)
        timeline = materialize_tracks(tracks, total, max_interpolation_gap=max_gap)
        accepted = sum(len(track.frames) for track in tracks)
        return timeline, {
            'ocr_calls': len(sampled), 'sampled': len(sample_frames), 'refined': refined,
            'tracks': len(tracks), 'discarded': sum(map(len, sampled.values())) - accepted,
            'sampled_frames': sample_frames,
            'scene_change_frames': scene_changes,
        }

    @staticmethod
    def boxes_to_mask(boxes, h, w):
        mask = np.zeros((h, w), dtype='uint8')
        for ymin, ymax, xmin, xmax in boxes:
            mask[max(0, ymin):min(h, ymax + MASK_EXPAND_DOWN),
                 max(0, xmin):min(w, xmax)] = 255
        return mask

    def propainter_boxes_to_mask(self, boxes, frame_rgb, region, sticker_boxes=()):
        """为 ProPainter 生成精确遮罩,避免把字幕框内背景整体重绘。

        OCR 只返回文字行的外接矩形,而不是字形轮廓。矩形中未被文字
        覆盖的楼梯、裤腿等真实像素若一并送入 ProPainter,模型会重新生成
        它们,在 4–5 秒这类字幕压在物体上的场景尤其明显。对明显横向的
        白色字幕框,改用原帧白色字形作为遮罩;无白字框保留整框。
        贴纸类型独立传入，保持矩形遮罩。
        """
        h, w = frame_rgb.shape[:2]
        if not boxes and not sticker_boxes:
            return np.zeros((h, w), dtype='uint8')
        raw = self.white_glyph(frame_rgb, region)
        glyph = self.filter_glyph_by_height(raw)
        loose = self.filter_glyph_by_height(self.white_glyph(frame_rgb, region, WHITE_EDGE_TH))
        mask = np.zeros((h, w), dtype='uint8')
        ry1, ry2, rx1, rx2 = region
        for ymin, ymax, xmin, xmax in boxes:
            y1, y2 = max(0, ry1, ymin), min(h, ry2, ymax)
            x1, x2 = max(0, rx1, xmin), min(w, rx2, xmax)
            if y2 <= y1 or x2 <= x1:
                continue
            local_glyph = glyph[y1:y2, x1:x2]
            # OCR 已完成文字形态过滤，外扩后的宽高比不能再排除短字幕。
            if int((local_glyph > 0).sum()) >= PROP_TEXT_MIN_GLYPH_PIXELS:
                near_core = cv2.dilate(local_glyph, np.ones(
                    (2 * WHITE_EDGE_RADIUS + 1, 2 * WHITE_EDGE_RADIUS + 1), dtype='uint8'))
                edge = cv2.bitwise_and(loose[y1:y2, x1:x2], near_core)
                edge = cv2.dilate(edge, np.ones((WHITE_EDGE_DILATE, WHITE_EDGE_DILATE), dtype='uint8'))
                text_mask = cv2.bitwise_or(local_glyph, edge)
                text_mask = cv2.dilate(text_mask, np.ones(
                    (2 * GLYPH_OUTLINE_RADIUS + 1, 2 * GLYPH_OUTLINE_RADIUS + 1), dtype='uint8'))
                mask[y1:y2, x1:x2] = np.maximum(mask[y1:y2, x1:x2], text_mask)
            elif np.count_nonzero(raw[y1:y2, x1:x2]) >= PROP_TEXT_MIN_GLYPH_PIXELS:
                # 白色大物体被连通域过滤后不能退回整行矩形擦除。
                mask[y1:y2, x1:x2] = np.maximum(mask[y1:y2, x1:x2], local_glyph)
            else:
                mask[y1:y2, x1:x2] = 255
        for ymin, ymax, xmin, xmax in sticker_boxes:
            y1, y2 = max(0, ry1, ymin), min(h, ry2, ymax)
            x1, x2 = max(0, rx1, xmin), min(w, rx2, xmax)
            if y1 < y2 and x1 < x2:
                mask[y1:y2, x1:x2] = 255
        return mask

    @staticmethod
    def white_glyph(frame, region, threshold=WHITE_ORIG_TH):
        """原帧白字形检测(独立于 OCR,差分验收的同款判据)。"""
        glyph = np.zeros(frame.shape[:2], dtype='uint8')
        ymin, ymax, xmin, xmax = region
        r = frame[ymin:ymax, xmin:xmax].astype(np.int16)
        white = ((r[:, :, 0] > threshold) & (r[:, :, 1] > threshold)
                 & (r[:, :, 2] > threshold) & (np.abs(r[:, :, 0] - r[:, :, 2]) < WHITE_RB_MAX))
        glyph[ymin:ymax, xmin:xmax] = white.astype('uint8') * 255
        return glyph

    @staticmethod
    def filter_glyph_by_height(glyph, max_h=90):
        """按连通域高度过滤白字形:字幕单行高 ≤60px;白色衣物/大块白色物体
        是几百 px 的大连通块,必须剔除,否则 LAMA 会把人/物当字幕抹掉(实测灾难)。"""
        num, labels, stats, _ = cv2.connectedComponentsWithStats(glyph, connectivity=8)
        out = np.zeros_like(glyph)
        for i in range(1, num):
            if stats[i, cv2.CC_STAT_HEIGHT] <= max_h:
                out[labels == i] = 255
        return out

    @staticmethod
    def residual_white(fixed_rgb, glyph):
        """修复帧在原白字形位置上仍是白的像素数(=漏擦残留)。"""
        f = fixed_rgb.astype(np.int16)
        still = ((f[:, :, 0] > WHITE_FIXED_TH) & (f[:, :, 1] > WHITE_FIXED_TH)
                 & (f[:, :, 2] > WHITE_FIXED_TH) & (np.abs(f[:, :, 0] - f[:, :, 2]) < WHITE_RB_MAX))
        return int(((glyph > 0) & still).sum())

    def _residual_mask(self, fixed_bgr, original_bgr, boxes):
        h, w = fixed_bgr.shape[:2]
        original = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
        original_gray = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY)
        fixed_gray = cv2.cvtColor(fixed_bgr, cv2.COLOR_BGR2GRAY)
        # 亮度本身不能区分白衣服与白字，也看不到擦除白字后留下的暗描边。
        # 同极性连通笔画必须大部分与原结构重合，不能只取偶然相交的像素。
        raw_white = self.white_glyph(original, (0, h, 0, w))
        allowed = cv2.dilate(raw_white, np.ones((9, 9), dtype='uint8'))
        kernel = np.ones((15, 15), dtype='uint8')
        residual = np.zeros((h, w), dtype='uint8')
        for operation in (cv2.MORPH_TOPHAT, cv2.MORPH_BLACKHAT):
            before = cv2.morphologyEx(original_gray, operation, kernel)
            after = cv2.morphologyEx(fixed_gray, operation, kernel)
            count, labels, stats, _ = cv2.connectedComponentsWithStats(
                (after >= 12).astype('uint8'), connectivity=8)
            supported = (before >= 18) & (allowed > 0)
            overlap = np.bincount(labels[supported], minlength=count)
            accepted = ((overlap >= stats[:, cv2.CC_STAT_AREA] * 0.65)
                        & (stats[:, cv2.CC_STAT_AREA] >= 6))
            accepted[0] = False
            matched = (accepted[labels] & supported).astype('uint8') * 255
            residual = cv2.bitwise_or(residual, matched)
        return cv2.bitwise_and(residual, self.boxes_to_mask(boxes, h, w))

    def _repair_propainter_segment(self, frames_bgr, masks, boxes, white_glyph_check=True):
        if len(frames_bgr) == 1:
            # RAFT 需要帧对；复制末帧只补上下文，不增加输出帧数。
            result, repairs = self._repair_propainter_segment(
                frames_bgr * 2, masks * 2, boxes * 2, white_glyph_check)
            return result[:1], repairs
        raw = self.inpainter.inpaint(frames_bgr, masks)
        # 模型内部膨胀仅用于推理，输出严格限制在调用方的精确遮罩内。
        first = [np.where(mask[:, :, None] > 0, fixed, original)
                 for fixed, original, mask in zip(raw, frames_bgr, masks)]
        if not white_glyph_check:
            return first, 0
        try:
            # 先在模型完整输出上核验结构，避免合成裁断背景线条后误判成字形。
            residual = [self._residual_mask(fixed, original, text_boxes)
                        for fixed, original, text_boxes in zip(raw, frames_bgr, boxes)]
            del raw
            residual = [cv2.bitwise_and(candidate, mask)
                        for candidate, mask in zip(residual, masks)]
            runs = merge_residual_runs(
                [i for i, mask in enumerate(residual) if np.count_nonzero(mask) >= RESID_MIN_PX],
                total=len(first), context=5, max_runs=1)
            repairs = 0
            for lo, hi in runs:
                local_masks = [cv2.bitwise_and(
                    cv2.dilate(residual[i], np.ones((5, 5), dtype='uint8')),
                    cv2.bitwise_and(masks[i], self.boxes_to_mask(boxes[i], *residual[i].shape)))
                    for i in range(lo, hi + 1)]
                # 二次修复以首轮结果为输入，避免把已擦掉的文字重新传播回来。
                second = self.inpainter.inpaint([f.copy() for f in first[lo:hi + 1]], local_masks)
                if len(second) != hi - lo + 1:
                    raise ValueError('unexpected repair frame count')
                repaired_frames = [np.where(mask[:, :, None] > 0, repaired, first[i])
                                   for i, (mask, repaired) in enumerate(zip(local_masks, second), start=lo)]
                first[lo:hi + 1] = repaired_frames
                repairs += 1
            return first, repairs
        except Exception as exc:
            print(f'[propainter] 局部残留复修失败，保留首轮结果: {type(exc).__name__}')
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return first, 0

    def auto_region(self, input_path, samples=24):
        """自动探测字幕区域:均匀采样帧做全屏 OCR,取检出框并集外扩。

        全屏检测会漏检低对比度字幕,但检出的框足以定位字幕活动带
        (漏检字总是紧挨检出字行),外扩后即可覆盖。无检出时回退 None(全屏)。
        """
        c = av.open(input_path)
        vstream = next(s for s in c.streams if s.type == 'video')
        total = vstream.duration and int(float(vstream.duration * vstream.time_base * float(vstream.average_rate))) or 0
        h = vstream.codec_context.height
        w = vstream.codec_context.width
        full = (0, h, 0, w)
        step = max(1, total // samples) if total else 1
        boxes_all = []
        n = 0
        for frame in c.decode(video=0):
            if n % step == 0:
                boxes_all += self.detect(np.asarray(frame.to_image()), full)
            n += 1
        c.close()
        if not boxes_all:
            print('[auto-region] 采样未检出任何字幕框,回退全屏')
            return None
        # 聚类过滤:按框 y 中心分簇(间隔 120px),只保留框数≥2 的簇——
        # 真实字幕在多帧持续出现形成密集带,孤立的单框多为画面误检(高光/接缝)
        centers = sorted((b[0] + b[1]) / 2 for b in boxes_all)
        clusters = [[centers[0]]]
        for cy in centers[1:]:
            if cy - clusters[-1][-1] <= 120:
                clusters[-1].append(cy)
            else:
                clusters.append([cy])
        keep_ranges = [(c[0], c[-1]) for c in clusters if len(c) >= 2]
        if not keep_ranges:
            print('[auto-region] 检出框过于孤立,回退全屏')
            return None
        boxes_all = [b for b in boxes_all
                     if any(lo <= (b[0] + b[1]) / 2 <= hi for lo, hi in keep_ranges)]
        print(f'[auto-region] 聚类过滤后保留 {len(boxes_all)}/{len(centers)} 框')
        ymin = max(0, min(b[0] for b in boxes_all) - 80)
        ymax = min(h, max(b[1] for b in boxes_all) + 80)
        xmin = max(0, min(b[2] for b in boxes_all) - 40)
        xmax = min(w, max(b[3] for b in boxes_all) + 40)
        print(f'[auto-region] 采样检出 {len(boxes_all)} 框 → region: {(ymin, ymax, xmin, xmax)}')
        return (ymin, ymax, xmin, xmax)

    @staticmethod
    def _boxes_overlap(b1, b2, pad=60):
        """两框(外扩 pad)是否重叠:字幕随镜头轻微移动仍视为同一字幕。"""
        return not (b1[3] < b2[2] - pad or b1[2] > b2[3] + pad
                    or b1[1] < b2[0] - pad or b1[0] > b2[1] + pad)

    @classmethod
    def filter_boxes_by_continuity(cls, all_boxes, max_gap=5):
        """检出框的位置连续性过滤:剔除孤立错位的误检框。

        字幕在时间上连续,检出框应与前后检出帧的框位置衔接。一个框若与
        前后(帧距 ≤max_gap)检出帧的所有框均不重叠,则是画面误检
        (高光/动物被当文字),交给 ProPainter 会传播来错误内容
        (实测 f271 雾块)。删除后该帧交由最近邻继承。
        """
        hits = [i for i, b in enumerate(all_boxes) if b]
        out = [list(b) for b in all_boxes]

        def overlap_any(b, others):
            return any(cls._boxes_overlap(b, pb) for pb in others)

        for i in hits:
            prev = max((j for j in hits if j < i), default=None)
            nxt = min((j for j in hits if j > i), default=None)
            prev_ok = prev is not None and i - prev <= max_gap
            nxt_ok = nxt is not None and nxt - i <= max_gap
            if not prev_ok and not nxt_ok:
                continue  # 邻居太远无法判定,保守保留
            kept = []
            for b in all_boxes[i]:
                fail_prev = prev_ok and not overlap_any(b, all_boxes[prev])
                fail_nxt = nxt_ok and not overlap_any(b, all_boxes[nxt])
                if prev_ok and nxt_ok:
                    if fail_prev and fail_nxt:
                        continue  # 两侧都脱节:孤立误检,剔除
                elif fail_prev or fail_nxt:
                    continue  # 单侧可判定且脱节:剔除
                kept.append(b)
            out[i] = kept
        return out

    @classmethod
    def expand_timeline(cls, all_boxes, merge_gap=10):
        """字幕时间线区间化 + 最近邻传播。

        目标是防字幕闪现(逐帧独立检测时字幕'忽检出忽漏检',擦与不擦交替):
        帧号间隔 ≤merge_gap 的检出帧合并为同一区间,区间内无检出框的帧
        继承时间上最近的检出帧的框。

        注意不能用区间内全部检出框的并集:动态字幕(位置随镜头移动)的
        并集会横跨整条移动带,把人物/背景大面积罩进 mask,LAMA 会把
        画面重绘成模糊涂抹(实测灾难)。最近邻帧的框最贴近该帧字幕的
        真实位置。
        """
        all_boxes = cls.filter_boxes_by_continuity(all_boxes)
        n = len(all_boxes)
        hits = [i for i, b in enumerate(all_boxes) if b]
        if not hits:
            return [[] for _ in range(n)]
        # 按帧号聚类成区间
        ranges = [[hits[0], hits[0]]]
        for i in hits[1:]:
            if i - ranges[-1][1] <= merge_gap:
                ranges[-1][1] = i
            else:
                ranges.append([i, i])
        expanded = [list(b) for b in all_boxes]
        for lo, hi in ranges:
            frames_with = [i for i in range(lo, hi + 1) if all_boxes[i]]
            for i in range(lo, hi + 1):
                # 局部窗口(±2帧)并集:补跨帧漏检(某帧漏检的字行,常被相邻帧
                # 检出)。窗口小,字幕移动量有限,并集不会横跨移动带(对比:
                # 全区间并集会把整条移动带罩住,无真值可抄→白雾)
                near = [j for j in frames_with if abs(j - i) <= 2]
                if near:
                    union = []
                    for j in near:
                        for b in all_boxes[j]:
                            if b not in union:
                                union.append(b)
                    expanded[i] = union
                    continue
                # 窗口内无检出(漏检串>5帧):继承最近检出帧的框
                prev = max((j for j in frames_with if j < i), default=None)
                nxt = min((j for j in frames_with if j > i), default=None)
                if prev is None:
                    expanded[i] = all_boxes[nxt]
                elif nxt is None:
                    expanded[i] = all_boxes[prev]
                else:
                    expanded[i] = all_boxes[prev if i - prev <= nxt - i else nxt]
        return expanded

    def process_video(self, input_path, output_path, region=None,
                      white_glyph_check=True, progress=None, locate_stickers=True,
                      ocr_stride=OCR_STRIDE, ocr_refine_radius=OCR_REFINE_RADIUS,
                      vlm_max_calls=32):
        """处理单条视频：反馈式 OCR 和轨迹检测，然后按帧或分段修复。

        :param region: (ymin, ymax, xmin, xmax) 字幕区域;None = 全屏检测
        :param ocr_stride: 稳定检测时逐步增大的帧间隔上限，默认 5
        :param white_glyph_check: 白字自检开关(白字幕场景必开;彩色字幕场景关闭,
                                  避免把画面中的白色物体误当残留)
        :param progress: 回调 fn(done_frames, total_frames, stage)
        """
        input_path, output_path = os.fspath(input_path), os.fspath(output_path)
        mode = 'full-frame' if region is None else 'roi'
        with av.open(input_path) as metadata:
            vstream = metadata.streams.video[0]
            rate = vstream.average_rate or Fraction(30, 1)
            w, h = vstream.codec_context.width, vstream.codec_context.height
        fps = float(rate)
        region = tuple(region) if region is not None else (0, h, 0, w)
        ry1, ry2, rx1, rx2 = region
        if not (0 <= ry1 < ry2 <= h and 0 <= rx1 < rx2 <= w):
            raise ValueError(f'字幕区域超出视频尺寸 {w}x{h}: {region}')
        ocr_stride, ocr_refine_radius = max(1, int(ocr_stride)), max(1, int(ocr_refine_radius))
        vlm_max_calls = max(1, int(vlm_max_calls))
        t0 = time.time()
        print(f'[detect] mode={mode} roi={region} stride={ocr_stride} adaptive=feedback')
        all_boxes, detection = self._detect_timeline(
            input_path, region, ocr_stride, ocr_refine_radius, progress)
        total = len(all_boxes)
        print(f'[detect] sampled={detection["sampled"]} refined={detection["refined"]} '
              f'ocr_calls={detection["ocr_calls"]} tracks={detection["tracks"]} '
              f'discarded={detection["discarded"]}')
        sticker_boxes = {}
        if locate_stickers:
            try:
                samples = plan_vlm_frames(total, all_boxes, vlm_max_calls, max(1, round(fps)),
                                          scene_change_frames=detection['scene_change_frames'])
                hits = locate_stickers_vlm(input_path, region, sample_frames=samples,
                                           max_calls=vlm_max_calls)
                associated = associate_sticker_hits(hits, all_boxes, total,
                    max_gap=max(1, round(fps * 2)), scene_change_frames=detection['scene_change_frames'])
                sticker_boxes = {i: [(max(ry1, y1 - STICKER_MASK_PAD), min(ry2, y2 + STICKER_MASK_PAD),
                                      max(rx1, x1 - STICKER_MASK_PAD), min(rx2, x2 + STICKER_MASK_PAD))
                                     for y1, y2, x1, x2 in boxes]
                                 for i, boxes in associated.items()}
                print(f'[sticker-vlm] associated_frames={len(sticker_boxes)}')
            except Exception as exc:
                print(f'[sticker-vlm] 跳过贴纸层: {type(exc).__name__}')

        # 第二遍:修复 + 写出
        frame_tb = 1 / rate
        tmp_out = output_path + '.tmp.mp4'
        dst = av.open(tmp_out, 'w')
        ov = dst.add_stream('libx264', rate=rate)
        ov.width = w; ov.height = h; ov.pix_fmt = 'yuv420p'
        ov.options = {'crf': '18', 'bf': '0'}
        src = av.open(input_path)
        n_fixed = n_repair = n_checked = n = 0
        n_recovered = n_unresolved = n_check_failed = 0
        roi_mask = self.boxes_to_mask([region], h, w)
        scene_changes = set(detection['scene_change_frames'])

        if self.inpaint_mode == 'propainter':
            # ---- ProPainter 分支:按连续字幕段批处理(时序模型,不可逐帧) ----
            SEG_LEN, OVERLAP = PROPAINTER_SEG_LEN, PROPAINTER_OVERLAP
                                        # 每段输出 40 帧,尾部 20 帧重叠给下一段当上下文
                                        # (24G 卡在实际服务器非 PyTorch 显存占用较高时,
                                        #  80 帧窗口仍会 OOM;60 帧输入优先保证稳定运行)
            seg_frames, seg_masks, seg_pts, seg_boxes = [], [], [], []
            templates = SubtitleTemplates(region, max_age=PROPAINTER_SUB_VIDEO_LENGTH)

            def flush_segment(n_out):
                """处理当前缓冲:送入全部帧(含尾部重叠上下文),只输出前 n_out 帧。

                重叠帧不输出、留给下一段作为它的"过去"——段边界的最后一帧
                因此拥有后向上下文,消除段边界跳变(实测 4.96x)。
                逐帧 mask 列表(非并集):保证传播源不被并集污染(并集会让
                所有帧的移动带都成空洞,无真值可抄→白雾)。
                """
                nonlocal seg_frames, seg_masks, seg_pts, seg_boxes, n_fixed, n_repair, n_checked
                nonlocal n_recovered, n_unresolved, n_check_failed
                if not seg_frames or n_out <= 0:
                    return
                effective_masks, effective_boxes = templates.refine(
                    seg_frames, seg_masks, seg_boxes, seg_pts)
                n_recovered += sum(np.any((new > 0) & (old == 0))
                                   for new, old in zip(effective_masks[:n_out], seg_masks[:n_out]))
                if any(mask.any() for mask in effective_masks):
                    self._ensure_propainter()
                    comps, repairs = self._repair_propainter_segment(
                        seg_frames, effective_masks, effective_boxes, white_glyph_check)
                else:
                    comps, repairs = seg_frames, 0
                n_repair += repairs
                if white_glyph_check:
                    try:
                        remaining = sum(np.count_nonzero(self._residual_mask(
                            comps[j], seg_frames[j], effective_boxes[j])) >= RESID_MIN_PX
                            for j in range(n_out))
                    except Exception as exc:
                        n_check_failed += n_out
                        print(f'[propainter] 复核未完成 {n_out} 帧，保留修复结果: '
                              f'{type(exc).__name__}')
                    else:
                        n_checked += n_out
                        n_unresolved += remaining
                        if remaining:
                            print(f'[propainter] 帧 {seg_pts[0]}-{seg_pts[n_out - 1]} '
                                  f'疑似残留 {remaining}，已到本段复修上限或缺少可信遮罩')
                for j in range(n_out):
                    comp = np.where(roi_mask[:, :, None] > 0, comps[j], seg_frames[j])
                    frame = av.VideoFrame.from_ndarray(
                        cv2.cvtColor(comp, cv2.COLOR_BGR2RGB), format='rgb24')
                    frame.pts = seg_pts[j]
                    frame.time_base = frame_tb
                    for pkt in ov.encode(frame):
                        dst.mux(pkt)
                n_fixed += sum(bool(mask.any()) for mask in effective_masks[:n_out])
                seg_frames = seg_frames[n_out:]
                seg_masks = seg_masks[n_out:]
                seg_pts = seg_pts[n_out:]
                seg_boxes = seg_boxes[n_out:]

            for frame in src.decode(video=0):
                n += 1
                if n - 1 in scene_changes:
                    flush_segment(len(seg_frames))
                    templates.reset()
                img = np.asarray(frame.to_image())  # RGB
                boxes = all_boxes[n - 1] if n - 1 < len(all_boxes) else []
                stickers = sticker_boxes.get(n - 1, [])
                mask = self.propainter_boxes_to_mask(boxes, img, region, sticker_boxes=stickers)
                if boxes or mask.any():
                    # 文字框只遮白色字形,保留字间的楼梯/裤腿等真实像素;
                    # VLM 贴纸和有色字幕仍由精确矩形覆盖。
                    seg_masks.append(mask)
                    seg_frames.append(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                    seg_pts.append(n - 1)
                    seg_boxes.append(boxes)
                    if len(seg_frames) >= SEG_LEN + OVERLAP:
                        flush_segment(SEG_LEN)
                else:
                    flush_segment(len(seg_frames))   # 段结束:重叠无意义,全部输出
                    frame.pts = n - 1
                    frame.time_base = frame_tb
                    for pkt in ov.encode(frame):
                        dst.mux(pkt)
                if progress and (n % 30 == 0 or n == total):
                    progress(n, total, f'ProPainter 修复 {n_fixed}')
            flush_segment(len(seg_frames))
            for pkt in ov.encode():
                dst.mux(pkt)
            dst.close()
        else:
            # ---- LAMA 分支:逐帧修复 + 白字自检 + 补擦 + 防闪混合 ----
            for frame in src.decode(video=0):
                n += 1
                img = np.asarray(frame.to_image())  # RGB
                text_boxes = all_boxes[n - 1] if n - 1 < len(all_boxes) else []
                boxes = text_boxes + sticker_boxes.get(n - 1, [])
                if boxes:
                    mask = self.boxes_to_mask(boxes, h, w)
                    fixed = self.inpainter.inpaint(img, mask)
                    n_fixed += 1
                    # 白字自检:仅在 OCR 框邻域内找漏擦字(远离框的白色物体不误伤)
                    if white_glyph_check:
                        hood = np.zeros((h, w), dtype='uint8')
                        for gy1, gy2, gx1, gx2 in text_boxes:
                            hood[max(0, gy1 - GLYPH_NEIGHBORHOOD):min(h, gy2 + GLYPH_NEIGHBORHOOD),
                                 max(0, gx1 - GLYPH_NEIGHBORHOOD):min(w, gx2 + GLYPH_NEIGHBORHOOD)] = 255
                        glyph = self.white_glyph(img, region)
                        glyph = cv2.bitwise_and(glyph, hood)
                        glyph = self.filter_glyph_by_height(glyph)
                        resid = self.residual_white(fixed, glyph)
                        if resid > RESID_MIN_PX:
                            kernel = np.ones((GLYPH_DILATE, GLYPH_DILATE), 'uint8')
                            glyph_mask = cv2.dilate(glyph, kernel)
                            glyph_mask = cv2.bitwise_and(glyph_mask, roi_mask)
                            fixed = self.inpainter.inpaint(fixed, glyph_mask)
                            n_repair += 1
                            print(f'  [补擦] 帧 {n}: 残留 {resid}px 已二次修复')
                    # 帧间防闪:mask 外严格保留原帧像素(模型对 mask 外的输出有逐帧
                    # 随机细微差,整帧替换会造成全画面轻微闪烁)
                    blend_mask = cv2.bitwise_and(cv2.dilate(mask, np.ones((7, 7), 'uint8')), roi_mask)
                    m3 = blend_mask.astype(np.float32)[:, :, None] / 255
                    blended = (img.astype(np.float32) * (1 - m3) + fixed.astype(np.float32) * m3)
                    frame = av.VideoFrame.from_ndarray(blended.astype('uint8'), format='rgb24')
                # 显式 pts:PyAV 对 VideoStream.encode 的自动 pts 分配在长序列上会
                # 产生乱序包(实测 flush 时 pts 跳回 3 导致 mux EINVAL/服务器丢帧),
                # 按帧号单调递增是标准做法,时间戳完全可控
                frame.pts = n - 1
                frame.time_base = frame_tb
                for pkt in ov.encode(frame):
                    dst.mux(pkt)
                if progress and (n % 30 == 0 or n == total):
                    progress(n, total, f'修复 {n_fixed} / 补擦 {n_repair}')
            for pkt in ov.encode():
                dst.mux(pkt)
            dst.close()

        # 源音频以 AAC 合回，源无音频时直接改名。
        has_audio = any(s.type == 'audio' for s in src.streams)
        src.close()
        if has_audio:
            final = output_path + '.mux.mp4'
            subprocess.check_output([
                FFMPEG, '-y', '-i', tmp_out, '-i', input_path,
                '-map', '0:v:0', '-map', '1:a:0',
                # 音频边界可能略早于视频；-shortest 会截掉已写出的末帧。
                '-c:v', 'copy', '-c:a', 'aac',
                '-loglevel', 'error', final])
            os.remove(tmp_out)
            os.replace(final, output_path)
        else:
            os.replace(tmp_out, output_path)
        print(f'[done] {n} 帧 | 修复 {n_fixed} | 字形补全 {n_recovered} | '
              f'残留复核 {n_checked} | 补擦 {n_repair} | 疑似残留 {n_unresolved} | '
              f'复核未完成 {n_check_failed} | '
              f'耗时 {time.time() - t0:.0f}s → {output_path}')
        return {'frames': n, 'inpainted': n_fixed, 'repaired': n_repair,
                'template_recovered': int(n_recovered), 'unresolved': n_unresolved,
                'residual_check_failed': n_check_failed,
                'ocr_calls': detection['ocr_calls'], 'tracks': detection['tracks'],
                'seconds': time.time() - t0}


# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description='去字幕生产流水线(实测验证版)')
    ap.add_argument('-i', '--input', required=True)
    ap.add_argument('-o', '--output', required=True)
    ap.add_argument('-c', '--region', nargs=4, type=int, metavar=('YMIN', 'YMAX', 'XMIN', 'XMAX'),
                    help='手动检测区域;不传则全屏自适应检测')
    ap.add_argument('--ocr-stride', type=int, default=OCR_STRIDE,
                    help='OCR 稳定后逐步增大的采样间隔上限(帧),默认 5')
    ap.add_argument('--ocr-refine-radius', type=int, default=OCR_REFINE_RADIUS,
                    help='检测变化时向前补查的最大帧数,默认 15')
    ap.add_argument('--vlm-max-calls', type=int, default=32,
                    help='单视频 DashScope 最大请求次数(含失败),默认 32')
    ap.add_argument('--no-white-glyph-check', action='store_true',
                    help='关闭白字自检(彩色字幕场景)')
    ap.add_argument('--threads', type=int, default=None, help='torch CPU 线程数(多 worker 并发时调小)')
    ap.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'],
                    help="推理设备:auto=有 CUDA 用 GPU(默认)")
    ap.add_argument('--inpaint-mode', default='lama', choices=['lama', 'propainter'],
                    help='lama=单帧快速;propainter=时序修复(质量高,需 GPU,显存大)')
    ap.add_argument('--no-locate-stickers', dest='locate_stickers', action='store_false',
                    help='关闭 VLM 贴纸/emoji 定位(默认开启;需 DASHSCOPE_API_KEY,'
                         '未设置时自动跳过并保留 emoji)')
    args = ap.parse_args()

    pipe = Pipeline(threads=args.threads, device=args.device, inpaint_mode=args.inpaint_mode)
    stat = pipe.process_video(
        args.input, args.output,
        region=tuple(args.region) if args.region else None,
        white_glyph_check=not args.no_white_glyph_check,
        locate_stickers=args.locate_stickers,
        ocr_stride=args.ocr_stride, ocr_refine_radius=args.ocr_refine_radius,
        vlm_max_calls=args.vlm_max_calls,
        progress=lambda d, t, s: print(f'  进度 {d}/{t} ({s})'))
    print('统计:', stat)


if __name__ == '__main__':
    main()
