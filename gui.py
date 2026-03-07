# -*- coding: utf-8 -*-
"""
Modern GUI for video-subtitle-remover.
Features:
- Pink themed UI inspired by LosslessCut workflow.
- Video/image adaptive preview.
- Snipaste-like drag selection boxes (rect/round-rect/ellipse), multi-box support.
- B-key marks clip points, pairwise segment generation, segment export.
- Subtitle removal pipeline integration with backend.main.SubtitleRemover.
"""

from __future__ import annotations

import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import backend.main  # noqa: E402
from backend import config as backend_config  # noqa: E402


MediaPath = str
FrameNo = int
ShapeType = str


@dataclass
class SelectionBox:
    x1: int
    y1: int
    x2: int
    y2: int
    shape: ShapeType = "rect"

    def normalized(self) -> "SelectionBox":
        lx, rx = sorted([self.x1, self.x2])
        ty, by = sorted([self.y1, self.y2])
        return SelectionBox(lx, ty, rx, by, self.shape)

    def as_xyxy(self) -> Tuple[int, int, int, int]:
        n = self.normalized()
        return n.x1, n.y1, n.x2, n.y2

    def is_valid(self, min_size: int = 8) -> bool:
        x1, y1, x2, y2 = self.as_xyxy()
        return (x2 - x1) >= min_size and (y2 - y1) >= min_size


class LosslessCutLikeGUI:
    SUPPORTED_VIDEO = {".mp4", ".flv", ".wmv", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
    SUPPORTED_IMAGE = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title(f"Video Subtitle Remover v{backend.main.config.VERSION} - Pink Studio")
        self.root.geometry("1380x860")
        self.root.minsize(1120, 760)

        self._apply_theme()

        self.media_path: Optional[MediaPath] = None
        self.media_cap: Optional[cv2.VideoCapture] = None
        self.current_frame: Optional[cv2.typing.MatLike] = None
        self.preview_frame: Optional[cv2.typing.MatLike] = None
        self.is_image: bool = False

        self.frame_count: int = 1
        self.fps: float = 25.0
        self.frame_w: int = 0
        self.frame_h: int = 0
        self.current_frame_no: int = 1

        self.preview_w = 960
        self.preview_h = 540
        self.scale: float = 1.0
        self.offset_x: int = 0
        self.offset_y: int = 0

        self.selection_boxes: List[SelectionBox] = []
        self.selected_box_index: Optional[int] = None
        self.draft_start: Optional[Tuple[int, int]] = None
        self.draft_canvas_id: Optional[int] = None
        self.shape_var = tk.StringVar(value="rect")

        self.b_mark_points: List[int] = []
        self.segment_checks: List[tk.BooleanVar] = []
        self.timeline_segment_ids: List[Tuple[Tuple[int, int], int]] = []

        self.is_playing = False
        self.ignore_slider_callback = False
        self.last_image_token = None

        self.worker: Optional[threading.Thread] = None
        self.log_queue: "queue.Queue[str]" = queue.Queue()
        self.progress_queue: "queue.Queue[float]" = queue.Queue()
        self.active_sr: Optional["backend.main.SubtitleRemover"] = None
        self.active_progress_base: float = 0.0
        self.active_progress_span: float = 0.0

        self._build_layout()
        self._bind_events()
        self._poll_logs()

    def _apply_theme(self) -> None:
        self.root.configure(bg="#FFF2F7")
        style = ttk.Style(self.root)
        style.theme_use("clam")

        style.configure("Root.TFrame", background="#FFF2F7")
        style.configure("Card.TFrame", background="#FFE4EF", borderwidth=1, relief="solid")
        style.configure("Pink.TButton", background="#F06292", foreground="#ffffff", padding=(12, 7))
        style.map("Pink.TButton", background=[("active", "#EC407A")])
        style.configure("Ghost.TButton", background="#F8BBD0", foreground="#5C2843", padding=(10, 6))
        style.map("Ghost.TButton", background=[("active", "#F48FB1")])
        style.configure("TLabel", background="#FFF2F7", foreground="#4A1D35")
        style.configure("Muted.TLabel", background="#FFE4EF", foreground="#6A3550")
        style.configure("TCheckbutton", background="#FFE4EF", foreground="#4A1D35")
        style.configure("Header.TLabel", font=("Microsoft YaHei UI", 12, "bold"), background="#FFF2F7")
        style.configure("Small.TLabel", font=("Microsoft YaHei UI", 9), background="#FFE4EF", foreground="#6A3550")

    def _build_layout(self) -> None:
        root_frame = ttk.Frame(self.root, style="Root.TFrame", padding=10)
        root_frame.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(root_frame, style="Card.TFrame", padding=10)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        right = ttk.Frame(root_frame, style="Card.TFrame", padding=10, width=350)
        right.pack(side=tk.RIGHT, fill=tk.Y)
        right.pack_propagate(False)

        toolbar = ttk.Frame(left, style="Card.TFrame")
        toolbar.pack(fill=tk.X, pady=(0, 8))

        ttk.Button(toolbar, text="打开媒体", style="Pink.TButton", command=self.open_media).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(toolbar, text="播放/暂停", style="Ghost.TButton", command=self.toggle_play).pack(side=tk.LEFT, padx=6)
        ttk.Button(toolbar, text="B 标记", style="Ghost.TButton", command=self.add_b_mark).pack(side=tk.LEFT, padx=6)
        ttk.Button(toolbar, text="清空标记", style="Ghost.TButton", command=self.clear_b_marks).pack(side=tk.LEFT, padx=6)

        self.media_info_var = tk.StringVar(value="未加载媒体")
        ttk.Label(toolbar, textvariable=self.media_info_var, style="Small.TLabel").pack(side=tk.RIGHT)

        self.canvas = tk.Canvas(left, bg="#1F1020", highlightthickness=1, highlightbackground="#F8BBD0")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        timeline = ttk.Frame(left, style="Card.TFrame", padding=(0, 8, 0, 0))
        timeline.pack(fill=tk.X)

        self.frame_slider = tk.Scale(
            timeline,
            from_=1,
            to=1,
            orient=tk.HORIZONTAL,
            showvalue=False,
            bg="#FFE4EF",
            highlightthickness=0,
            troughcolor="#F48FB1",
            activebackground="#F06292",
            command=self.on_slider_change,
        )
        self.frame_slider.pack(fill=tk.X, padx=6)

        self.timeline_canvas = tk.Canvas(
            timeline,
            height=42,
            bg="#FFF4F8",
            highlightthickness=1,
            highlightbackground="#F8BBD0",
        )
        self.timeline_canvas.pack(fill=tk.X, padx=6, pady=(6, 2))

        self.frame_text_var = tk.StringVar(value="Frame 1 / 1")
        ttk.Label(timeline, textvariable=self.frame_text_var).pack(anchor=tk.W, padx=8, pady=(2, 4))

        # Right panel
        ttk.Label(right, text="选择框工具", style="Header.TLabel").pack(anchor=tk.W)

        tools = ttk.Frame(right, style="Card.TFrame")
        tools.pack(fill=tk.X, pady=(6, 10))
        for shape, name in [("rect", "矩形"), ("round", "圆角矩形"), ("ellipse", "椭圆")]:
            ttk.Radiobutton(tools, text=name, value=shape, variable=self.shape_var).pack(side=tk.LEFT, padx=8, pady=6)

        box_btn_row = ttk.Frame(right, style="Card.TFrame")
        box_btn_row.pack(fill=tk.X, pady=(0, 8))
        ttk.Button(box_btn_row, text="删除选中框", style="Ghost.TButton", command=self.delete_selected_box).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(box_btn_row, text="清空全部框", style="Ghost.TButton", command=self.clear_boxes).pack(side=tk.LEFT)

        ttk.Label(right, text="B 标记生成区间（两两配对）", style="Header.TLabel").pack(anchor=tk.W, pady=(4, 0))

        self.segment_box = ttk.Frame(right, style="Card.TFrame")
        self.segment_box.pack(fill=tk.BOTH, expand=True, pady=(6, 10))

        action_box = ttk.Frame(right, style="Card.TFrame")
        action_box.pack(fill=tk.X)

        ttk.Button(action_box, text="导出选中区间", style="Pink.TButton", command=self.export_cut_only).pack(fill=tk.X, pady=(0, 8))
        ttk.Button(action_box, text="去字幕并导出", style="Pink.TButton", command=self.process_and_export).pack(fill=tk.X, pady=(0, 8))
        ttk.Button(action_box, text="仅处理选中区间并导出完整视频", style="Ghost.TButton", command=self.process_selected_segments_export_full).pack(fill=tk.X, pady=(0, 8))

        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress = ttk.Progressbar(action_box, maximum=100.0, variable=self.progress_var)
        self.progress.pack(fill=tk.X, pady=(0, 8))

        self.log_text = tk.Text(action_box, height=11, wrap=tk.WORD, bg="#FFF8FB", fg="#4A1D35")
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def _bind_events(self) -> None:
        self.root.bind("<KeyPress-b>", lambda _e: self.add_b_mark())
        self.root.bind("<KeyPress-Delete>", lambda _e: self.delete_selected_box())
        self.root.bind("<Configure>", lambda _e: self.refresh_preview())

        self.canvas.bind("<ButtonPress-1>", self._on_canvas_down)
        self.canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_canvas_up)
        self.timeline_canvas.bind("<Button-1>", self._on_timeline_click)

    def run(self) -> None:
        self.root.mainloop()

    # -------------------- Media --------------------
    def open_media(self) -> None:
        file_path = filedialog.askopenfilename(
            title="打开视频或图片",
            filetypes=[
                ("Media", "*.mp4 *.flv *.wmv *.avi *.mov *.mkv *.webm *.m4v *.jpg *.jpeg *.png *.bmp *.webp *.tiff"),
                ("All", "*.*"),
            ],
        )
        if not file_path:
            return

        self.release_media()
        self.media_path = file_path
        ext = Path(file_path).suffix.lower()
        self.is_image = ext in self.SUPPORTED_IMAGE
        self.selection_boxes = []
        self.selected_box_index = None
        self.b_mark_points = []
        self.refresh_segment_list()

        if self.is_image:
            image = self.read_image(file_path)
            if image is None:
                messagebox.showerror("打开失败", "图片读取失败")
                return
            self.current_frame = image
            self.frame_h, self.frame_w = image.shape[:2]
            self.frame_count, self.fps, self.current_frame_no = 1, 1.0, 1
            self.frame_slider.configure(to=1)
            self._set_frame_slider(1)
        else:
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                messagebox.showerror("打开失败", "视频读取失败")
                return
            self.media_cap = cap
            self.frame_count = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1))
            self.fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
            self.frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            self.frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            self.current_frame_no = 1
            self.frame_slider.configure(to=self.frame_count)
            self._set_frame_slider(1)
            self.seek_frame(1)

        self.media_info_var.set(f"{Path(file_path).name} | {self.frame_w}x{self.frame_h}")
        self.refresh_preview()
        self.refresh_timeline()

    def release_media(self) -> None:
        self.is_playing = False
        if self.media_cap is not None:
            self.media_cap.release()
            self.media_cap = None

    def seek_frame(self, frame_no: int) -> None:
        if self.is_image or self.media_cap is None:
            return
        frame_no = max(1, min(self.frame_count, int(frame_no)))
        self.media_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no - 1)
        ok, frame = self.media_cap.read()
        if ok:
            self.current_frame = frame
            self.current_frame_no = frame_no
            self.frame_text_var.set(f"Frame {self.current_frame_no} / {self.frame_count}")
            self.refresh_preview()
            self.refresh_timeline()

    def on_slider_change(self, value: str) -> None:
        if self.ignore_slider_callback:
            return
        if self.worker and self.worker.is_alive():
            return
        if self.is_image:
            self.current_frame_no = 1
            self.frame_text_var.set("Image Mode")
            return
        self.seek_frame(int(float(value)))

    def toggle_play(self) -> None:
        if self.is_image or self.media_cap is None:
            return
        self.is_playing = not self.is_playing
        if self.is_playing:
            self._play_loop()

    def _play_loop(self) -> None:
        if not self.is_playing or self.media_cap is None:
            return
        ok, frame = self.media_cap.read()
        if not ok:
            self.is_playing = False
            return
        self.current_frame = frame
        self.current_frame_no = min(self.frame_count, self.current_frame_no + 1)
        self._set_frame_slider(self.current_frame_no)
        self.frame_text_var.set(f"Frame {self.current_frame_no} / {self.frame_count}")
        self.refresh_preview()
        self.refresh_timeline()
        delay = max(10, int(1000 / max(self.fps, 1.0)))
        self.root.after(delay, self._play_loop)

    # -------------------- Drawing --------------------
    def _on_canvas_down(self, event: tk.Event) -> None:
        if self.current_frame is None:
            return
        hit_idx = self._hit_test(event.x, event.y)
        if hit_idx is not None:
            self.selected_box_index = hit_idx
            self.refresh_preview()
            return
        ix, iy = self._canvas_to_image(event.x, event.y)
        if ix is None or iy is None:
            return
        self.draft_start = (ix, iy)

    def _on_canvas_drag(self, event: tk.Event) -> None:
        if self.current_frame is None or self.draft_start is None:
            return
        ix, iy = self._canvas_to_image(event.x, event.y)
        if ix is None or iy is None:
            return
        self.refresh_preview()
        sx, sy = self.draft_start
        cx1, cy1 = self._image_to_canvas(sx, sy)
        cx2, cy2 = self._image_to_canvas(ix, iy)
        color = "#FF4D8D"
        if self.shape_var.get() == "ellipse":
            self.draft_canvas_id = self.canvas.create_oval(cx1, cy1, cx2, cy2, outline=color, width=2, dash=(4, 2))
        else:
            self.draft_canvas_id = self.canvas.create_rectangle(cx1, cy1, cx2, cy2, outline=color, width=2, dash=(4, 2))

    def _on_canvas_up(self, event: tk.Event) -> None:
        if self.current_frame is None or self.draft_start is None:
            return
        ix, iy = self._canvas_to_image(event.x, event.y)
        if ix is None or iy is None:
            self.draft_start = None
            return
        sx, sy = self.draft_start
        box = SelectionBox(sx, sy, ix, iy, self.shape_var.get())
        if box.is_valid():
            self.selection_boxes.append(box.normalized())
            self.selected_box_index = len(self.selection_boxes) - 1
        self.draft_start = None
        self.draft_canvas_id = None
        self.refresh_preview()

    def _hit_test(self, cx: int, cy: int) -> Optional[int]:
        for i in range(len(self.selection_boxes) - 1, -1, -1):
            box = self.selection_boxes[i]
            x1, y1, x2, y2 = box.as_xyxy()
            c1x, c1y = self._image_to_canvas(x1, y1)
            c2x, c2y = self._image_to_canvas(x2, y2)
            if c1x <= cx <= c2x and c1y <= cy <= c2y:
                return i
        return None

    def delete_selected_box(self) -> None:
        if self.selected_box_index is None:
            return
        if 0 <= self.selected_box_index < len(self.selection_boxes):
            self.selection_boxes.pop(self.selected_box_index)
        self.selected_box_index = None
        self.refresh_preview()

    def clear_boxes(self) -> None:
        self.selection_boxes = []
        self.selected_box_index = None
        self.refresh_preview()

    # -------------------- B marks / segments --------------------
    def add_b_mark(self) -> None:
        if self.media_path is None:
            return
        point = 1 if self.is_image else self.current_frame_no
        if point not in self.b_mark_points:
            self.b_mark_points.append(point)
            self.b_mark_points.sort()
            self.refresh_segment_list()
            self.log(f"B 标记: frame={point}")

    def clear_b_marks(self) -> None:
        self.b_mark_points = []
        self.refresh_segment_list()

    def get_segments(self) -> List[Tuple[int, int]]:
        points = sorted(self.b_mark_points)
        pairs: List[Tuple[int, int]] = []
        for i in range(0, len(points) - 1, 2):
            a, b = points[i], points[i + 1]
            pairs.append((min(a, b), max(a, b)))
        return pairs

    def refresh_segment_list(self) -> None:
        for child in self.segment_box.winfo_children():
            child.destroy()
        self.segment_checks = []
        segments = self.get_segments()
        if not segments:
            ttk.Label(self.segment_box, text="暂无区间。按 B 键打点，两两成段。", style="Small.TLabel").pack(anchor=tk.W, padx=8, pady=8)
            self.refresh_timeline()
            return

        for idx, (start_f, end_f) in enumerate(segments, start=1):
            var = tk.BooleanVar(value=True)
            self.segment_checks.append(var)
            text = f"段 {idx}: {self._frame_to_time(start_f)} - {self._frame_to_time(end_f)} (f{start_f}-{end_f})"
            ttk.Checkbutton(self.segment_box, text=text, variable=var, command=self.refresh_timeline).pack(anchor=tk.W, padx=6, pady=2)
        self.refresh_timeline()

    def selected_segments(self) -> List[Tuple[int, int]]:
        all_segments = self.get_segments()
        if not all_segments:
            if self.is_image:
                return [(1, 1)]
            return [(1, self.frame_count)]
        selected: List[Tuple[int, int]] = []
        for seg, flag in zip(all_segments, self.segment_checks):
            if flag.get():
                selected.append(seg)
        return selected if selected else all_segments

    # -------------------- Export / process --------------------
    def export_cut_only(self) -> None:
        if not self.media_path or self.is_image:
            messagebox.showinfo("提示", "仅视频支持 Cut 导出。")
            return
        out_path = filedialog.asksaveasfilename(
            title="导出剪辑",
            defaultextension=".mp4",
            filetypes=[("MP4", "*.mp4")],
            initialfile=f"{Path(self.media_path).stem}_cut.mp4",
        )
        if not out_path:
            return
        segments = self.selected_segments()

        def task() -> None:
            try:
                self.log("开始导出剪辑...")
                self.report_progress(5)
                self.export_segments_with_ffmpeg(self.media_path, segments, out_path)
                self.report_progress(100)
                self.log(f"导出完成: {out_path}")
            except Exception as exc:
                self.log(f"导出失败: {exc}")

        self.start_worker(task)

    def process_and_export(self) -> None:
        if not self.media_path:
            return
        suffix = ".png" if self.is_image else ".mp4"
        out_path = filedialog.asksaveasfilename(
            title="导出处理结果",
            defaultextension=suffix,
            filetypes=[("Media", f"*{suffix}")],
            initialfile=f"{Path(self.media_path).stem}_processed{suffix}",
        )
        if not out_path:
            return

        def task() -> None:
            try:
                self.log("开始处理...")
                self.report_progress(1)
                boxes = [b.normalized() for b in self.selection_boxes if b.is_valid()]

                if self.is_image:
                    if boxes:
                        self.log(f"图片多框逐框处理: {len(boxes)} 个选择框")
                        self.process_image_with_boxes(self.media_path, boxes, out_path)
                        self.report_progress(100)
                        self.log(f"处理完成: {out_path}")
                        return
                    self.log("图片未选择框，使用后端自动检测处理")
                    self.active_progress_base = 10.0
                    self.active_progress_span = 85.0
                    sr = backend.main.SubtitleRemover(self.media_path, None, True)
                    self.active_sr = sr
                    sr.run()
                    self.active_sr = None
                    generated = sr.video_out_name
                    if not os.path.exists(generated):
                        raise RuntimeError("未找到输出文件")
                    os.replace(generated, out_path)
                    self.report_progress(100)
                    self.log(f"处理完成: {out_path}")
                    return

                subtitle_area = self.merged_area_for_backend()
                if subtitle_area is None:
                    self.log("未选择框，将按全画面处理")
                else:
                    self.log(f"按合并区域处理: {subtitle_area}")

                self.active_progress_base = 12.0
                self.active_progress_span = 84.0
                sr = backend.main.SubtitleRemover(self.media_path, subtitle_area, True)
                self.active_sr = sr
                sr.run()
                self.active_sr = None
                generated = sr.video_out_name
                if not os.path.exists(generated):
                    raise RuntimeError("未找到输出文件")
                os.replace(generated, out_path)
                self.report_progress(100)
                self.log(f"处理完成: {out_path}")
            except Exception as exc:
                self.active_sr = None
                self.log(f"处理失败: {exc}")

        self.start_worker(task)

    def process_selected_segments_export_full(self) -> None:
        if not self.media_path:
            return
        if self.is_image:
            messagebox.showinfo("提示", "该功能仅支持视频。")
            return

        segments = self.selected_segments()
        if segments == [(1, self.frame_count)]:
            messagebox.showinfo("提示", "请先使用 B 标记并勾选需要处理的区间。")
            return

        out_path = filedialog.asksaveasfilename(
            title="导出完整视频",
            defaultextension=".mp4",
            filetypes=[("MP4", "*.mp4")],
            initialfile=f"{Path(self.media_path).stem}_partial_processed.mp4",
        )
        if not out_path:
            return

        def task() -> None:
            try:
                self.log("开始处理选中区间并重组完整视频...")
                self.report_progress(1)
                boxes = [b.normalized() for b in self.selection_boxes if b.is_valid()]
                if not boxes:
                    boxes = [SelectionBox(0, 0, self.frame_w, self.frame_h, "rect")]
                    self.log("未选择框，将按全画面处理选中区间")
                else:
                    self.log(f"选中区间多框逐框处理: {len(boxes)} 个选择框")

                self.process_selected_segments_and_export_full_video(self.media_path, segments, boxes, out_path)
                self.report_progress(100)
                self.log(f"处理完成: {out_path}")
            except Exception as exc:
                self.active_sr = None
                self.log(f"处理失败: {exc}")

        self.start_worker(task)

    @staticmethod
    def _frame_to_sec(frame_no: int, fps: float) -> float:
        return max(0.0, (frame_no - 1) / max(1.0, fps))

    def _frame_to_time(self, frame_no: int) -> str:
        s = self._frame_to_sec(frame_no, self.fps)
        hh = int(s // 3600)
        mm = int((s % 3600) // 60)
        ss = s % 60
        return f"{hh:02d}:{mm:02d}:{ss:06.3f}"

    def export_segments_with_ffmpeg(
        self,
        input_path: str,
        segments: Sequence[Tuple[int, int]],
        output_path: str,
        precise: bool = False,
    ) -> None:
        ffmpeg = backend_config.FFMPEG_PATH
        if not ffmpeg or not os.path.exists(ffmpeg):
            raise RuntimeError("FFmpeg 不可用，请检查 backend/config.py 中的 FFMPEG_PATH")
        if not segments:
            raise RuntimeError("没有可导出的区间")

        with tempfile.TemporaryDirectory() as tmp:
            part_files = []
            for idx, (start_f, end_f) in enumerate(segments, start=1):
                part = os.path.join(tmp, f"part_{idx:03d}.mp4")
                start = self._frame_to_sec(start_f, self.fps)
                end = self._frame_to_sec(end_f + 1, self.fps)
                duration = max(0.001, end - start)
                if precise:
                    cmd = [
                        ffmpeg,
                        "-y",
                        "-i",
                        input_path,
                        "-ss",
                        f"{start:.6f}",
                        "-t",
                        f"{duration:.6f}",
                        "-c:v",
                        "libx264",
                        "-preset",
                        "fast",
                        "-crf",
                        "18",
                        "-c:a",
                        "aac",
                        "-movflags",
                        "+faststart",
                        part,
                    ]
                else:
                    cmd = [
                        ffmpeg,
                        "-y",
                        "-ss",
                        f"{start:.6f}",
                        "-to",
                        f"{end:.6f}",
                        "-i",
                        input_path,
                        "-c",
                        "copy",
                        "-avoid_negative_ts",
                        "1",
                        part,
                    ]
                self.run_cmd(cmd)
                part_files.append(part)

            concat_list = os.path.join(tmp, "list.txt")
            with open(concat_list, "w", encoding="utf-8") as f:
                for p in part_files:
                    p2 = p.replace("\\", "/").replace("'", "'\\''")
                    f.write(f"file '{p2}'\n")

            concat_cmd = [ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", concat_list, "-c", "copy", output_path]
            self.run_cmd(concat_cmd)

    def extract_audio_with_ffmpeg(self, input_path: str, output_path: str) -> None:
        ffmpeg = backend_config.FFMPEG_PATH
        if not ffmpeg or not os.path.exists(ffmpeg):
            raise RuntimeError("FFmpeg 不可用，请检查 backend/config.py 中的 FFMPEG_PATH")
        cmd = [
            ffmpeg,
            "-y",
            "-i",
            input_path,
            "-vn",
            "-acodec",
            "copy",
            output_path,
        ]
        self.run_cmd(cmd)

    def concat_video_parts_video_only_with_ffmpeg(self, parts: Sequence[str], output_path: str) -> None:
        ffmpeg = backend_config.FFMPEG_PATH
        if not ffmpeg or not os.path.exists(ffmpeg):
            raise RuntimeError("FFmpeg 不可用，请检查 backend/config.py 中的 FFMPEG_PATH")
        if not parts:
            raise RuntimeError("没有可拼接的视频片段")
        if len(parts) == 1:
            cmd = [
                ffmpeg,
                "-y",
                "-i",
                parts[0],
                "-map",
                "0:v:0",
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "18",
                "-an",
                output_path,
            ]
            self.run_cmd(cmd)
            return

        cmd = [ffmpeg, "-y"]
        for part in parts:
            cmd.extend(["-i", part])

        filter_inputs = "".join(f"[{idx}:v:0]" for idx in range(len(parts)))
        cmd.extend(
            [
                "-filter_complex",
                f"{filter_inputs}concat=n={len(parts)}:v=1:a=0[v]",
                "-map",
                "[v]",
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "18",
                "-an",
                "-movflags",
                "+faststart",
                output_path,
            ]
        )
        self.run_cmd(cmd)

    def concat_video_parts_with_ffmpeg(self, parts: Sequence[str], output_path: str) -> None:
        ffmpeg = backend_config.FFMPEG_PATH
        if not ffmpeg or not os.path.exists(ffmpeg):
            raise RuntimeError("FFmpeg 不可用，请检查 backend/config.py 中的 FFMPEG_PATH")
        if not parts:
            raise RuntimeError("没有可拼接的视频片段")
        if len(parts) == 1:
            shutil.copy2(parts[0], output_path)
            return

        cmd = [ffmpeg, "-y"]
        for part in parts:
            cmd.extend(["-i", part])

        filter_inputs = "".join(f"[{idx}:v:0][{idx}:a:0]" for idx in range(len(parts)))
        cmd.extend(
            [
                "-filter_complex",
                f"{filter_inputs}concat=n={len(parts)}:v=1:a=1[v][a]",
                "-map",
                "[v]",
                "-map",
                "[a]",
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "18",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-movflags",
                "+faststart",
                output_path,
            ]
        )
        self.run_cmd(cmd)

    def merge_audio_with_video_ffmpeg(self, video_path: str, audio_path: str, output_path: str) -> None:
        ffmpeg = backend_config.FFMPEG_PATH
        if not ffmpeg or not os.path.exists(ffmpeg):
            raise RuntimeError("FFmpeg 不可用，请检查 backend/config.py 中的 FFMPEG_PATH")
        cmd = [
            ffmpeg,
            "-y",
            "-i",
            video_path,
            "-i",
            audio_path,
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "copy",
            "-shortest",
            "-movflags",
            "+faststart",
            output_path,
        ]
        self.run_cmd(cmd)

    def process_video_with_boxes(
        self,
        input_path: str,
        boxes: Sequence[SelectionBox],
        output_path: str,
        progress_base: float,
        progress_span: float,
    ) -> None:
        pipeline_temp_paths: List[str] = []
        current_input = input_path
        total = len(boxes)
        for idx, box in enumerate(boxes, start=1):
            area = self.box_to_backend_area(box)
            self.log(f"处理框 {idx}/{total}: {area}")
            self.active_progress_base = progress_base + (idx - 1) * (progress_span / total)
            self.active_progress_span = progress_span / total
            sr = backend.main.SubtitleRemover(current_input, area, True)
            self.active_sr = sr
            sr.run()
            self.active_sr = None

            generated = sr.video_out_name
            if not os.path.exists(generated):
                raise RuntimeError(f"第 {idx} 个框处理后未找到输出文件")

            if idx < total:
                next_temp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
                next_temp.close()
                pipeline_temp_paths.append(next_temp.name)
                os.replace(generated, next_temp.name)
                if current_input != input_path and os.path.exists(current_input):
                    try:
                        os.remove(current_input)
                    except Exception:
                        pass
                current_input = next_temp.name
            else:
                os.replace(generated, output_path)

        for p in pipeline_temp_paths:
            if os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass

    def process_selected_segments_and_export_full_video(
        self,
        input_path: str,
        segments: Sequence[Tuple[int, int]],
        boxes: Sequence[SelectionBox],
        output_path: str,
    ) -> None:
        assembled_parts: List[str] = []
        temp_paths: List[str] = []
        previous_end = 0
        work_span = 60.0
        concat_base = 74.0
        per_segment_span = work_span / max(1, len(segments))

        try:
            audio_path = tempfile.NamedTemporaryFile(suffix=".mka", delete=False)
            audio_path.close()
            temp_paths.append(audio_path.name)
            self.log("分离原始音轨...")
            self.report_progress(6)
            self.extract_audio_with_ffmpeg(input_path, audio_path.name)

            for idx, (start_f, end_f) in enumerate(segments, start=1):
                if start_f > previous_end + 1:
                    untouched = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
                    untouched.close()
                    temp_paths.append(untouched.name)
                    self.export_segments_with_ffmpeg(
                        input_path,
                        [(previous_end + 1, start_f - 1)],
                        untouched.name,
                        precise=True,
                    )
                    assembled_parts.append(untouched.name)

                raw_segment = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
                raw_segment.close()
                temp_paths.append(raw_segment.name)
                self.log(f"裁剪待处理区间 {idx}/{len(segments)}: {start_f}-{end_f}")
                self.export_segments_with_ffmpeg(input_path, [(start_f, end_f)], raw_segment.name, precise=True)

                processed_segment = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
                processed_segment.close()
                temp_paths.append(processed_segment.name)
                self.process_video_with_boxes(
                    raw_segment.name,
                    boxes,
                    processed_segment.name,
                    progress_base=12.0 + (idx - 1) * per_segment_span,
                    progress_span=per_segment_span,
                )
                assembled_parts.append(processed_segment.name)
                previous_end = end_f

            if previous_end < self.frame_count:
                tail = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
                tail.close()
                temp_paths.append(tail.name)
                self.export_segments_with_ffmpeg(
                    input_path,
                    [(previous_end + 1, self.frame_count)],
                    tail.name,
                    precise=True,
                )
                assembled_parts.append(tail.name)

            stitched_video = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            stitched_video.close()
            temp_paths.append(stitched_video.name)
            self.report_progress(concat_base)
            self.log("拼接完整视频轨...")
            self.concat_video_parts_video_only_with_ffmpeg(assembled_parts, stitched_video.name)

            self.report_progress(90)
            self.log("合成原始音轨...")
            self.merge_audio_with_video_ffmpeg(stitched_video.name, audio_path.name, output_path)
        finally:
            self.active_sr = None
            for p in temp_paths:
                if os.path.exists(p):
                    try:
                        os.remove(p)
                    except Exception:
                        pass

    def process_image_with_boxes(self, input_path: str, boxes: Sequence[SelectionBox], output_path: str) -> None:
        image = self.read_image(input_path)
        if image is None:
            raise RuntimeError("图片读取失败")
        h, w = image.shape[:2]
        result = image.copy()
        total = max(1, len(boxes))
        for idx, box in enumerate(boxes, start=1):
            mask = self.build_shape_mask((h, w), box)
            result = cv2.inpaint(result, mask, 3, cv2.INPAINT_TELEA)
            self.report_progress(8 + 88 * idx / total)
        self.write_image(output_path, result)

    def build_shape_mask(self, hw: Tuple[int, int], box: SelectionBox):
        import numpy as np

        h, w = hw
        mask = np.zeros((h, w), dtype=np.uint8)
        x1, y1, x2, y2 = box.as_xyxy()
        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w - 1, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h - 1, y2))
        if box.shape == "ellipse":
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            ax, ay = max(1, (x2 - x1) // 2), max(1, (y2 - y1) // 2)
            cv2.ellipse(mask, (cx, cy), (ax, ay), 0, 0, 360, 255, -1)
        elif box.shape == "round":
            radius = max(4, min(x2 - x1, y2 - y1) // 8)
            cv2.rectangle(mask, (x1 + radius, y1), (x2 - radius, y2), 255, -1)
            cv2.rectangle(mask, (x1, y1 + radius), (x2, y2 - radius), 255, -1)
            cv2.circle(mask, (x1 + radius, y1 + radius), radius, 255, -1)
            cv2.circle(mask, (x2 - radius, y1 + radius), radius, 255, -1)
            cv2.circle(mask, (x1 + radius, y2 - radius), radius, 255, -1)
            cv2.circle(mask, (x2 - radius, y2 - radius), radius, 255, -1)
        else:
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        return mask

    def box_to_backend_area(self, box: SelectionBox) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = box.as_xyxy()
        xmin = max(0, min(self.frame_w, x1))
        xmax = max(0, min(self.frame_w, x2))
        ymin = max(0, min(self.frame_h, y1))
        ymax = max(0, min(self.frame_h, y2))
        if xmax <= xmin:
            xmax = min(self.frame_w, xmin + 1)
        if ymax <= ymin:
            ymax = min(self.frame_h, ymin + 1)
        return ymin, ymax, xmin, xmax

    def merged_area_for_backend(self) -> Optional[Tuple[int, int, int, int]]:
        if not self.selection_boxes:
            return None
        xs1, ys1, xs2, ys2 = [], [], [], []
        for box in self.selection_boxes:
            x1, y1, x2, y2 = box.as_xyxy()
            xs1.append(x1)
            ys1.append(y1)
            xs2.append(x2)
            ys2.append(y2)
        xmin = max(0, min(xs1))
        ymin = max(0, min(ys1))
        xmax = min(self.frame_w, max(xs2))
        ymax = min(self.frame_h, max(ys2))
        if xmax <= xmin or ymax <= ymin:
            return None
        self.log(f"多选框合并区域: ({ymin},{ymax},{xmin},{xmax})")
        return (ymin, ymax, xmin, xmax)

    @staticmethod
    def run_cmd(cmd: Sequence[str]) -> None:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=False)

    @staticmethod
    def read_image(path: str):
        data = cv2.imread(path)
        if data is not None:
            return data
        try:
            import numpy as np

            raw = np.fromfile(path, dtype=np.uint8)
        except Exception:
            return None
        if raw is None or raw.size == 0:
            return None
        return cv2.imdecode(raw, cv2.IMREAD_COLOR)

    @staticmethod
    def write_image(path: str, image) -> None:
        ext = Path(path).suffix or ".png"
        ok, enc = cv2.imencode(ext, image)
        if not ok:
            raise RuntimeError("图片编码失败")
        try:
            enc.tofile(path)
        except Exception as exc:
            raise RuntimeError(f"图片保存失败: {exc}") from exc

    # -------------------- Preview --------------------
    def refresh_preview(self) -> None:
        if self.current_frame is None:
            return

        canvas_w = max(300, self.canvas.winfo_width())
        canvas_h = max(200, self.canvas.winfo_height())
        h, w = self.current_frame.shape[:2]
        if h <= 0 or w <= 0:
            return

        scale = min(canvas_w / w, canvas_h / h)
        draw_w = max(1, int(w * scale))
        draw_h = max(1, int(h * scale))
        self.scale = scale
        self.offset_x = (canvas_w - draw_w) // 2
        self.offset_y = (canvas_h - draw_h) // 2

        frame = cv2.resize(self.current_frame, (draw_w, draw_h), interpolation=cv2.INTER_AREA)
        # OpenCV frame is BGR; encode directly to avoid channel swap in Tk preview.
        ok, png = cv2.imencode(".png", frame)
        if not ok:
            return

        img = tk.PhotoImage(data=png.tobytes())
        self.last_image_token = img
        self.canvas.delete("all")
        self.canvas.create_image(self.offset_x, self.offset_y, anchor=tk.NW, image=img)

        for idx, box in enumerate(self.selection_boxes):
            self._draw_box(box, selected=(idx == self.selected_box_index))

        if self.is_image:
            self.frame_text_var.set("Image Mode")
        else:
            self.frame_text_var.set(f"Frame {self.current_frame_no} / {self.frame_count}")

    def _draw_box(self, box: SelectionBox, selected: bool = False) -> None:
        x1, y1, x2, y2 = box.as_xyxy()
        cx1, cy1 = self._image_to_canvas(x1, y1)
        cx2, cy2 = self._image_to_canvas(x2, y2)
        outline = "#FF2D75" if selected else "#FF85AD"
        fill = ""

        if box.shape == "ellipse":
            self.canvas.create_oval(cx1, cy1, cx2, cy2, outline=outline, fill=fill, width=2)
        elif box.shape == "round":
            r = max(6, min(abs(cx2 - cx1), abs(cy2 - cy1)) // 8)
            self._draw_round_rect(cx1, cy1, cx2, cy2, r, outline, fill)
        else:
            self.canvas.create_rectangle(cx1, cy1, cx2, cy2, outline=outline, fill=fill, width=2)

    def _on_timeline_click(self, event: tk.Event) -> None:
        if self.is_image or self.frame_count <= 1:
            return
        w = max(1, self.timeline_canvas.winfo_width())
        x = max(12, min(w - 12, event.x))
        frame = int(round(((x - 12) / max(1, (w - 24))) * (self.frame_count - 1))) + 1
        self._set_frame_slider(frame)
        self.seek_frame(frame)

    def refresh_timeline(self) -> None:
        if not hasattr(self, "timeline_canvas"):
            return
        c = self.timeline_canvas
        c.delete("all")
        w = max(200, c.winfo_width())
        h = max(30, c.winfo_height())
        y0, y1 = 10, h - 10
        x0, x1 = 12, w - 12

        c.create_rectangle(x0, y0, x1, y1, fill="#F8BBD0", outline="#F48FB1")
        c.create_text(6, h // 2, text="I", fill="#A73867", anchor=tk.W, font=("Consolas", 9, "bold"))
        c.create_text(w - 6, h // 2, text="O", fill="#A73867", anchor=tk.E, font=("Consolas", 9, "bold"))

        def f2x(frame_no: int) -> int:
            if self.frame_count <= 1:
                return x0
            ratio = (frame_no - 1) / (self.frame_count - 1)
            return int(x0 + ratio * (x1 - x0))

        self.timeline_segment_ids = []
        segments = self.get_segments()
        for idx, (seg, var) in enumerate(zip(segments, self.segment_checks), start=1):
            s, e = seg
            sx, ex = f2x(s), f2x(e)
            fill = "#F06292" if var.get() else "#E2A5BE"
            item = c.create_rectangle(sx, y0 + 2, max(sx + 2, ex), y1 - 2, fill=fill, outline="")
            self.timeline_segment_ids.append((seg, item))
            c.create_text((sx + ex) // 2, y0 - 1, text=str(idx), fill="#7A2E4F", anchor=tk.S, font=("Consolas", 8))

        for p in self.b_mark_points:
            px = f2x(p)
            c.create_line(px, y0 - 3, px, y1 + 3, fill="#C2185B", width=2)

        if not self.is_image:
            hx = f2x(self.current_frame_no)
            c.create_polygon(hx - 5, y0 - 5, hx + 5, y0 - 5, hx, y0 + 3, fill="#6A1B9A", outline="")
            c.create_line(hx, y0, hx, y1, fill="#6A1B9A", width=1)

    def _draw_round_rect(self, x1: int, y1: int, x2: int, y2: int, r: int, outline: str, fill: str) -> None:
        x1, x2 = sorted([x1, x2])
        y1, y2 = sorted([y1, y2])
        points = [
            x1 + r, y1,
            x2 - r, y1,
            x2, y1,
            x2, y1 + r,
            x2, y2 - r,
            x2, y2,
            x2 - r, y2,
            x1 + r, y2,
            x1, y2,
            x1, y2 - r,
            x1, y1 + r,
            x1, y1,
        ]
        self.canvas.create_polygon(points, smooth=True, splinesteps=24, outline=outline, fill=fill, width=2)

    def _canvas_to_image(self, cx: int, cy: int) -> Tuple[Optional[int], Optional[int]]:
        ix = int((cx - self.offset_x) / max(self.scale, 1e-6))
        iy = int((cy - self.offset_y) / max(self.scale, 1e-6))
        if ix < 0 or iy < 0 or ix >= self.frame_w or iy >= self.frame_h:
            return None, None
        return ix, iy

    def _image_to_canvas(self, ix: int, iy: int) -> Tuple[int, int]:
        cx = int(ix * self.scale + self.offset_x)
        cy = int(iy * self.scale + self.offset_y)
        return cx, cy

    # -------------------- Worker/log --------------------
    def start_worker(self, fn) -> None:
        if self.worker and self.worker.is_alive():
            messagebox.showinfo("忙碌", "当前有任务正在执行")
            return
        self.worker = threading.Thread(target=fn, daemon=True)
        self.worker.start()

    def log(self, msg: str) -> None:
        self.log_queue.put(msg)

    def report_progress(self, value: float) -> None:
        self.progress_queue.put(float(value))

    def _set_frame_slider(self, value: int) -> None:
        self.ignore_slider_callback = True
        try:
            self.frame_slider.set(value)
        finally:
            self.ignore_slider_callback = False

    def _poll_logs(self) -> None:
        if self.active_sr is not None:
            try:
                p = max(0.0, min(100.0, float(self.active_sr.progress_total)))
                self.progress_var.set(min(99.0, self.active_progress_base + p * self.active_progress_span / 100.0))
                if self.active_sr.preview_frame is not None:
                    self.current_frame = self.active_sr.preview_frame
                    self.refresh_preview()
            except Exception:
                pass
        while True:
            try:
                value = self.progress_queue.get_nowait()
            except queue.Empty:
                break
            self.progress_var.set(value)
        while True:
            try:
                msg = self.log_queue.get_nowait()
            except queue.Empty:
                break
            self.log_text.insert(tk.END, msg + "\n")
            self.log_text.see(tk.END)
        self.refresh_timeline()
        self.root.after(120, self._poll_logs)


if __name__ == "__main__":
    app = LosslessCutLikeGUI()
    app.run()
