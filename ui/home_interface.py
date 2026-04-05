import os
import cv2
import threading
import multiprocessing
import time
import traceback
from PySide6.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout
from PySide6.QtCore import Slot, QRect, Signal
from PySide6 import QtWidgets
from qfluentwidgets import (PushButton, CardWidget, PlainTextEdit, FluentIcon)
from ui.setting_interface import SettingInterface
from ui.component.video_display_component import VideoDisplayComponent
from ui.component.task_list_component import TaskListComponent, TaskStatus, TaskOptions
from ui.icon.my_fluent_icon import MyFluentIcon
from backend.config import config, tr
from backend.tools.subtitle_remover_remote_call import SubtitleRemoverRemoteCall
from backend.tools.process_manager import ProcessManager
from backend.tools.common_tools import get_readable_path, is_image_file, read_image

class HomeInterface(QWidget):
    progress_signal = Signal(int, bool) 
    append_log_signal = Signal(list)
    update_preview_with_comp_signal = Signal(list)
    task_error_signal = Signal(object)
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setObjectName("HomeInterface")
        # 初始化一些变量
        self.video_path = None
        self.video_cap = None
        self.fps = None
        self.frame_count = None
        self.frame_width = None
        self.frame_height = None
        self.se = None  # 后台字幕提取器

        # 字幕区域参数
        self.xmin = None
        self.xmax = None
        self.ymin = None
        self.ymax = None

        # 添加自动滚动控制标志
        self.auto_scroll = True
        self.running_task = False
        self.running_process = None
        
        # 当前正在处理的任务索引
        self.current_processing_task_index = -1

        self.__init_widgets()
        self.progress_signal.connect(self.update_progress)
        self.append_log_signal.connect(self.append_log)
        self.update_preview_with_comp_signal.connect(self.update_preview_with_comp)
        self.task_error_signal.connect(self.on_task_error)

    def __init_widgets(self):
        """创建主页面"""
        main_layout = QHBoxLayout(self)
        main_layout.setSpacing(8)
        main_layout.setContentsMargins(16, 16, 16, 16)

        # 左侧视频区域
        left_layout = QVBoxLayout()
        left_layout.setSpacing(8)
        
        # 创建视频显示组件
        self.video_display_component = VideoDisplayComponent(self)
        self.video_display_component.ab_sections_changed.connect(self.ab_sections_changed)
        self.video_display_component.selections_changed.connect(self.selections_changed)
        left_layout.addWidget(self.video_display_component)
        
        # 获取视频显示和滑块的引用
        self.video_display = self.video_display_component.video_display
        self.video_slider = self.video_display_component.video_slider
        self.video_slider.valueChanged.connect(self.slider_changed)
        
        # 输出文本区域
        self.output_text = PlainTextEdit()
        self.output_text.setMinimumHeight(150)
        self.output_text.setReadOnly(True)
        self.output_text.document().setDocumentMargin(10)        
        # 连接滚动条值变化信号
        self.output_text.verticalScrollBar().valueChanged.connect(self.on_scroll_change)
        
        output_container = CardWidget(self)
        output_layout = QVBoxLayout()
        output_layout.setContentsMargins(0, 0, 0, 0)
        output_layout.addWidget(self.output_text)
        output_container.setLayout(output_layout)
        left_layout.addWidget(output_container)

        main_layout.addLayout(left_layout, 2)

        # 右侧设置区域
        right_layout = QVBoxLayout()
        right_layout.setSpacing(10)

        # 设置容器
        settings_container = CardWidget(self)
        settings_container.setLayout(SettingInterface(settings_container))
        right_layout.addWidget(settings_container)
        
        # 添加任务列表容器
        task_list_container = CardWidget(self)
        task_list_layout = QHBoxLayout()
        task_list_layout.setContentsMargins(0, 0, 0, 0)
        task_list_layout.setSpacing(0)
        self.task_list_component = TaskListComponent(self)
        self.task_list_component.task_selected.connect(self.on_task_selected)
        self.task_list_component.task_deleted.connect(self.on_task_deleted)
        task_list_layout.addWidget(self.task_list_component)
        task_list_container.setLayout(task_list_layout)
        right_layout.addWidget(task_list_container, 1)  # 占满剩余空间
        
        # 操作按钮容器
        button_container = CardWidget(self)
        button_layout = QHBoxLayout()
        button_layout.setContentsMargins(16, 16, 16, 16)
        button_layout.setSpacing(8)
        
        self.file_button = PushButton(tr['SubtitleExtractorGUI']['Open'], self)
        self.file_button.setIcon(FluentIcon.FOLDER)
        self.file_button.clicked.connect(self.open_file)
        button_layout.addWidget(self.file_button)
        
        self.run_button = PushButton(tr['SubtitleExtractorGUI']['Run'], self)
        self.run_button.setIcon(FluentIcon.PLAY)
        self.run_button.clicked.connect(self.run_button_clicked)
        button_layout.addWidget(self.run_button)
        
        self.stop_button = PushButton(tr['SubtitleExtractorGUI']['Stop'], self)
        self.stop_button.setIcon(MyFluentIcon.Stop)
        self.stop_button.setVisible(False)
        self.stop_button.clicked.connect(self.stop_button_clicked)
        
        button_layout.addWidget(self.stop_button)
        
        button_container.setLayout(button_layout)
        right_layout.addWidget(button_container)

        main_layout.addLayout(right_layout, 1)
    
    def on_scroll_change(self, value):
        """监控滚动条位置变化"""
        scrollbar = self.output_text.verticalScrollBar()
        # 如果滚动到底部，启用自动滚动
        if value == scrollbar.maximum():
            self.auto_scroll = True
        # 如果用户向上滚动，禁用自动滚动
        elif self.auto_scroll and value < scrollbar.maximum():
            self.auto_scroll = False

    
    def slider_changed(self, value):
        if self.video_cap is not None and self.video_cap.isOpened():
            frame_no = self.video_slider.value()
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ret, frame = self.video_cap.read()
            if ret:
                # 更新预览图像
                self.update_preview(frame)

    def ab_sections_changed(self, ab_sections):
        get_current_task_index = self.task_list_component.get_current_task_index()
        if get_current_task_index == -1:
            return
        self.task_list_component.update_task_option(get_current_task_index, TaskOptions.AB_SECTIONS, ab_sections)

    def selections_changed(self, selections):
        get_current_task_index = self.task_list_component.get_current_task_index()
        if get_current_task_index == -1:
            return
        self.task_list_component.update_task_option(get_current_task_index, TaskOptions.SUB_AREAS, selections)

    def on_task_selected(self, index, file_path):
        """处理任务被选中事件
        
        Args:
            index: 任务索引
            file_path: 文件路径
        """
        # 加载选中的视频进行预览
        self.load_video(file_path)
        ab_sections = self.task_list_component.get_task_option(index, TaskOptions.AB_SECTIONS, [])
        self.video_display_component.set_ab_sections(ab_sections)
        selections = self.task_list_component.get_task_option(index, TaskOptions.SUB_AREAS, [])
        if len(selections) <= 0:
            self.video_display_component.load_selections_from_config()
        else:
            self.video_display_component.set_selection_rects(selections)
    
    def on_task_deleted(self, index):
        """处理任务被删除事件
        
        Args:
            index: 任务索引
        """
        # 如果删除的是正在处理的任务，则需要更新状态
        if index == self.current_processing_task_index:
            self.current_processing_task_index = -1
        
        task = self.task_list_component.get_task(0)
        if task:
            # 如果还有任务，选中第一个
            self.task_list_component.select_task(0)

    def update_preview(self, frame):
        # 先缩放图像
        resized_frame = self._img_resize(frame)

        # 设置视频参数
        self.video_display_component.set_video_parameters(
            self.frame_width, self.frame_height, 
            self.scaled_width if hasattr(self, 'scaled_width') else None,
            self.scaled_height if hasattr(self, 'scaled_height') else None,
            self.border_left if hasattr(self, 'border_left') else 0,
            self.border_top if hasattr(self, 'border_top') else 0,
            self.fps if self.fps is not None else 30,
        )
        
        # 更新视频显示（这会同时保存current_pixmap）
        self.video_display_component.update_video_display(resized_frame)

    def _img_resize(self, image):
        height, width = image.shape[:2]
        
        video_preview_width = self.video_display_component.video_preview_width
        video_preview_height = self.video_display_component.video_preview_height
        # 计算等比缩放后的尺寸
        target_ratio = video_preview_width / video_preview_height
        image_ratio = width / height
        
        if image_ratio > target_ratio:
            # 宽度适配，高度按比例缩放
            new_width = video_preview_width
            new_height = int(new_width / image_ratio)
            top_border = (video_preview_height - new_height) // 2
            bottom_border = video_preview_height - new_height - top_border
            left_border = 0
            right_border = 0
        else:
            # 高度适配，宽度按比例缩放
            new_height = video_preview_height
            new_width = int(new_height * image_ratio)
            left_border = (video_preview_width - new_width) // 2
            right_border = video_preview_width - new_width - left_border
            top_border = 0
            bottom_border = 0
        
        # 先缩放图像
        resized = cv2.resize(image, (new_width, new_height))
        
        # 添加黑边以填充到目标尺寸
        padded = cv2.copyMakeBorder(
            resized, 
            top_border, bottom_border, 
            left_border, right_border, 
            cv2.BORDER_CONSTANT, 
            value=[0, 0, 0]
        )
        
        # 保存边框信息，用于坐标转换
        self.border_left = left_border / video_preview_width
        self.border_right = right_border / video_preview_width
        self.border_top = top_border / video_preview_height
        self.border_bottom = bottom_border / video_preview_height
        self.original_width = width
        self.original_height = height
        self.is_vertical = width < height
        self.scaled_width = new_width / video_preview_width
        self.scaled_height = new_height / video_preview_height
        
        return padded

    def stop_button_clicked(self):
        try:
            self.running_task = False
            running_process = self.running_process
            if running_process:
                ProcessManager.instance().terminate_by_process(running_process)
            # 更新任务状态为待处理
            if self.current_processing_task_index >= 0:
                self.task_list_component.update_task_status(self.current_processing_task_index, TaskStatus.PENDING)
        finally:    
            self.running_process = None
            self.run_button.setVisible(True)
            self.stop_button.setVisible(False)

    def run_button_clicked(self):
        if not self.task_list_component.get_pending_tasks():
            self.append_output(tr['SubtitleExtractorGUI']['OpenVideoFirst'])
            return
            
        try:
            # 获取所有待执行的任务
            pending_tasks = self.task_list_component.get_pending_tasks()
            if not pending_tasks:
                return
            
            self.run_button.setVisible(False)
            self.stop_button.setVisible(True)
            # 开启后台线程处理视频
            def task():
                self.running_task = True
                try:
                    while self.running_task:
                        try:
                            pending_tasks = self.task_list_component.get_pending_tasks()
                            if not pending_tasks:
                                break
                            pending_task = pending_tasks[0]
                            # 更新当前处理的任务索引
                            self.current_processing_task_index, task = pending_task
                            if not self.load_video(task.path):
                                self.append_output(tr['SubtitleExtractorGUI']['OpenVideoFailed'].format(task.path))
                                self.task_list_component.update_task_status(self.current_processing_task_index, TaskStatus.FAILED)
                                continue
                            
                            # 获取字幕区域坐标
                            subtitle_areas = self.task_list_component.get_task_option(self.current_processing_task_index, TaskOptions.SUB_AREAS, [])
                            if not subtitle_areas or len(subtitle_areas) <= 0:
                                self.append_output(tr['SubtitleExtractorGUI']['SelectSubtitleArea'].format(task.path))
                                self.task_list_component.update_task_status(self.current_processing_task_index, TaskStatus.FAILED)
                                continue

                            self.video_display_component.save_selections_to_config()

                            # 更新任务状态为运行中
                            self.task_list_component.update_task_progress(self.current_processing_task_index, 1)
                            
                            # 选中当前任务
                            self.task_list_component.select_task(self.current_processing_task_index)
                            
                            if self.video_cap:
                                self.video_cap.release()
                                self.video_cap = None
                            
                            self.task_list_component.update_task_status(self.current_processing_task_index, TaskStatus.PROCESSING)
                            options = {}
                            for key in task.options:
                                value = task.options[key]
                                if key == TaskOptions.SUB_AREAS.value:
                                    value = self.video_display_component.preview_coordinates_to_video_coordinates(value)
                                options[key] = value
                            # 清理缓存, 使用动态路径
                            task.output_path = None
                            output_path = task.output_path
                            process = self.run_subtitle_remover_process(task.path, output_path, options)
                            
                            # 更新任务状态为已完成
                            task = self.task_list_component.get_task(self.current_processing_task_index)
                            if process.exitcode == 0 and task and task.status == TaskStatus.PROCESSING:
                                self.progress_signal.emit(100, True)
                                # 任务完成, 更新输出路径为只读
                                task.output_path = output_path
                                self.task_list_component.update_task_status(self.current_processing_task_index, TaskStatus.COMPLETED)
                            else:
                                self.task_list_component.update_task_status(self.current_processing_task_index, TaskStatus.FAILED)
                            
                        except Exception as e:
                            print(e)
                            self.append_output(f"Error: {e}")
                            # 更新任务状态为失败
                            if self.current_processing_task_index >= 0:
                                self.task_list_component.update_task_status(self.current_processing_task_index, TaskStatus.FAILED)
                            break
                        finally:
                            if self.video_cap:
                                self.video_cap.release()
                                self.video_cap = None
                            time.sleep(1)
                finally:
                    self.running_task = False
                    self.run_button.setVisible(True)
                    self.stop_button.setVisible(False)

            threading.Thread(target=task, daemon=True).start()
        except Exception as e:
            print(traceback.format_exc())
            self.append_output(f"Error: {e}")
            # 没有待执行的任务，恢复按钮状态
            self.run_button.setVisible(True)
            self.stop_button.setVisible(False)

    @staticmethod
    def remover_process(queue, video_path, output_path, options):
        """
        在子进程中执行字幕提取的函数
        
        Args:
            video_path: 视频文件路径
            output_path: 输出文件路径
            options: 选项
        """
        sr = None
        try:
            from backend.main import SubtitleRemover
            sr = SubtitleRemover(video_path, True)
            sr.video_out_path = output_path
            for key in options:
                setattr(sr, key, options[key])
            sr.add_progress_listener(lambda progress, isFinished: SubtitleRemoverRemoteCall.remote_call_update_progress(queue, progress, isFinished))
            sr.append_output = lambda *args: SubtitleRemoverRemoteCall.remote_call_append_log(queue, args)
            sr.manage_process = lambda pid: SubtitleRemoverRemoteCall.remote_call_manage_process(queue, pid)
            sr.update_preview_with_comp = lambda *args: SubtitleRemoverRemoteCall.remote_call_update_preview_with_comp(queue, args)
            sr.run()
        except Exception as e:
            traceback.print_exc()
            SubtitleRemoverRemoteCall.remote_call_catch_error(queue, e)
        finally:
            if sr:
                sr.isFinished = True
                sr.vsf_running = False
            SubtitleRemoverRemoteCall.remote_call_finish(queue)
            

    # 修改run_subtitle_remover_process方法
    def run_subtitle_remover_process(self, video_path, output_path, options):
        """
        使用多进程执行字幕提取，并等待进程完成
        
        Args:
            video_path: 视频文件路径
            output_path: 输出文件路径
            options: 任务选项
        """
        subtitle_remover_remote_caller = SubtitleRemoverRemoteCall()
        subtitle_remover_remote_caller.register_update_progress_callback(self.progress_signal.emit)
        subtitle_remover_remote_caller.register_log_callback(self.append_log_signal.emit)
        subtitle_remover_remote_caller.register_update_preview_with_comp_callback(self.update_preview_with_comp_signal.emit)
        subtitle_remover_remote_caller.register_error_callback(self.task_error_signal.emit)
        process = multiprocessing.Process(
            target=HomeInterface.remover_process,
            args=(subtitle_remover_remote_caller.queue, video_path, output_path, options)
        )
        try:
            if not self.running_task:
                return process
            process.start()
            ProcessManager.instance().add_process(process)
            self.running_process = process
            process.join()
            print(f"Process exited with code {process.exitcode}")
        finally:
            subtitle_remover_remote_caller.stop()
        return process

    @Slot()
    def processing_finished(self):
        pending_tasks = self.task_list_component.get_pending_tasks()
        if pending_tasks:
            # 还有待执行任务, 忽略
            return
        # 处理完成后恢复界面可用性
        self.run_button.setVisible(True)
        self.stop_button.setVisible(False)
        self.se = None
        # 重置视频滑块
        self.video_slider.setValue(1)
        # 重置当前处理任务索引
        self.current_processing_task_index = -1

    @Slot(int, bool)
    def update_progress(self, progress_total, isFinished):
        try:
            pos = min(self.frame_count - 1, int(progress_total / 100 * self.frame_count))
            if pos != self.video_slider.value():
                self.video_slider.blockSignals(True)
                self.video_slider.setValue(pos)
                self.video_slider.blockSignals(False)
            
            # 更新任务进度
            if self.current_processing_task_index >= 0:
                self.task_list_component.update_task_progress(
                    self.current_processing_task_index, 
                    progress_total,
                )
            
            # 检查是否完成
            if isFinished:
                self.processing_finished()
        except Exception as e:
            # 捕获任何异常，防止崩溃
            print(f"更新进度时出错: {str(e)}")

    @Slot(list)
    def append_log(self, log):
        self.append_output(*log)

    def append_output(self, *args):
        """添加文本到输出区域并控制滚动
        Args:
            *args: 要输出的内容，多个参数将用空格连接
        """
        # 将所有参数转换为字符串并用空格连接
        text = ' '.join(str(arg) for arg in args).rstrip()
        self.output_text.appendPlainText(text)
        print(*args)  # 保持原始的 print 行为
        # 如果启用了自动滚动，则滚动到底部
        if self.auto_scroll:
            scrollbar = self.output_text.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())

    @Slot(list)
    def update_preview_with_comp(self, args):
        """更新执行时预览"""
        frame_ori, frame_comp = args
        if self.current_processing_task_index >= 0:
            subtitle_areas = self.task_list_component.get_task_option(self.current_processing_task_index, TaskOptions.SUB_AREAS, [])
            if len(subtitle_areas) > 0:
                subtitle_areas = self.video_display_component.preview_coordinates_to_video_coordinates(subtitle_areas)
                if frame_ori is frame_comp:
                    frame_ori = frame_ori.copy()
                for rect in subtitle_areas:
                    ymin, ymax, xmin, xmax = rect
                    cv2.rectangle(frame_ori, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        preview_frame = cv2.hconcat([frame_ori, frame_comp])
        # 先缩放图像
        resized_frame = self._img_resize(preview_frame)
        # 更新视频显示（这会同时保存current_pixmap）
        self.video_display_component.update_video_display(resized_frame, draw_selection=False)
        self.video_display_component.set_dragger_enabled(False)

    @Slot(object)
    def on_task_error(self, e):
        self.append_output(tr['SubtitleExtractorGUI']['ErrorDuringProcessing'].format(str(e)))
        if self.current_processing_task_index >= 0:
            self.task_list_component.update_task_status(self.current_processing_task_index, TaskStatus.FAILED)

    def load_video(self, video_path):
        self.video_path = video_path
        if self.video_cap:
            self.video_cap.release()
            self.video_cap = None
        self.video_cap = cv2.VideoCapture(get_readable_path(self.video_path))
        if not self.video_cap.isOpened():
            return self.load_as_picture(video_path)
        ret, frame = self.video_cap.read()
        if not ret:
            return self.load_as_picture(video_path)
        self.frame_count = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_height = int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_width = int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.fps = self.video_cap.get(cv2.CAP_PROP_FPS)
        
        self.update_preview(frame)
        self.video_slider.setMaximum(self.frame_count)
        self.video_slider.setValue(1)
        self.video_display_component.set_dragger_enabled(True)
        return True

    def load_as_picture(self, path):
        if not is_image_file(path):
            return False
        self.video_path = path
        self.video_cap = None
        frame = read_image(get_readable_path(path))
        if frame is None:
            return False
        self.frame_count = 1
        self.frame_height = frame.shape[0]
        self.frame_width = frame.shape[1]
        self.fps = 1
        self.update_preview(frame)
        self.video_slider.setMaximum(self.frame_count)
        self.video_slider.setValue(1)
        self.video_display_component.set_dragger_enabled(True)
        return True


    def open_file(self):
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            tr['SubtitleExtractorGUI']['Open'],
            "",
            "All Files (*.*);;MP4 Files (*.mp4);;FLV Files (*.flv);;WMV Files (*.wmv);;AVI Files (*.avi)"
        )
        if files:
            files_loaded = []
            # 倒序打开, 确保第一个视频截图显示在屏幕上
            for path in reversed(files):
                if self.load_video(path):
                    self.append_output(f"{tr['SubtitleExtractorGUI']['OpenVideoSuccess']}: {path}")
                    files_loaded.append(path)
                else:
                    self.append_output(f"{tr['SubtitleExtractorGUI']['OpenVideoFailed']}: {path}")
            # 正序添加, 确保任务列表顺序一致
            for path in reversed(files_loaded):
                # 添加到任务列表
                self.task_list_component.add_task(path)
                index = max(0, self.task_list_component.find_task_index_by_path(path))
                self.task_list_component.select_task(index)

    def closeEvent(self, event):
        """窗口关闭时断开信号连接"""
        try:
            # 断开信号连接
            self.progress_signal.disconnect(self.update_progress)
            self.append_log_signal.disconnect(self.append_log)
            self.update_preview_with_comp_signal.disconnect(self.update_preview_with_comp)
            self.task_error_signal.disconnect(self.on_task_error)
            self.video_display_component.video_slider.valueChanged.disconnect(self.slider_changed)
            self.video_display_component.ab_sections_changed.disconnect(self.ab_sections_changed)
            self.video_display_component.selections_changed.disconnect(self.selections_changed)
            # 释放视频资源
            if self.video_cap:
                self.video_cap.release()
                self.video_cap = None
                
            # 确保所有子进程都已终止
            ProcessManager.instance().terminate_all()
        except Exception as e:
            print(f"Error during close window:", e)
        super().closeEvent(event)
    