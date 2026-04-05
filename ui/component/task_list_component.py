import os
from pathlib import Path
from enum import Enum, unique
from dataclasses import dataclass
from functools import cached_property

from PySide6.QtWidgets import QWidget, QVBoxLayout, QMenu, QAbstractItemView, QTableWidgetItem, QHeaderView
from PySide6.QtCore import Qt, Signal, QModelIndex, QUrl
from qfluentwidgets import TableWidget, BodyLabel, FluentIcon, InfoBar, InfoBarPosition
from PySide6.QtGui import QAction, QColor, QBrush
from showinfm import show_in_file_manager

from backend.config import config, tr
from backend.tools.common_tools import is_image_file

@unique
class TaskStatus(Enum):
    PENDING = tr['TaskList']['Pending']
    PROCESSING = tr['TaskList']['Processing']
    COMPLETED = tr['TaskList']['Completed']
    FAILED = tr['TaskList']['Failed']


@unique
class TaskOptions(Enum):
    AB_SECTIONS = "ab_sections"
    SUB_AREAS = "sub_areas"

@dataclass
class Task:
    path: str
    name: str
    progress: int
    status: TaskStatus
    options: dict
    # 用于储存只读的输出路径, 在任务完成后设置
    _output_path: str = None

    @property
    def output_path(self):
        """获取输出路径"""
        if self._output_path is not None:
            return self._output_path
        save_directory = os.path.dirname(self.path) if not config.saveDirectory.value else config.saveDirectory.value
        if self.is_image:
            output_path = os.path.abspath(os.path.join(save_directory, f'{Path(self.path).stem}_no_sub.png'))
        else:
            output_path = os.path.abspath(os.path.join(save_directory, f'{Path(self.path).stem}_no_sub.mp4'))
        return output_path

    @output_path.setter
    def output_path(self, value):
        self._output_path = value

    @cached_property
    def is_image(self):
        """判断是否是图片文件"""
        return is_image_file(self.path)

class TaskListComponent(QWidget):
    """任务列表组件"""
    
    # 定义信号
    task_selected = Signal(int, str)  # 任务被选中时发出信号，参数为任务索引和视频路径
    task_deleted = Signal(int)  # 任务被删除时发出信号，参数为任务索引
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("TaskListComponent")
        
        # 初始化变量
        self.tasks = []  # 存储任务列表
        self.current_task_index = -1  # 当前选中的任务索引
        
        # 创建布局
        self.__init_widgets()
        
    def __init_widgets(self):
        """初始化组件"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # 创建表格
        self.table = TableWidget(self)
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels([tr['TaskList']['Name'], tr['TaskList']['Progress'], tr['TaskList']['Status']])
        
        # 设置表格样式
        self.table.setShowGrid(False)
        self.table.setAlternatingRowColors(True)
        
        # 设置列宽模式
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)           # 名称列拉伸填充
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)  # 进度列自适应内容宽度
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)  # 状态列自适应内容宽度
        
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        
        # 连接信号
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self.show_context_menu)
        self.table.clicked.connect(self.on_task_clicked)
        
        layout.addWidget(self.table)
        
    def add_task(self, video_path):
        """添加任务到列表
        
        Args:
            video_path: 视频文件路径
        """
        # 覆盖相同路径的任务
        for row, task in enumerate(self.tasks[:]):
            if task.path == video_path:
                self.delete_task(row)
                continue
                
        # 获取文件名
        file_name = os.path.basename(video_path)
        
        # 添加到任务列表
        task = Task(
            path=video_path,
            name=file_name,
            progress=0,
            status=TaskStatus.PENDING,
            options={},
        )
        self.tasks.append(task)
        
        # 更新表格
        row = len(self.tasks) - 1
        self.table.setRowCount(len(self.tasks))
        
        item0 = QTableWidgetItem(file_name)
        item1 = QTableWidgetItem("0%")
        item2 = QTableWidgetItem(TaskStatus.PENDING.value)
        
        # 设置文件名单元格的省略模式为中间省略
        item0.setTextAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        item0.setToolTip(video_path)  # 设置完整路径为工具提示
        # 设置表格的文本省略模式
        self.table.setTextElideMode(Qt.ElideMiddle)
        
        item1.setTextAlignment(Qt.AlignCenter)
        item2.setTextAlignment(Qt.AlignCenter)
        
        self.table.setItem(row, 0, item0)
        self.table.setItem(row, 1, item1)
        self.table.setItem(row, 2, item2)
        
        # 滚动到最新添加的行
        self.table.scrollToBottom()
        return True
        
    def update_task_progress(self, index, progress):
        """更新任务进度
        
        Args:
            index: 任务索引
            progress: 进度值(0-100)
        """
        if 0 <= index < len(self.tasks):
            self.tasks[index].progress = progress
            
            # 更新进度单元格
            progress_item = self.table.item(index, 1)
            if progress_item:
                progress_item.setText(f"{progress}%")
            
            # 如果是当前处理的任务，滚动到可见区域
            if index == self.current_task_index:
                self.table.scrollTo(self.table.model().index(index, 0))
                
    def update_task_status(self, index, status):
        """更新任务状态
        
        Args:
            index: 任务索引
            status: 任务状态
        """
        if 0 <= index < len(self.tasks):
            self.tasks[index].status = status
            status_item = self.table.item(index, 2)
            if status_item:
                status_item.setText(status.value)
                
                # 根据状态设置不同颜色
                if status == TaskStatus.COMPLETED:
                    status_item.setForeground(QBrush(QColor("#2ecc71")))  # 绿色
                elif status == TaskStatus.PROCESSING:
                    status_item.setForeground(QBrush(QColor("#3498db")))  # 蓝色
                elif status == TaskStatus.FAILED:
                    status_item.setForeground(QBrush(QColor("#e74c3c")))  # 红色
            
            # 如果是当前处理的任务，滚动到可见区域
            if index == self.current_task_index:
                self.table.scrollTo(self.table.model().index(index, 0))
                
            # 选中当前行
            self.table.selectRow(index)
    
    def get_pending_tasks(self):
        """获取所有待处理的任务
        
        Returns:
            list: 待处理任务列表，每项为 (索引, 任务) 元组
        """
        return [(i, task) for i, task in enumerate(self.tasks) if task.status == TaskStatus.PENDING]
    
    def get_all_tasks(self):
        """获取所有任务
        
        Returns:
            list: 所有任务列表
        """
        return self.tasks

    def get_task(self, index):
        """获取指定索引的任务

        Args:
            index: 任务索引

        Returns:
            Task: 任务对象
        """
        if 0 <= index < len(self.tasks):
            return self.tasks[index]
        return None
    
    def find_task_index_by_path(self, path):
        tasks = self.get_all_tasks()
        for idx, task in enumerate(tasks):
            if task.path == path:
                return idx
        return -1  # 没找到返回-1
        
    def show_context_menu(self, pos):
        """显示右键菜单
        
        Args:
            pos: 鼠标位置
        """
        index = self.table.indexAt(pos)
        if index.isValid():
            menu = QMenu(self)
            
            # 打开视频文件位置
            open_video_location_action = QAction(tr['TaskList']['OpenSourceVideoLocation'], self)
            open_video_location_action.triggered.connect(lambda: self.open_file_location(self.tasks[index.row()].path))
            menu.addAction(open_video_location_action)
            
            # 打开目标文件位置
            def open_target_location():
                task = self.tasks[index.row()]
                path = task.output_path
                if task.status != TaskStatus.COMPLETED:
                    InfoBar.warning(
                        title=tr['TaskList']['Warning'],
                        content=tr['TaskList']['TargetFileNotFound'],
                        parent=self.get_root_parent(),
                        duration=3000
                    )
                    return
                self.open_file_location(path)
            open_target_location_action = QAction(tr['TaskList']['OpenTargetVideoLocation'], self)
            open_target_location_action.triggered.connect(open_target_location)
            menu.addAction(open_target_location_action)

            reset_task_status_action = QAction(tr['TaskList']['ResetTaskStatus'], self)
            reset_task_status_action.triggered.connect((lambda: (
                    self.update_task_status(index.row(), TaskStatus.PENDING), 
                    self.update_task_progress(index.row(), 0)
                )
            ))
            menu.addAction(reset_task_status_action)
            
            # 删除任务
            delete_action = QAction(tr['TaskList']['DeleteTask'], self)
            delete_action.triggered.connect(lambda: self.delete_task(index.row()))
            menu.addAction(delete_action)
            
            # 显示菜单
            menu.exec_(self.table.viewport().mapToGlobal(pos))
    
    def delete_task(self, row):
        """删除任务
        
        Args:
            row: 行索引
        """
        if 0 <= row < len(self.tasks):
            # 从列表中删除
            del self.tasks[row]
            
            # 从表格中删除
            self.table.removeRow(row)
                
            # 如果删除的是当前任务，重置当前任务索引
            if row == self.current_task_index:
                self.current_task_index = -1
                
            # 发出任务删除信号
            self.task_deleted.emit(row)
    
    def on_task_clicked(self, index):
        """任务被点击时的处理
        
        Args:
            index: 索引
        """
        row = index.row()
        if 0 <= row < len(self.tasks):
            self.current_task_index = row
            # 发出信号，通知外部加载对应视频
            self.task_selected.emit(row, self.tasks[row].path)
            
    def set_current_task(self, index):
        """设置当前处理的任务
        
        Args:
            index: 任务索引
        """
        if 0 <= index < len(self.tasks):
            self.current_task_index = index
            self.table.selectRow(index)
            self.table.scrollTo(self.table.model().index(index, 0))
        
    def get_current_task_index(self):
        """获取当前处理的任务索引

        Returns:
            int: 任务索引
        """
        return self.current_task_index
            
    def select_task(self, index):
        """选中指定任务
        
        Args:
            index: 任务索引
        """
        self.set_current_task(index)
        if 0 <= index < len(self.tasks):
            self.task_selected.emit(index, self.tasks[index].path)

    def open_file_location(self, path):
        """打开文件所在位置
        
        Args:
            row: 行索引
            path: 目标路径
        """                
        # 检查视频文件是否存在
        if not os.path.exists(path):
            InfoBar.warning(
                title=tr['TaskList']['Warning'],
                content=tr['TaskList']['UnableToLocateFile'],
                parent=self.get_root_parent(),
                duration=3000
            )
            return
            
        show_in_file_manager(os.path.abspath(path))

    def get_root_parent(self):
        parent = self
        while parent.parent():
            parent = parent.parent()
        return parent

    def update_task_option(self, index, task_option: TaskOptions, value):
        """更新任务选项

        Args:
            index: 任务索引
            task_option: 选项名
            value: 选项值
        """
        if 0 <= index < len(self.tasks):
            self.tasks[index].options[task_option.value] = value

    def get_task_option(self, index, task_option: TaskOptions, default=None):
        """获取任务选项
        Args:
            index: 任务索引
            task_option: 选项名
            default: 默认值
        Returns:
            选项值
        """
        if 0 <= index < len(self.tasks):
            return self.tasks[index].options.get(task_option.value, default)