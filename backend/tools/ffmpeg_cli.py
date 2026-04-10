import os
import sys
import stat

import platform
from .common_tools import merge_big_file_if_not_exists
from backend.config import BASE_DIR

class FFmpegCLI:
    
    """
    进程管理器类，用于管理子进程的生命周期
    使用弱引用避免内存泄漏
    """
    _instance = None
    
    @classmethod
    def instance(cls):
        """单例模式获取实例"""
        if cls._instance is None:
            cls._instance = FFmpegCLI()
        return cls._instance
    
    def __init__(self):
        # 设置 FFmpeg 可执行文件权限
        try:
            os.chmod(self.ffmpeg_path, stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
        except Exception as e:
            print(f"Warning: Could not set ffmpeg executable permissions: {e}")

    @property
    def ffmpeg_path(self):
        system = platform.system()

        # 确保路径正确（打包环境 vs 开发环境）
        if getattr(sys, 'frozen', False):
            # 打包环境：BASE_DIR 指向 sys._MEIPASS
            base_path = os.path.join(BASE_DIR, 'backend')
        else:
            # 开发环境：BASE_DIR 已经是项目根目录
            base_path = BASE_DIR

        if system == "Windows":
            ffmpeg_dir = os.path.join(base_path, 'ffmpeg', 'win_x64')
            merge_big_file_if_not_exists(ffmpeg_dir, 'ffmpeg.exe')
            return os.path.join(ffmpeg_dir, 'ffmpeg.exe')
        elif system == "Linux":
            return os.path.join(base_path, 'ffmpeg',  'linux_x64', 'ffmpeg')
        else:
            return os.path.join(base_path, 'ffmpeg', 'macos', 'ffmpeg')