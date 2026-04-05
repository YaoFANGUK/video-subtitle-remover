import os
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
        os.chmod(self.ffmpeg_path, stat.S_IRWXU + stat.S_IRWXG + stat.S_IRWXO)
        
    @property
    def ffmpeg_path(self):
        system = platform.system()
        if system == "Windows":
            ffmpeg_dir = os.path.join(BASE_DIR, 'ffmpeg', 'win_x64')
            merge_big_file_if_not_exists(ffmpeg_dir, 'ffmpeg.exe')
            return os.path.join(ffmpeg_dir, 'ffmpeg.exe')
        elif system == "Linux":
            return os.path.join(BASE_DIR, 'ffmpeg',  'linux_x64', 'ffmpeg')
        else:
            return os.path.join(BASE_DIR, 'ffmpeg', 'macos', 'ffmpeg')