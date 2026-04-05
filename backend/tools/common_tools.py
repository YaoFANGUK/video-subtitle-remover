import os
import sys
import ctypes

import cv2
import numpy as np
from fsplit.filesplit import Filesplit

video_extensions = {
    '.mp4', '.m4a', '.m4v', '.f4v', '.f4a', '.m4b', '.m4r', '.f4b', '.mov',
    '.3gp', '.3gp2', '.3g2', '.3gpp', '.3gpp2', '.ogg', '.oga', '.ogv', '.ogx',
    '.wmv', '.wma', '.asf', '.webm', '.flv', '.avi', '.gifv', '.mkv', '.rm',
    '.rmvb', '.vob', '.dvd', '.mpg', '.mpeg', '.mp2', '.mpe', '.mpv', '.mpg',
    '.mpeg', '.m2v', '.svi', '.3gp', '.mxf', '.roq', '.nsv', '.flv', '.f4v',
    '.f4p', '.f4a', '.f4b'
}

image_extensions = {
    '.jpg', '.jpeg', '.jpe', '.jif', '.jfif', '.jfi', '.png', '.gif',
    '.webp', '.tiff', '.tif', '.psd', '.raw', '.arw', '.cr2', '.nrw',
    '.k25', '.bmp', '.dib', '.heif', '.heic', '.ind', '.indd', '.indt',
    '.jp2', '.j2k', '.jpf', '.jpx', '.jpm', '.mj2', '.svg', '.svgz',
    '.ai', '.eps', '.ico'
}


def is_video_file(filename):
    return os.path.splitext(filename)[-1].lower() in video_extensions


def is_image_file(filename):
    return os.path.splitext(filename)[-1].lower() in image_extensions


def is_video_or_image(filename):
    file_extension = os.path.splitext(filename)[-1].lower()
    # 检查扩展名是否在定义的视频或图片文件后缀集合中
    return file_extension in video_extensions or file_extension in image_extensions

def merge_big_file_if_not_exists(dir, file, man_filename = None):
    if file not in os.listdir(dir):
        fs = Filesplit()
        if man_filename is not None:
            fs.man_filename = man_filename
        fs.merge(input_dir=dir)

def get_readable_path(path):
    if sys.platform != 'win32':
        return path
    buf = ctypes.create_unicode_buffer(4096)
    ctypes.windll.kernel32.GetShortPathNameW(path, buf, 4096)
    return buf.value

def read_image(path):
    if os.path.getsize(path) > 100*1024*1024: # 100MB
        print(f"Image {path} is too large, skip")
        return None
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
    if img is not None and img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img