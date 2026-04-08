from enum import Enum, unique

@unique
class InpaintMode(Enum):
    """
    图像重绘算法枚举
    """
    STTN_AUTO = "sttn-auto"
    STTN_DET = "sttn-det"
    LAMA = "lama"
    PROPAINTER = "propainter"
    OPENCV = "opencv"

@unique
class SubtitleDetectMode(Enum):
    """
    字幕检测算法枚举
    """
    PP_OCRv5_MOBILE = "PP_OCRv5_MOBILE"
    PP_OCRv5_SERVER = "PP_OCRv5_SERVER"