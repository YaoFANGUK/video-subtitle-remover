
import os
from pathlib import Path
from qfluentwidgets import (qconfig, ConfigItem, QConfig, OptionsValidator, BoolValidator, OptionsConfigItem, 
                            EnumSerializer, RangeValidator, RangeConfigItem, ConfigValidator)
from backend.tools.constant import InpaintMode, SubtitleDetectMode
import configparser

# 项目版本号
VERSION = "1.4.0"
PROJECT_HOME_URL = "https://github.com/YaoFANGUK/video-subtitle-remover"
PROJECT_ISSUES_URL = PROJECT_HOME_URL + "/issues"
PROJECT_RELEASES_URL = PROJECT_HOME_URL + "/releases"
PROJECT_UPDATE_URLS = [
    "https://api.github.com/repos/YaoFANGUK/video-subtitle-remover/releases/latest",
    "https://accelerate.xdow.net/api/repos/YaoFANGUK/video-subtitle-remover/releases/latest",
] 

# 硬件加速选项开关
HARDWARD_ACCELERATION_OPTION = True

class Config(QConfig):
    # 界面语言设置
    intefaceTexts = {
        '简体中文': 'ch',
        '繁體中文': 'chinese_cht',
        'English': 'en',
        '한국어': 'ko',
        '日本語': 'japan',
        'Tiếng Việt': 'vi',
        'Español': 'es'
    }
    interface = OptionsConfigItem("Window", "Interface", "ChineseSimplified", OptionsValidator(intefaceTexts.values()), restart = True)
    
    # 窗口位置和大小
    windowX = ConfigItem("Window", "X", None)
    windowY = ConfigItem("Window", "Y", None)
    windowW = ConfigItem("Window", "Width", 1200)
    windowH = ConfigItem("Window", "Height", 1200)

    # 使用一个配置项存储所有选区
    # 默认值为一个选区，格式为："ymin,ymax,xmin,xmax;ymin,ymax,xmin,xmax;..."，分号分隔不同选区
    subtitleSelectionAreas = ConfigItem("Main", "SubtitleSelectionAreas", "0.88,0.99,0.15,0.85")

    """
    MODE可选算法类型
    - InpaintMode.STTN_AUTO 智能擦除版
    - InpaintMode.STTN_DET 带字幕检测版, 无智能擦除
    - InpaintMode.LAMA 算法：对于动画类视频效果好，速度一般，不可以跳过字幕检测
    - InpaintMode.PROPAINTER 算法： 需要消耗大量显存，速度较慢，对运动非常剧烈的视频效果较好
    """
    # 【设置inpaint算法】
    inpaintMode = OptionsConfigItem("Main", "InpaintMode", InpaintMode.STTN_AUTO, OptionsValidator(InpaintMode), EnumSerializer(InpaintMode))
    
    subtitleDetectMode =  OptionsConfigItem("Main", "SubtitleDetectMode", SubtitleDetectMode.PP_OCRv4_SERVER, OptionsValidator(SubtitleDetectMode), EnumSerializer(SubtitleDetectMode))

    # 【设置像素点偏差】
    # 用于判断是不是非字幕区域(一般认为字幕文本框的长度是要大于宽度的，如果字幕框的高大于宽，且大于的幅度超过指定像素点大小，则认为是错误检测)
    subtitleYXAxisDifferencePixel = RangeConfigItem("Main", "SubtitleYXAxisDifferencePixel", 10, RangeValidator(0, 300))
    # 用于放大mask大小，防止自动检测的文本框过小，inpaint阶段出现文字边，有残留
    subtitleAreaDeviationPixel = RangeConfigItem("Main", "SubtitleAreaDeviationPixel", 10, RangeValidator(1, 300))
    # 同于判断两个文本框是否为同一行字幕，高度差距指定像素点以内认为是同一行
    subtitleAreaYAxisDifferencePixel = RangeConfigItem("Main", "SubtitleAreaYAxisDifferencePixel", 20, RangeValidator(0, 300))
    # 用于判断两个字幕文本的矩形框是否相似，如果X轴和Y轴偏差都在指定阈值内，则认为时同一个文本框
    subtitleAreaPixelToleranceYPixel = RangeConfigItem("Main", "SubtitleAreaPixelToleranceYPixel", 20, RangeValidator(0, 300))
    subtitleAreaPixelToleranceXPixel = RangeConfigItem("Main", "SubtitleAreaPixelToleranceXPixel", 20, RangeValidator(0, 300))
    subtitleTimelineBackwardFrameCount = RangeConfigItem("Main", "SubtitleTimelineBackwardFrameCount", 3, RangeValidator(0, 300))
    subtitleTimelineForwardFrameCount = RangeConfigItem("Main", "subtitleTimelineForwardFrameCount", 3, RangeValidator(0, 300))
    # 以下参数仅适用STTN算法时，才生效
    """
    1. STTN_SKIP_DETECTION
    含义：是否使用跳过检测
    效果：设置为True跳过字幕检测，会省去很大时间，但是可能误伤无字幕的视频帧或者会导致去除的字幕漏了

    2. STTN_NEIGHBOR_STRIDE
    含义：相邻帧数步长, 如果需要为第50帧填充缺失的区域，STTN_NEIGHBOR_STRIDE=5，那么算法会使用第45帧、第40帧等作为参照。
    效果：用于控制参考帧选择的密度，较大的步长意味着使用更少、更分散的参考帧，较小的步长意味着使用更多、更集中的参考帧。

    3. STTN_REFERENCE_LENGTH
    含义：参数帧数量，STTN算法会查看每个待修复帧的前后若干帧来获得用于修复的上下文信息
    效果：调大会增加显存占用，处理效果变好，但是处理速度变慢

    4. STTN_MAX_LOAD_NUM
    含义：STTN算法每次最多加载的视频帧数量
    效果：设置越大速度越慢，但效果越好
    注意：要保证STTN_MAX_LOAD_NUM大于STTN_NEIGHBOR_STRIDE和STTN_REFERENCE_LENGTH
    """
    # 参考帧步长
    sttnNeighborStride = RangeConfigItem("Sttn", "NeighborStride", 5, RangeValidator(1, 100))
    # 参考帧数量
    sttnReferenceLength = RangeConfigItem("Sttn", "ReferenceLength", 10, RangeValidator(1, 100))
    # 设置STTN算法最大同时处理的帧数量
    sttnMaxLoadNum = RangeConfigItem("Sttn", "MaxLoadNum", 50, RangeValidator(1, 300))
    getSttnMaxLoadNum = lambda self: max(self.sttnMaxLoadNum.value, self.sttnNeighborStride.value * self.sttnReferenceLength.value)
    
    # 以下参数仅适用PROPAINTER算法时，才生效
    # 【根据自己的GPU显存大小设置】最大同时处理的图片数量，设置越大处理效果越好，但是要求显存越高
    # 1280x720p视频设置80需要25G显存，设置50需要19G显存
    # 720x480p视频设置80需要8G显存，设置50需要7G显存
    propainterMaxLoadNum = RangeConfigItem("ProPainter", "MaxLoadNum", 70, RangeValidator(1, 300))

    # 是否使用硬件加速
    hardwareAcceleration = ConfigItem("Main", "HardwareAcceleration", HARDWARD_ACCELERATION_OPTION, BoolValidator())
    
    # 启动时检查应用更新
    checkUpdateOnStartup = ConfigItem("Main", "CheckUpdateOnStartup", True, BoolValidator())

    # 视频保存目录
    saveDirectory = ConfigItem("Main", "SaveDirectory", "", ConfigValidator())

CONFIG_FILE = 'config/config.json'
config = Config()
qconfig.load(CONFIG_FILE, config)

# 读取界面语言配置
tr = configparser.ConfigParser()

TRANSLATION_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'interface', f"{config.interface.value}.ini")
tr.read(TRANSLATION_FILE, encoding='utf-8')

# 项目的base目录
BASE_DIR = str(Path(os.path.abspath(__file__)).parent)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'