import warnings
warnings.filterwarnings('ignore')
import os
import torch
import logging
import platform
import stat
from fsplit.filesplit import Filesplit
import paddle
paddle.disable_signal_handler()
logging.disable(logging.DEBUG)  # 关闭DEBUG日志的打印
logging.disable(logging.WARNING)  # 关闭WARNING日志的打印
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LAMA_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'big-lama')
STTN_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'sttn', 'infer_model.pth')
VIDEO_INPAINT_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'video')
MODEL_VERSION = 'V4'
DET_MODEL_BASE = os.path.join(BASE_DIR, 'models')
DET_MODEL_PATH = os.path.join(DET_MODEL_BASE, MODEL_VERSION, 'ch_det')

# ×××××××××××××××××××× [可以改] start ××××××××××××××××××××
# 是否使用跳过检测
SKIP_DETECTION = True
# 单个字符的高度大于宽度阈值
HEIGHT_WIDTH_DIFFERENCE_THRESHOLD = 10
# 容忍的像素点偏差
PIXEL_TOLERANCE_Y = 20  # 允许检测框纵向偏差50个像素点
PIXEL_TOLERANCE_X = 20  # 允许检测框横向偏差100个像素点
# 字幕区域偏移量， 放大诗歌像素点，防止字幕偏移
SUBTITLE_AREA_DEVIATION_PIXEL = 20
# 20个像素点以内的差距认为是同一行
TOLERANCE_Y = 20
# 高度差阈值
THRESHOLD_HEIGHT_DIFFERENCE = 20
# 相邻帧数
NEIGHBOR_STRIDE = 5
# 参考帧长度
REFERENCE_LENGTH = 5
# 模式列表，请根据自己需求选择inpaint模式
# ACCURATE模式将消耗大量GPU显存，如果您的显卡显存较少，建议设置为NORMAL
MODE_LIST = ['FAST', 'NORMAL', 'ACCURATE']
MODE = 'NORMAL'
# 【根据自己的GPU显存大小设置】最大同时处理的图片数量，设置越大处理效果越好，但是要求显存越高
# 1280x720p视频设置80需要25G显存，设置50需要19G显存
# 720x480p视频设置80需要8G显存，设置50需要7G显存
MAX_PROCESS_NUM = 70
# 【根据自己内存大小设置】设置的越大效果越好，但是时间越长
MAX_LOAD_NUM = 20
# 如果仅需要去除文字区域，则可以将SUPER_FAST设置为True
SUPER_FAST = False
# ×××××××××××××××××××× [可以改] start ××××××××××××××××××××


# ×××××××××××××××××××× [不要改] start ××××××××××××××××××××
# 查看该路径下是否有模型完整文件，没有的话合并小文件生成完整文件
if 'big-lama.pt' not in (os.listdir(LAMA_MODEL_PATH)):
    fs = Filesplit()
    fs.merge(input_dir=LAMA_MODEL_PATH)

if 'inference.pdiparams' not in os.listdir(DET_MODEL_PATH):
    fs = Filesplit()
    fs.merge(input_dir=DET_MODEL_PATH)

if 'ProPainter.pth' not in os.listdir(VIDEO_INPAINT_MODEL_PATH):
    fs = Filesplit()
    fs.merge(input_dir=VIDEO_INPAINT_MODEL_PATH)

# 指定ffmpeg可执行程序路径
sys_str = platform.system()
if sys_str == "Windows":
    ffmpeg_bin = os.path.join('win_x64', 'ffmpeg.exe')
elif sys_str == "Linux":
    ffmpeg_bin = os.path.join('linux_x64', 'ffmpeg')
else:
    ffmpeg_bin = os.path.join('macos', 'ffmpeg')
FFMPEG_PATH = os.path.join(BASE_DIR, '', 'ffmpeg', ffmpeg_bin)

if 'ffmpeg.exe' not in os.listdir(os.path.join(BASE_DIR, '', 'ffmpeg', 'win_x64')):
    fs = Filesplit()
    fs.merge(input_dir=os.path.join(BASE_DIR, '', 'ffmpeg', 'win_x64'))
# 将ffmpeg添加可执行权限
os.chmod(FFMPEG_PATH, stat.S_IRWXU+stat.S_IRWXG+stat.S_IRWXO)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
if SUPER_FAST:
    MODE = 'FAST'
if SKIP_DETECTION:
    MODE = 'NORMAL'
# ×××××××××××××××××××× [不要改] end ××××××××××××××××××××
