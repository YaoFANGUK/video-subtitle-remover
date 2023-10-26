import warnings
warnings.filterwarnings("ignore")
import os
import torch
import logging
from fsplit.filesplit import Filesplit

logging.disable(logging.DEBUG)  # 关闭DEBUG日志的打印
logging.disable(logging.WARNING)  # 关闭WARNING日志的打印
device = "cuda" if torch.cuda.is_available() else "cpu"
BASE_DIR = os.path.dirname(__file__)
LAMA_CONFIG = os.path.join(BASE_DIR, 'inpaint', 'lama', 'configs', 'prediction', 'default.yaml')
LAMA_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'big-lama')
MODEL_VERSION = 'V4'
DET_MODEL_BASE = os.path.join(BASE_DIR, 'models')
DET_MODEL_PATH = os.path.join(DET_MODEL_BASE, MODEL_VERSION, 'ch_det')
# 字幕区域偏移量
SUBTITLE_AREA_DEVIATION_PIXEL = 50

# 查看该路径下是否有模型完整文件，没有的话合并小文件生成完整文件
if 'best.ckpt' not in (os.listdir(os.path.join(LAMA_MODEL_PATH, 'models'))):
    fs = Filesplit()
    fs.merge(input_dir=os.path.join(LAMA_MODEL_PATH, 'models'))

if 'inference.pdiparams' not in DET_MODEL_PATH:
    fs = Filesplit()
    fs.merge(input_dir=DET_MODEL_PATH)
