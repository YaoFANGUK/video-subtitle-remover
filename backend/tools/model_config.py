import os
import sys
from backend.config import config, BASE_DIR
from backend.tools.common_tools import merge_big_file_if_not_exists
from backend.tools.constant import SubtitleDetectMode

_MODEL_NAME_MAP = {
    SubtitleDetectMode.PP_OCRv5_MOBILE: "PP-OCRv5_mobile_det",
    SubtitleDetectMode.PP_OCRv5_SERVER: "PP-OCRv5_server_det",
}

class ModelConfig:
    def __init__(self):
        # 确保模型路径正确（打包环境 vs 开发环境）
        if getattr(sys, 'frozen', False):
            # 打包环境：BASE_DIR 指向 sys._MEIPASS
            model_base = os.path.join(BASE_DIR, 'backend')
        else:
            # 开发环境：BASE_DIR 已经是项目根目录
            model_base = BASE_DIR

        self.LAMA_MODEL_DIR = os.path.join(model_base, 'models', 'big-lama')
        self.STTN_AUTO_MODEL_PATH = os.path.join(model_base, 'models', 'sttn-auto', 'infer_model.pth')
        self.STTN_DET_MODEL_PATH = os.path.join(model_base, 'models', 'sttn-det', 'sttn.pth')
        if config.subtitleDetectMode.value == SubtitleDetectMode.PP_OCRv5_MOBILE:
            self.DET_MODEL_DIR = os.path.join(model_base,'models', 'V5', 'ch_det_fast')
        elif config.subtitleDetectMode.value == SubtitleDetectMode.PP_OCRv5_SERVER:
            self.DET_MODEL_DIR = os.path.join(model_base, 'models', 'V5', 'ch_det')
        else:
            raise ValueError(f"Invalid subtitle detect mode: {config.subtitleDetectMode.value}")
        self.DET_MODEL_NAME = _MODEL_NAME_MAP[config.subtitleDetectMode.value]

        # 尝试合并大文件（如果需要）
        lama_file = 'big-lama.pt'  # 修正文件名：实际模型文件是 big-lama.pt
        lama_file_path = os.path.join(self.LAMA_MODEL_DIR, lama_file)
        if not os.path.exists(lama_file_path):
            merge_big_file_if_not_exists(self.LAMA_MODEL_DIR, lama_file)
            # 检查合并后文件是否存在
            if not os.path.exists(lama_file_path):
                raise FileNotFoundError(
                    f"LAMA model file not found: {lama_file_path}. "
                    f"Please ensure the model file exists in {self.LAMA_MODEL_DIR}"
                )

