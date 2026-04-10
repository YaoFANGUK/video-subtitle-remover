# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.hooks import collect_dynamic_libs
from PyInstaller.utils.hooks import copy_metadata

import os
import sys

# 使用当前工作目录作为基准路径
current_dir = os.getcwd()

datas = [
    (os.path.join(current_dir, 'backend/models/big-lama'), 'backend/models/big-lama'),
    (os.path.join(current_dir, 'backend/models/sttn-auto'), 'backend/models/sttn-auto'),
    (os.path.join(current_dir, 'backend/models/sttn-det'), 'backend/models/sttn-det'),
    (os.path.join(current_dir, 'backend/models/V5'), 'backend/models/V5'),
    (os.path.join(current_dir, 'backend/interface'), 'backend/interface'),
    (os.path.join(current_dir, 'backend/ffmpeg/win_x64/ffmpeg.exe'), 'backend/ffmpeg/win_x64'),
    (os.path.join(current_dir, 'config/config.json'), 'config'),
    (os.path.join(current_dir, 'design/vsr.ico'), 'design'),
    (os.path.join(current_dir, 'ui/icon'), 'ui/icon'),
]

binaries = []

# 收集重要包的数据文件
try:
    datas += collect_data_files('paddleocr')
except:
    pass

try:
    datas += collect_data_files('skimage')
except:
    pass

try:
    datas += collect_data_files('opencv-python')
except:
    pass

try:
    datas += collect_data_files('opencv-contrib-python')
except:
    pass

try:
    datas += collect_data_files('cv2')
except:
    pass

try:
    datas += collect_data_files('pypdfium2')
except:
    pass

try:
    datas += collect_data_files('pyclipper')
except:
    pass

try:
    datas += collect_data_files('matplotlib')
except:
    pass

try:
    datas += collect_data_files('pandas')
except:
    pass

try:
    datas += collect_data_files('huggingface_hub')
except:
    pass

try:
    datas += collect_data_files('modelscope')
except:
    pass

try:
    datas += collect_data_files('paddlex')
except:
    pass

# 收集Paddle的二进制文件
try:
    binaries += collect_dynamic_libs('paddle')
except:
    pass

# 复制重要包的元数据
try:
    datas += copy_metadata('numpy')
except:
    pass

try:
    datas += copy_metadata('PySide6')
except:
    pass

try:
    datas += copy_metadata('torch')
except:
    pass

try:
    datas += copy_metadata('torchvision')
except:
    pass

try:
    datas += copy_metadata('matplotlib')
except:
    pass

try:
    datas += copy_metadata('scipy')
except:
    pass

try:
    datas += copy_metadata('Pillow')
except:
    pass

try:
    datas += copy_metadata('opencv-python')
except:
    pass

try:
    datas += copy_metadata('opencv-contrib-python')
except:
    pass

try:
    datas += copy_metadata('pypdfium2')
except:
    pass

try:
    datas += copy_metadata('pandas')
except:
    pass

try:
    datas += copy_metadata('paddleocr')
except:
    pass

try:
    datas += copy_metadata('paddlex')
except:
    pass

try:
    datas += copy_metadata('pyclipper')
except:
    pass

hiddenimports = [
    # GUI 相关
    'PySide6.QtCore',
    'PySide6.QtGui',
    'PySide6.QtWidgets',
    'PySide6.QtNetwork',
    'qfluentwidgets',
    'qframelesswindow',

    # 深度学习相关
    'torch',
    'torchvision',
    'paddle',
    'paddleocr',
    'paddleocr.ppocr',
    'paddleocr.ppocr.api',
    'paddleocr.ppocr.utils',

    # 图像处理
    'cv2',
    'cv2.data',
    'cv2.matplotlib',
    'cv2.typing',
    'cv2.xfeatures2d',  # opencv-contrib-python
    'cv2.xphoto',  # opencv-contrib-python
    'matplotlib',
    'matplotlib.pyplot',
    'pandas',
    'pandas._libs',
    'pandas._libs.tslibs',
    'numpy',
    'PIL',
    'pypdfium2',  # PaddleOCR PDF backend
    'pypdfium2._helpers',  # PaddleOCR PDF backend
    'scipy',
    'scipy.spatial',
    'scipy.spatial.transform',
    'scipy.ndimage',

    # 视频处理
    'av',

    # 工具库
    'tqdm',
    'requests',
    'configparser',
    'einops',
    'darkdetect',
    'je_showinfilemanager',
    'filesplit',
    'pyclipper',  # PaddleOCR DBPostProcess dependency
    # PaddleX 相关依赖
    'paddlex',
    'PyYAML',
    'chardet',
    'ujson',
    'pydantic',
    'prettytable',
    'ruamel.yaml',
    'huggingface_hub',
    'modelscope',
    'colorlog',
    'filelock',
    'py_cpuinfo',

    # 项目模块
    'backend',
    'backend.inpaint',
    'backend.inpaint.lama_inpaint',
    'backend.inpaint.sttn_auto_inpaint',
    'backend.inpaint.sttn_det_inpaint',
    'backend.inpaint.opencv_inpaint',
    'backend.tools',
    'backend.tools.ocr',
    'backend.tools.subtitle_detect',
    'backend.tools.subtitle_remover_remote_call',
    'backend.tools.video_io',
    'backend.tools.ffmpeg_cli',
    'backend.tools.process_manager',
    'backend.tools.hardware_accelerator',
    'backend.tools.inpaint_tools',
    'backend.tools.common_tools',
    'backend.config',
    'backend.scenedetect',
    'ui',
    'ui.home_interface',
    'ui.advanced_setting_interface',
    'ui.setting_interface',
    'ui.component',
    'ui.component.video_display_component',
    'ui.component.task_list_component',
    'ui.icon.my_fluent_icon',
]

excludes = [
    'pytest', 'test', 'tests',
    'IPython', 'jupyter', 'notebook', 'ipykernel',
    'paddle.fluid.contrib', 'paddle.fluid.dygraph', 'paddle.fluid.optimizer',
    'torch.utils.tensorboard', 'torch.utils.bottleneck',
    'sphinx', 'docutils', 'seaborn',
    'scrapy', 'beautifulsoup4', 'lxml', 'sklearn', 'xgboost', 'lightgbm',
]

a = Analysis(
    ['gui.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='VideoSubtitleRemover',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,  # 显示控制台（调试用，后续可通过启动器隐藏）
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['design\\vsr.ico'],
    # 请求管理员权限
    uac_admin=True,  # 以管理员身份运行
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='VideoSubtitleRemover',
)