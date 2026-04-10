#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基于官方 PaddleX 打包脚本改编的视频字幕去除器打包脚本
"""

import importlib.metadata
import argparse
import subprocess
import sys
import os

def get_installed_packages():
    """获取当前环境中已安装的包"""
    return [dist.metadata["Name"] for dist in importlib.metadata.distributions()]

def build_package(main_file, include_cuda=False):
    """构建打包命令"""
    # 基础命令
    cmd = [
        "pyinstaller", main_file,
        "--clean",
        "--noconfirm",
        "--name", "VideoSubtitleRemover",
        "--icon", "design/vsr.ico",
        "--console",  # 先启用控制台以便调试
    ]

    # 收集数据文件
    cmd += [
        "--add-data", "backend/models/big-lama;backend/models/big-lama",
        "--add-data", "backend/models/sttn-auto;backend/models/sttn-auto",
        "--add-data", "backend/models/sttn-det;backend/models/sttn-det",
        "--add-data", "backend/models/V5;backend/models/V5",
        "--add-data", "backend/interface;backend/interface",
        "--add-data", "backend/ffmpeg/win_x64/ffmpeg.exe;backend/ffmpeg/win_x64",
        "--add-data", "config/config.json;config",
        "--add-data", "design/vsr.ico;design",
    ]

    # 收集重要包的数据和二进制文件
    cmd += [
        "--collect-data", "paddleocr",
        "--collect-data", "skimage",
        "--collect-binaries", "paddle",
        "--collect-data", "opencv-python",
        "--collect-data", "cv2",
    ]

    # 隐藏导入
    hidden_imports = [
        # GUI 相关
        "PySide6.QtCore",
        "PySide6.QtGui",
        "PySide6.QtWidgets",
        "PySide6.QtNetwork",
        "qfluentwidgets",
        "qframelesswindow",

        # 深度学习相关
        "torch",
        "torchvision",
        "paddle",
        "paddleocr",
        "paddleocr.ppocr",
        "paddleocr.ppocr.api",
        "paddleocr.ppocr.utils",

        # 图像处理
        "cv2",
        "cv2.data",
        "cv2.matplotlib",
        "cv2.typing",
        "numpy",
        "PIL",
        "scipy",
        "scipy.spatial",
        "scipy.spatial.transform",
        "scipy.ndimage",

        # 视频处理
        "av",

        # 工具库
        "tqdm",
        "requests",
        "configparser",
        "einops",
        "darkdetect",
        "je_showinfilemanager",
        "filesplit",

        # 项目模块
        "backend",
        "backend.inpaint",
        "backend.inpaint.lama_inpaint",
        "backend.inpaint.sttn_auto_inpaint",
        "backend.inpaint.sttn_det_inpaint",
        "backend.inpaint.opencv_inpaint",
        "backend.tools",
        "backend.tools.ocr",
        "backend.tools.subtitle_detect",
        "backend.tools.subtitle_remover_remote_call",
        "backend.tools.video_io",
        "backend.tools.ffmpeg_cli",
        "backend.tools.process_manager",
        "backend.tools.hardware_accelerator",
        "backend.tools.inpaint_tools",
        "backend.tools.common_tools",
        "backend.config",
        "backend.scenedetect",
        "ui",
        "ui.home_interface",
        "ui.advanced_setting_interface",
        "ui.setting_interface",
        "ui.component",
        "ui.component.video_display_component",
        "ui.component.task_list_component",
        "ui.icon.my_fluent_icon",
    ]

    for imp in hidden_imports:
        cmd += ["--hidden-import", imp]

    # 排除不需要的模块
    excludes = [
        "pytest", "unittest", "test", "tests",
        "IPython", "jupyter", "notebook", "ipykernel",
        "paddle.fluid.contrib", "paddle.fluid.dygraph", "paddle.fluid.optimizer",
        "torch.utils.tensorboard", "torch.utils.bottleneck",
        "sphinx", "docutils", "pandas", "matplotlib", "seaborn",
        "scrapy", "beautifulsoup4", "lxml", "sklearn", "xgboost", "lightgbm",
    ]

    for exc in excludes:
        cmd += ["--exclude-module", exc]

    # CUDA 支持
    if include_cuda:
        cmd += ["--collect-binaries", "nvidia"]
        print("注意: 包含 NVIDIA CUDA 依赖")

    # 复制重要包的元数据（只复制确实存在的包）
    important_packages = ["PySide6", "paddleocr", "torch", "torchvision", "numpy", "opencv-python", "scipy", "Pillow", "opencv-contrib-python"]
    installed = get_installed_packages()

    for pkg in important_packages:
        if pkg in installed:
            try:
                cmd += ["--copy-metadata", pkg]
            except:
                print(f"Warning: Could not copy metadata for {pkg}")
                continue

    # UPX 压缩
    cmd += ["--upx-dir", "C:/upx"] if os.path.exists("C:/upx") else ["--noupx"]

    return cmd

def main():
    parser = argparse.ArgumentParser(description="视频字幕去除器打包脚本")
    parser.add_argument('--file', default='gui.py', help='主文件名，默认为 gui.py')
    parser.add_argument('--nvidia', action='store_true', help='包含 NVIDIA CUDA 和 cuDNN 依赖')
    parser.add_argument('--console', action='store_true', help='显示控制台窗口（用于调试）')

    args = parser.parse_args()
    main_file = args.file

    if not os.path.exists(main_file):
        print(f"错误: 找不到文件 {main_file}")
        sys.exit(1)

    print("========================================")
    print("  视频字幕去除器 - PyInstaller 打包")
    print("========================================")
    print(f"主文件: {main_file}")
    print(f"CUDA 支持: {'是' if args.nvidia else '否'}")
    print(f"控制台: {'显示' if args.console else '隐藏'}")
    print()

    # 构建命令
    cmd = build_package(main_file, include_cuda=args.nvidia)

    # 如果不显示控制台，添加 --noconsole
    if not args.console:
        cmd.append("--noconsole")

    print("PyInstaller 命令:")
    print(" ".join(cmd))
    print()
    print("开始打包，这可能需要几分钟...")
    print()

    try:
        # 执行打包，使用当前Python环境（vsr环境）
        result = subprocess.run(cmd, check=True, env=os.environ.copy())
        print()
        print("========================================")
        print("  ✓ 打包成功完成！")
        print("========================================")
        print(f"输出目录: dist/VideoSubtitleRemover/")
        print()
        print("下一步:")
        print("1. 测试运行: dist/VideoSubtitleRemover/VideoSubtitleRemover.exe")
        print("2. 如果测试通过，可以删除 build/ 目录以节省空间")
        return 0

    except subprocess.CalledProcessError as e:
        print()
        print("========================================")
        print("  ✗ 打包失败！")
        print("========================================")
        print(f"错误代码: {e.returncode}")
        print()
        print("故障排除:")
        print("1. 检查是否所有依赖都已安装: pip list")
        print("2. 查看详细错误日志: build/VideoSubtitleRemover/warn-VideoSubtitleRemover.txt")
        print("3. 尝试使用 --console 参数查看详细错误信息")
        return 1

if __name__ == '__main__':
    sys.exit(main())