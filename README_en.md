[简体中文](README.md) | English

## Project Introduction

![License](https://img.shields.io/badge/License-Apache%202-red.svg)
![python version](https://img.shields.io/badge/Python-3.11+-blue.svg)
![support os](https://img.shields.io/badge/OS-Windows/macOS/Linux-green.svg)

Video-subtitle-remover (VSR) is an AI-based software that removes hardcoded subtitles from videos. It mainly implements the following functionalities:

- **Lossless resolution**: Removes hardcoded subtitles from videos and generates files without subtitles.
- Fills in the removed subtitle text area using a powerful AI algorithm model (non-adjacent pixel filling and mosaic removal).
- Supports custom subtitle positions by only removing subtitles in the defined location (input position).
- Supports automatic removal of all text throughout the entire video (without inputting a position).
- Supports multi-selection of images for batch removal of watermark text.

<p style="text-align:center;"><img src="https://github.com/YaoFANGUK/video-subtitle-remover/raw/main/design/demo.png" alt="demo.png"/></p>

> Download the .zip package directly, extract, and run it. If it cannot run, follow the tutorial below to try installing the conda environment and running the source code.

**Download Links:**

Windows GPU Version v1.1.0 (GPU):

- Baidu Cloud Disk: <a href="https://pan.baidu.com/s/1zR6CjRztmOGBbOkqK8R1Ng?pwd=vsr1">vsr_windows_gpu_v1.1.0.zip</a> Extraction Code: **vsr1**

- Google Drive: <a href="https://drive.google.com/drive/folders/1NRgLNoHHOmdO4GxLhkPbHsYfMOB_3Elr?usp=sharing">vsr_windows_gpu_v1.1.0.zip</a>


**Pre-built Package Comparison**:

| Pre-built Package Name          | Python | Paddle | Torch | Environment                       | Supported Compute Capability Range |
|----------------------------------|--------|--------|--------|-----------------------------------|------------------------------------|
| `vse-windows-directml.7z`        | 3.12   | 3.0.0 | 2.4.1 | Windows without Nvidia GPU         | Universal                         |
| `vse-windows-nvidia-cuda-11.8.7z`| 3.12   | 3.0.0 | 2.7.0 | CUDA 11.8                         | 3.5 – 8.9                          |
| `vse-windows-nvidia-cuda-12.6.7z`| 3.12   | 3.0.0 | 2.7.0 | CUDA 12.6                         | 5.0 – 8.9                          |
| `vse-windows-nvidia-cuda-12.8.7z`| 3.12   | 3.0.0 | 2.7.0 | CUDA 12.8                         | 5.0 – 9.0+                          |

> NVIDIA provides a list of supported compute capabilities for each GPU model. You can refer to the following link: [CUDA GPUs](https://developer.nvidia.com/cuda-gpus) to check which CUDA version is compatible with your GPU.

**Docker Versions:**
```shell
  # Nvidia 10, 20, 30 Series Graphics Cards
  docker run -it --name vsr --gpus all eritpchy/video-subtitle-remover:1.1.1-cuda11.8 

  # Nvidia 40 Series Graphics Cards
  docker run -it --name vsr --gpus all eritpchy/video-subtitle-remover:1.1.1-cuda12.6 

  # Nvidia 50 Series Graphics Cards
  docker run -it --name vsr --gpus all eritpchy/video-subtitle-remover:1.1.1-cuda12.8 

  # AMD / Intel Dedicated or Integrated Graphics
  docker run -it --name vsr --gpus all eritpchy/video-subtitle-remover:1.1.1-directml 

  # Demo video, input
  /vsr/test/test.mp4
  docker cp vsr:/vsr/test/test_no_sub.mp4 ./
```

## Demonstration

- GUI:

<p style="text-align:center;"><img src="https://github.com/YaoFANGUK/video-subtitle-remover/raw/main/design/demo2.gif" alt="demo2.gif"/></p>

- <a href="https://b23.tv/guEbl9C">Click to view demo video👇</a>

<p style="text-align:center;"><a href="https://b23.tv/guEbl9C"><img src="https://github.com/YaoFANGUK/video-subtitle-remover/raw/main/design/demo.gif" alt="demo.gif"/></a></p>

## Source Code Usage Instructions

#### 1. Install Python

Please ensure that you have installed Python 3.12+.

- Windows users can go to the [Python official website](https://www.python.org/downloads/windows/) to download and install Python.
- MacOS users can install using Homebrew:
  ```shell
  brew install python@3.12
  ```
- Linux users can install via the package manager, such as on Ubuntu/Debian:
  ```shell
  sudo apt update && sudo apt install python3.12 python3.12-venv python3.12-dev
  ```

#### 2. Install Dependencies

It is recommended to use a virtual environment to manage project dependencies to avoid conflicts with the system environment.

(1) Create and activate the virtual environment:
```shell
python -m venv videoEnv
```

- Windows:
```shell
videoEnv\\Scripts\\activate
```
- MacOS/Linux:
```shell
source videoEnv/bin/activate
```

#### 3. Create and Activate Project Directory

Change to the directory where your source code is located:
```shell
cd <source_code_directory>
```
> For example, if your source code is in the `tools` folder on the D drive and the folder name is `video-subtitle-remover`, use:
> ```shell
> cd D:/tools/video-subtitle-remover-main
> ```

#### 4. Install the Appropriate Runtime Environment

This project supports two runtime modes: CUDA (NVIDIA GPU acceleration) and DirectML (AMD, Intel, and other GPUs/APUs).

##### (1) CUDA (For NVIDIA GPU users)

> Make sure your NVIDIA GPU driver supports the selected CUDA version.

- Recommended CUDA 11.8, corresponding to cuDNN 8.6.0.

- Install CUDA:
  - Windows: [Download CUDA 11.8](https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_522.06_windows.exe)
  - Linux:
    ```shell
    wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
    sudo sh cuda_11.8.0_520.61.05_linux.run
    ```
  - CUDA is not supported on MacOS.

- Install cuDNN (CUDA 11.8 corresponds to cuDNN 8.6.0):
  - [Windows cuDNN 8.6.0 Download](https://developer.download.nvidia.cn/compute/redist/cudnn/v8.6.0/local_installers/11.8/cudnn-windows-x86_64-8.6.0.163_cuda11-archive.zip)
  - [Linux cuDNN 8.6.0 Download](https://developer.download.nvidia.cn/compute/redist/cudnn/v8.6.0/local_installers/11.8/cudnn-linux-x86_64-8.6.0.163_cuda11-archive.tar.xz)
  - Follow the installation guide in the NVIDIA official documentation.

- Install PaddlePaddle GPU version (CUDA 11.8):
  ```shell
  pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
  ```

- Install Torch GPU version (CUDA 11.8):
  ```shell
  pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu118
  ```

- Install other dependencies:
  ```shell
  pip install -r requirements.txt
  ```

##### (2) DirectML (For AMD, Intel, and other GPU/APU users)

- Suitable for Windows devices with AMD/NVIDIA/Intel GPUs.
- Install ONNX Runtime DirectML version:
  ```shell
  pip install paddlepaddle==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
  pip install -r requirements.txt
  pip install -r requirements_directml.txt
  ```


#### 4. Run the program

- Run the graphical interface

```shell
python gui.py
```

- Run the command line version (CLI)

```shell
python ./backend/main.py
```

## Common Issues

1. How to deal with slow removal speed

You can greatly increase the removal speed by modifying the parameters in backend/config.py:

```python
MODE = InpaintMode.STTN  # Set to STTN algorithm
STTN_SKIP_DETECTION = True # Skip subtitle detection
```

2. What to do if the video removal results are not satisfactory

Modify the values in backend/config.py and try different removal algorithms. Here is an introduction to the algorithms:

> - **InpaintMode.STTN** algorithm: Good for live-action videos and fast in speed, capable of skipping subtitle detection
> - **InpaintMode.LAMA** algorithm: Best for images and effective for animated videos, moderate speed, unable to skip subtitle detection
> - **InpaintMode.PROPAINTER** algorithm: Consumes a significant amount of VRAM, slower in speed, works better for videos with very intense movement

- Using the STTN algorithm

```python
MODE = InpaintMode.STTN  # Set to STTN algorithm
# Number of neighboring frames, increasing this will increase memory usage and improve the result
STTN_NEIGHBOR_STRIDE = 10
# Length of reference frames, increasing this will increase memory usage and improve the result
STTN_REFERENCE_LENGTH = 10
# Set the maximum number of frames processed simultaneously by the STTN algorithm, a larger value leads to slower processing but better results
# Ensure that STTN_MAX_LOAD_NUM is greater than STTN_NEIGHBOR_STRIDE and STTN_REFERENCE_LENGTH
STTN_MAX_LOAD_NUM = 30
```
- Using the LAMA algorithm

```python
MODE = InpaintMode.LAMA  # Set to LAMA algorithm
LAMA_SUPER_FAST = False  # Ensure quality
```


3. CondaHTTPError

Place the .condarc file from the project in the user directory (C:/Users/<your_username>). If the file already exists in the user directory, overwrite it.

Solution: https://zhuanlan.zhihu.com/p/260034241

4. 7z file extraction error

Solution: Upgrade the 7-zip extraction program to the latest version.


