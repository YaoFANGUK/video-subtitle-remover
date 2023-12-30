[简体中文](README.md) | English

## Project Introduction

![License](https://img.shields.io/badge/License-Apache%202-red.svg)
![python version](https://img.shields.io/badge/Python-3.8+-blue.svg)
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

> For use only by users with Nvidia graphics cards (AMD graphics cards are not supported).

## Demonstration

- GUI:

<p style="text-align:center;"><img src="https://github.com/YaoFANGUK/video-subtitle-remover/raw/main/design/demo2.gif" alt="demo2.gif"/></p>

- <a href="https://b23.tv/guEbl9C">Click to view demo video👇</a>

<p style="text-align:center;"><a href="https://b23.tv/guEbl9C"><img src="https://github.com/YaoFANGUK/video-subtitle-remover/raw/main/design/demo.gif" alt="demo.gif"/></a></p>

## Source Code Usage Instructions

> **Do not use this project without an Nvidia graphics card**. The minimum requirements are:
>
> **GPU**: GTX 1060 or higher graphics card
> 
> CPU: Supports AVX instruction set

#### 1. Download and install Miniconda

- Windows: <a href="https://repo.anaconda.com/miniconda/Miniconda3-py38_4.11.0-Windows-x86_64.exe">Miniconda3-py38_4.11.0-Windows-x86_64.exe</a>

- Linux: <a href="https://repo.anaconda.com/miniconda/Miniconda3-py38_4.11.0-Linux-x86_64.sh">Miniconda3-py38_4.11.0-Linux-x86_64.sh</a>

#### 2. Create and activate a virtual environment

(1) Switch to the source code directory:

```shell
cd <source_code_directory>
```

> For example, if your source code is in the `tools` folder on drive D, and the source code folder name is `video-subtitle-remover`, enter `cd D:/tools/video-subtitle-remover-main`.

(2) Create and activate the conda environment:

```shell
conda create -n videoEnv python=3.8
```

```shell
conda activate videoEnv
```

#### 3. Install dependencies

Please make sure you have already installed Python 3.8+, use conda to create a project virtual environment and activate the environment (it is recommended to create a virtual environment to run to avoid subsequent problems).

  - Install **CUDA** and **cuDNN**

      <details>
          <summary>Linux</summary>
          <h5>(1) Download CUDA 11.7</h5>
          <pre><code>wget https://developer.download.nvidia.com/compute/cuda/11.7.0/local_installers/cuda_11.7.0_515.43.04_linux.run</code></pre>
          <h5>(2) Install CUDA 11.7</h5>
          <pre><code>sudo sh cuda_11.7.0_515.43.04_linux.run</code></pre>
          <p>1. Input accept</p>
          <img src="https://i.328888.xyz/2023/03/31/iwVoeH.png" width="500" alt="">
          <p>2. make sure CUDA Toolkit 11.7 is chosen (If you have already installed driver, do not select Driver)</p>
          <img src="https://i.328888.xyz/2023/03/31/iwVThJ.png" width="500" alt="">
          <p>3. Add environment variables</p>
          <p>add the following content in  <strong>~/.bashrc</strong></p>
          <pre><code># CUDA
      export PATH=/usr/local/cuda-11.7/bin${PATH:+:${PATH}}
      export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}</code></pre>
          <p>Make sure it works</p>
          <pre><code>source ~/.bashrc</code></pre>
          <h5>(3) Download cuDNN 8.4.1</h5>
          <p><a href="https://github.com/YaoFANGUK/video-subtitle-extractor/releases/download/1.0.0/cudnn-linux-x86_64-8.4.1.50_cuda11.6-archive.tar.xz">cudnn-linux-x86_64-8.4.1.50_cuda11.6-archive.tar.xz</a></p>
          <h5>(4) Install cuDNN 8.4.1</h5>
          <pre><code> tar -xf cudnn-linux-x86_64-8.4.1.50_cuda11.6-archive.tar.xz
     mv cudnn-linux-x86_64-8.4.1.50_cuda11.6-archive cuda
     sudo cp ./cuda/include/* /usr/local/cuda-11.7/include/
     sudo cp ./cuda/lib/* /usr/local/cuda-11.7/lib64/
     sudo chmod a+r /usr/local/cuda-11.7/lib64/*
     sudo chmod a+r /usr/local/cuda-11.7/include/*</code></pre>
      </details>

      <details>
          <summary>Windows</summary>
          <h5>(1) Download CUDA 11.7</h5>
          <a href="https://developer.download.nvidia.com/compute/cuda/11.7.0/local_installers/cuda_11.7.0_516.01_windows.exe">cuda_11.7.0_516.01_windows.exe</a>
          <h5>(2) Install CUDA 11.7</h5>
          <h5>(3) Download cuDNN 8.2.4</h5>
          <p><a href="https://github.com/YaoFANGUK/video-subtitle-extractor/releases/download/1.0.0/cudnn-windows-x64-v8.2.4.15.zip">cudnn-windows-x64-v8.2.4.15.zip</a></p>
          <h5>(4) Install cuDNN 8.2.4</h5>
          <p>
             unzip "cudnn-windows-x64-v8.2.4.15.zip", then move all files in "bin, include, lib" in cuda 
      directory to C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7\
          </p>
      </details>


- Install GPU version of Paddlepaddle:
  - windows:

      ```shell 
      python -m pip install paddlepaddle-gpu==2.4.2.post117 -f https://www.paddlepaddle.org.cn/whl/windows/mkl/avx/stable.html
      ```

  - Linux:

      ```shell
      python -m pip install paddlepaddle-gpu==2.4.2.post117 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
      ```

- Install GPU version of Pytorch:

  ```shell 
  conda install pytorch==2.0.1 torchvision==0.15.2 pytorch-cuda=11.7 -c pytorch -c nvidia
  ```
  or use
  
  ```shell 
  pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu117
  ```

- Install other dependencies:

  ```shell
  pip install -r requirements.txt
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

