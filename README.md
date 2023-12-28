简体中文 | [English](README_en.md)

## 项目简介

![License](https://img.shields.io/badge/License-Apache%202-red.svg)
![python version](https://img.shields.io/badge/Python-3.8+-blue.svg)
![support os](https://img.shields.io/badge/OS-Windows/macOS/Linux-green.svg)  

Video-subtitle-remover (VSR) 是一款基于AI技术，将视频中的硬字幕去除的软件。
主要实现了以下功能：
- **无损分辨率**将视频中的硬字幕去除，生成去除字幕后的文件
- 通过超强AI算法模型，对去除字幕文本的区域进行填充（非相邻像素填充与马赛克去除）
- 支持自定义字幕位置，仅去除定义位置中的字幕（传入位置）
- 支持全视频自动去除所有文本（不传入位置）
- 支持多选图片批量去除水印文本

<p style="text-align:center;"><img src="https://github.com/YaoFANGUK/video-subtitle-remover/raw/main/design/demo.png" alt="demo.png"/></p>

**使用说明：**

 - 有使用问题请加群讨论，QQ群：806152575
 - 直接下载压缩包解压运行，如果不能运行再按照下面的教程，尝试源码安装conda环境运行

**下载地址：**

Windows GPU版本v1.1.0（GPU）：

- 百度网盘:  <a href="https://pan.baidu.com/s/1zR6CjRztmOGBbOkqK8R1Ng?pwd=vsr1">vsr_windows_gpu_v1.1.0.zip</a> 提取码：**vsr1**

- Google Drive:  <a href="https://drive.google.com/drive/folders/1NRgLNoHHOmdO4GxLhkPbHsYfMOB_3Elr?usp=sharing">vsr_windows_gpu_v1.1.0.zip</a> 

> 仅供具有Nvidia显卡的用户使用(AMD的显卡不行)

## 演示

- GUI版：

<p style="text-align:center;"><img src="https://github.com/YaoFANGUK/video-subtitle-remover/raw/main/design/demo2.gif" alt="demo2.gif"/></p>

- <a href="https://b23.tv/guEbl9C">点击查看演示视频👇</a>

<p style="text-align:center;"><a href="https://b23.tv/guEbl9C"><img src="https://github.com/YaoFANGUK/video-subtitle-remover/raw/main/design/demo.gif" alt="demo.gif"/></a></p>

## 源码使用说明

> **无Nvidia显卡请勿使用本项目**，最低配置：
>
> **GPU**：GTX 1060或以上显卡
> 
> CPU: 支持AVX指令集

#### 1. 下载安装Miniconda 

- Windows: <a href="https://repo.anaconda.com/miniconda/Miniconda3-py38_4.11.0-Windows-x86_64.exe">Miniconda3-py38_4.11.0-Windows-x86_64.exe</a>

- Linux: <a href="https://repo.anaconda.com/miniconda/Miniconda3-py38_4.11.0-Linux-x86_64.sh">Miniconda3-py38_4.11.0-Linux-x86_64.sh</a>

#### 2. 创建并激活虚机环境

（1）切换到源码所在目录：
```shell
cd <源码所在目录>
```
> 例如：如果你的源代码放在D盘的tools文件下，并且源代码的文件夹名为video-subtitle-remover，就输入 ```cd D:/tools/video-subtitle-remover-main```

（2）创建激活conda环境
```shell
conda create -n videoEnv python=3.8
```

```shell
conda activate videoEnv
```

#### 3. 安装依赖文件

请确保你已经安装 python 3.8+，使用conda创建项目虚拟环境并激活环境 (建议创建虚拟环境运行，以免后续出现问题)

- 安装CUDA和cuDNN

  <details>
      <summary>Linux用户</summary>
      <h5>(1) 下载CUDA 11.7</h5>
      <pre><code>wget https://developer.download.nvidia.com/compute/cuda/11.7.0/local_installers/cuda_11.7.0_515.43.04_linux.run</code></pre>
      <h5>(2) 安装CUDA 11.7</h5>
      <pre><code>sudo sh cuda_11.7.0_515.43.04_linux.run</code></pre>
      <p>1. 输入accept</p>
      <img src="https://i.328888.xyz/2023/03/31/iwVoeH.png" width="500" alt="">
      <p>2. 选中CUDA Toolkit 11.7（如果你没有安装nvidia驱动则选中Driver，如果你已经安装了nvidia驱动请不要选中driver），之后选中install，回车</p>
      <img src="https://i.328888.xyz/2023/03/31/iwVThJ.png" width="500" alt="">
      <p>3. 添加环境变量</p>
      <p>在 ~/.bashrc 加入以下内容</p>
      <pre><code># CUDA
  export PATH=/usr/local/cuda-11.7/bin${PATH:+:${PATH}}
  export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}</code></pre>
      <p>使其生效</p>
      <pre><code>source ~/.bashrc</code></pre>
      <h5>(3) 下载cuDNN 8.4.1</h5>
      <p>国内：<a href="https://pan.baidu.com/s/1Gd_pSVzWfX1G7zCuqz6YYA">cudnn-linux-x86_64-8.4.1.50_cuda11.6-archive.tar.xz</a> 提取码：57mg</p>
      <p>国外：<a href="https://github.com/YaoFANGUK/video-subtitle-extractor/releases/download/1.0.0/cudnn-linux-x86_64-8.4.1.50_cuda11.6-archive.tar.xz">cudnn-linux-x86_64-8.4.1.50_cuda11.6-archive.tar.xz</a></p>
      <h5>(4) 安装cuDNN 8.4.1</h5>
      <pre><code> tar -xf cudnn-linux-x86_64-8.4.1.50_cuda11.6-archive.tar.xz
   mv cudnn-linux-x86_64-8.4.1.50_cuda11.6-archive cuda
   sudo cp ./cuda/include/* /usr/local/cuda-11.7/include/
   sudo cp ./cuda/lib/* /usr/local/cuda-11.7/lib64/
   sudo chmod a+r /usr/local/cuda-11.7/lib64/*
   sudo chmod a+r /usr/local/cuda-11.7/include/*</code></pre>
  </details>

  <details>
        <summary>Windows用户</summary>
        <h5>(1) 下载CUDA 11.7</h5>
        <a href="https://developer.download.nvidia.com/compute/cuda/11.7.0/local_installers/cuda_11.7.0_516.01_windows.exe">cuda_11.7.0_516.01_windows.exe</a>
        <h5>(2) 安装CUDA 11.7</h5>
        <h5>(3) 下载cuDNN 8.2.4</h5>
        <p><a href="https://github.com/YaoFANGUK/video-subtitle-extractor/releases/download/1.0.0/cudnn-windows-x64-v8.2.4.15.zip">cudnn-windows-x64-v8.2.4.15.zip</a></p>
        <h5>(4) 安装cuDNN 8.2.4</h5>
        <p>
           将cuDNN解压后的cuda文件夹中的bin, include, lib目录下的文件复制到C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7\对应目录下
        </p>
    </details>


- 安装GPU版本Paddlepaddle:

  - windows:

      ```shell 
      python -m pip install paddlepaddle-gpu==2.4.2.post117 -f https://www.paddlepaddle.org.cn/whl/windows/mkl/avx/stable.html
      ```

  - Linux:

      ```shell
      python -m pip install paddlepaddle-gpu==2.4.2.post117 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
      ```

- 安装GPU版本Pytorch:
      
  ```shell 
  conda install pytorch==2.0.1 torchvision==0.15.2 pytorch-cuda=11.7 -c pytorch -c nvidia
  ```
  或者使用
  ```shell 
  pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu117
  ```

- 安装其他依赖:

  ```shell
  pip install -r requirements.txt
  ```


#### 4. 运行程序

- 运行图形化界面

```shell
python gui.py
```

- 运行命令行版本(CLI)

```shell
python ./backend/main.py
```

## 常见问题
1. CondaHTTPError

将项目中的.condarc放在用户目录下(C:/Users/<你的用户名>)，如果用户目录已经存在该文件则覆盖

解决方案：https://zhuanlan.zhihu.com/p/260034241

2. 7z文件解压错误

解决方案：升级7-zip解压程序到最新版本

3. 4090使用cuda 11.7跑不起来

解决方案：改用cuda 11.8

## 赞助
<img src="https://i.imgur.com/EMCP5Lv.jpeg" width="600">

| 捐赠者 | 累计捐赠金额 | 赞助席位 |
| --- | --- | --- |
| 坤V | 400.00 RMB | 金牌赞助席位 |
| 陈凯 | 50.00 RMB | 银牌赞助席位 |
| Tshuang | 20.00 RMB | 银牌赞助席位 |
| 很奇异| 15.00 RMB | 银牌赞助席位 |
| 何斐| 10.00 RMB | 银牌赞助席位 |
| 长缨在手| 6.00 RMB | 银牌赞助席位 |
| Leo| 1.00 RMB | 铜牌赞助席位 |
