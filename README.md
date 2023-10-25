## 项目简介

![License](https://img.shields.io/badge/License-Apache%202-red.svg)
![python version](https://img.shields.io/badge/Python-3.8+-blue.svg)
![support os](https://img.shields.io/badge/OS-Windows/macOS/Linux-green.svg)  

Video-subtitle-remover (vsr) 是一款基于AI技术，将视频中的硬字幕去除的软件。


## 演示

<a href="https://b23.tv/guEbl9C"><p style="text-align:center;"><img src="https://github.com/YaoFANGUK/video-subtitle-remover/raw/main/design/demo.gif" alt="demo.gif"/>视频演示</p></a>


## 源码使用说明

#### 1. 下载安装Miniconda 

- Windows: <a href="https://repo.anaconda.com/miniconda/Miniconda3-py38_4.11.0-Windows-x86_64.exe">Miniconda3-py38_4.11.0-Windows-x86_64.exe</a>


- MacOS：<a href="https://repo.anaconda.com/miniconda/Miniconda3-py38_4.11.0-MacOSX-x86_64.pkg">Miniconda3-py38_4.11.0-MacOSX-x86_64.pkg</a>


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

- GPU用户(有N卡)：
  
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


  - 安装paddlepaddle:

    - windows:

        ```shell 
        python -m pip install paddlepaddle-gpu==2.4.2.post117 -f https://www.paddlepaddle.org.cn/whl/windows/mkl/avx/stable.html
        ```

    - Linux:

        ```shell
        python -m pip install paddlepaddle-gpu==2.4.2.post117 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
        ```

        > 如果安装cuda 10.2，请对应安装7.6.5的cuDNN，并使用对应cuda版本的paddlepaddle，**请不要使用cuDNN v8.x 和 cuda 10.2的组合**
 
        > 如果安装cuda 11.2，请对应安装8.1.1的cuDNN，并使用对应cuda版本的paddlepaddle，**30系列以上的显卡驱动可能不支持 cuda 11.2及以下版本的安装**  


  - 安装其他依赖:

    ```shell
    pip install -r requirements.txt
    ```
  

#### 4. 运行程序

- 运行命令行版本(CLI)

```shell
python ./backend/main.py
```

