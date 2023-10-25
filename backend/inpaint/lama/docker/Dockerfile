FROM nvidia/cuda:10.2-runtime-ubuntu18.04

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y wget mc tmux nano build-essential rsync libgl1

ARG USERNAME=user
RUN apt-get install -y sudo && \
    addgroup --gid 1000 $USERNAME && \
    adduser --uid 1000 --gid 1000 --disabled-password --gecos '' $USERNAME && \
    adduser $USERNAME sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
    USER=$USERNAME && \
    GROUP=$USERNAME

USER $USERNAME:$USERNAME
WORKDIR "/home/$USERNAME"
ENV PATH="/home/$USERNAME/miniconda3/bin:/home/$USERNAME/.local/bin:${PATH}"
ENV PYTHONPATH="/home/$USERNAME/project"

RUN wget -O /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-py39_4.9.2-Linux-x86_64.sh && \
    echo "536817d1b14cb1ada88900f5be51ce0a5e042bae178b5550e62f61e223deae7c /tmp/miniconda.sh" > /tmp/miniconda.sh.sha256 && \
    sha256sum --check --status < /tmp/miniconda.sh.sha256 && \
    bash /tmp/miniconda.sh -bt -p "/home/$USERNAME/miniconda3" && \
    rm /tmp/miniconda.sh && \
    conda build purge && \
    conda init

RUN pip install -U pip
RUN pip install numpy scipy torch==1.8.1 torchvision opencv-python tensorflow joblib matplotlib pandas \
    albumentations==0.5.2 pytorch-lightning==1.2.9 tabulate easydict==1.9.0 kornia==0.5.0 webdataset \
    packaging gpustat tqdm pyyaml hydra-core==1.1.0.dev6 scikit-learn==0.24.2 tabulate
RUN pip install scikit-image==0.17.2

ENV TORCH_HOME="/home/$USERNAME/.torch"

ADD entrypoint.sh /home/$USERNAME/.local/bin/entrypoint.sh
ENTRYPOINT [ "entrypoint.sh" ]
