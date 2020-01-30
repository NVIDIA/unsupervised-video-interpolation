# ===========
# base images
# ===========
FROM nvcr.io/nvidia/pytorch:19.04-py3


# ===============
# system packages
# ===============
RUN apt-get update
RUN apt-get install -y bash-completion \
    emacs \
    ffmpeg \
    git \
    graphviz \
    htop \
    libopenexr-dev \
    openssh-server \
    rsync \
    wget \
    curl


# ===========
# latest apex
# ===========
RUN pip uninstall -y apex
RUN git clone https://github.com/NVIDIA/apex.git ~/apex && \
    cd ~/apex && \
    pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .


# ============
# pip packages
# ============
RUN pip install --upgrade pip
RUN pip install --upgrade ffmpeg==1.4
RUN pip install --upgrade imageio==2.6.1
RUN pip install --upgrade natsort==6.2.0
RUN pip install --upgrade numpy==1.18.1
RUN pip install --upgrade pillow==6.1
RUN pip install --upgrade scikit-image==0.16.2
RUN pip install --upgrade tensorboardX==2.0
RUN pip install --upgrade torchvision==0.4.2
RUN pip install --upgrade tqdm==4.41.1
