FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update --fix-missing && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    wget bzip2 ca-certificates curl git \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    git mercurial subversion nano vim \
    libosmesa6-dev libgl1-mesa-glx libglfw3 build-essential \
    git-lfs openjdk-11-jdk

# RUN apt-get clean && rm -rf /var/lib/apt/lists/*

ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin/:$PATH"

ENV HF_HOME=/rlgdg/.cache/huggingface
