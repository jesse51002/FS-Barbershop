ARG PYTORCH="2.2.1"
ARG CUDA="12.1"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel as deps

SHELL ["/bin/bash", "--login", "-c"]

# Step 1. Set up Ubuntu
RUN apt update && apt install --yes wget ssh git vim build-essential gcc-12 ninja-build libgl1 libgtk2.0-dev pkg-config

FROM deps as compiler

WORKDIR /root

ENV TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6+PTX;8.9;9.0"
# ENV FORCE_CUDA="1"

RUN pip install cmake boto3 ftfy gdown imageio-ffmpeg matplotlib pandas scipy scikit-learn scikit-image sixdrepnet supervision timm tk ultralytics typing_extensions ipython
RUN pip install git+https://github.com/openai/CLIP.git

RUN git clone https://github.com/open-mmlab/mmcv.git
WORKDIR /root/mmcv
RUN git checkout v1.3.9
RUN MMCV_WITH_OPS=1 pip install -e .
WORKDIR /root


RUN git clone https://github.com/open-mmlab/mmpose.git
WORKDIR /root/mmpose
RUN mkdir -p /mmpose/data
RUN pip install --no-cache-dir -e .
WORKDIR /root


RUN pip uninstall -y opencv-python opencv-python-headless
RUN pip install opencv-python-headless


WORKDIR /root/FS-HairstyleScraping

ADD . .

CMD python src/ViTPose/pose_extract.py