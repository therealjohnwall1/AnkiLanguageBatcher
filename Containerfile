FROM nvcr.io/nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHON_VERSION=3.10

RUN apt-get update && apt-get install -y \
    python${PYTHON_VERSION} \
    python3-pip  \
    rustc



RUN python${PYTHON_VERSION} -m pip install --upgrade pip
RUN pip install transformers accelerate sentencepiece protobuf
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

RUN pip install numpy matplotlib pandas tqdm h5py

WORKDIR /workspace
CMD ["/bin/bash"]
