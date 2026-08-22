FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_cache CC=/usr/bin/gcc CXX=/usr/bin/g++
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ ninja-build ca-certificates git rsync unzip zip procps && \
    rm -rf /var/lib/apt/lists/*
RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install numpy pandas scipy scikit-learn joblib matplotlib tqdm pyyaml pillow
WORKDIR /workspace/voxel_poleline
COPY . /workspace/voxel_poleline
RUN chmod +x /workspace/voxel_poleline/*.sh /workspace/voxel_poleline/*.py
CMD ["bash"]
