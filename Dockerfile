FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install sytem dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends gnupg dirmngr ca-certificates \
    && apt-key adv --keyserver-options timeout=60 --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub \
    && apt-key adv --keyserver-options timeout=60 --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64/7fa2af80.pub \
    && apt-get update \
    && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libxrender-dev cmake \
    && apt-get install -y build-essential software-properties-common \
    && add-apt-repository ppa:ubuntu-toolchain-r/test \
    && apt-get update \
    && apt-get install -y gcc-9 g++-9 \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 60 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 60 \
    && apt-get install -y python3-dev python3-pip python3-distutils \
    && apt-get install -y --no-install-recommends libopenblas-dev nvidia-utils-530 \
    && apt-get install wget

# Symlink python3 to python
RUN ln -sf /usr/bin/python3 /usr/bin/python

# Install newer CMake
RUN apt-get update && apt-get install -y apt-transport-https ca-certificates gnupg software-properties-common && \
    wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc | gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null && \
    apt-add-repository 'deb https://apt.kitware.com/ubuntu/ focal main' && \
    apt-get update && apt-get install -y cmake

# Set path variables
ENV PATH=/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Install python debugger
RUN pip install debugpy
# Install PyTorch with CUDA 11.6
RUN pip install typing-extensions==4.13.2 \
    && pip install torch==1.13.1+cu116 \
    torchvision==0.14.1+cu116 \
    torchaudio==0.13.1 \
    --extra-index-url https://download.pytorch.org/whl/cu116
# Install openMMLab
RUN pip install --no-deps \
    mmengine==0.7.3 \
    mmdet==3.0.0 \
    mmsegmentation==1.0.0 \
    git+https://github.com/open-mmlab/mmdetection3d.git@22aaa47fdb53ce1870ff92cb7e3f96ae38d17f61
# Install mmcv
RUN pip install mmcv==2.0.0 -f https://download.openmmlab.com/mmcv/dist/cu116/torch1.13.0/index.html --no-deps
# Install MinkowskiEngine
RUN apt-get update \
    && apt-get -y install libopenblas-dev nvidia-cuda-dev
RUN TORCH_CUDA_ARCH_LIST="8.0" \ 
    pip install git+https://github.com/NVIDIA/MinkowskiEngine.git@02fc608bea4c0549b0a7b00ca1bf15dee4a0b228 -v --no-deps \
    --install-option="--blas=openblas" \
    --install-option="--force_cuda"
# Install torch-scatter
RUN git clone https://github.com/rusty1s/pytorch_scatter.git \
    && cd pytorch_scatter \
    && git checkout tags/2.0.9 -b v2.0.9 \
    && TORCH_CUDA_ARCH_LIST="6.1;7.0;8.0" FORCE_CUDA=1 pip install .
# Install torch-sparse and torch-geometric
RUN pip install pyg-lib torch-sparse -f https://data.pyg.org/whl/torch-1.13.0+cu116.html
RUN pip install torch-geometric

# Install pip packages
RUN pip install --no-deps \
    spconv-cu116==2.3.6 \
    addict==2.4.0 \
    yapf==0.33.0 \
    termcolor==2.3.0 \
    packaging==23.1 \
    numpy==1.24.1 \
    rich==13.3.5 \
    opencv-python==4.7.0.72 \
    pycocotools==2.0.6 \
    Shapely==1.8.5 \
    scipy==1.10.1 \
    terminaltables==3.1.10 \
    numba==0.57.0 \
    llvmlite==0.40.0 \
    pccm==0.4.7 \
    ccimport==0.4.2 \
    pybind11==2.10.4 \
    ninja==1.11.1 \
    lark==1.1.5 \
    cumm-cu116==0.4.9 \
    pyquaternion==0.9.9 \
    lyft-dataset-sdk==0.0.8 \
    pandas==2.0.1 \
    python-dateutil==2.8.2 \
    matplotlib==3.5.2 \
    pyparsing==3.0.9 \
    cycler==0.11.0 \
    kiwisolver==1.4.4 \
    scikit-learn==1.2.2 \
    joblib==1.2.0 \
    threadpoolctl==3.1.0 \
    cachetools==5.3.0 \
    nuscenes-devkit==1.1.10 \
    trimesh==3.21.6 \
    open3d==0.13.0 \
    plotly==5.18.0 \
    dash==2.14.2 \
    plyfile==1.0.2 \
    flask==3.0.0 \
    werkzeug==3.0.1 \
    click==8.1.7 \
    blinker==1.7.0 \
    itsdangerous==2.1.2 \
    importlib_metadata==2.1.2 \
    zipp==3.17.0 \
    tensorboard==2.14.0 \
    tensorboard-data-server==0.7.2 \
    laspy==2.5.4 \
    protobuf \
    absl-py \
    future \
    MarkupSafe==2.0.1 \
    markdown \
    grpcio \
    google-auth-oauthlib \
    google-auth \
    requests-oauthlib \
    oauthlib \
    pyyaml==6.0.2 \
    pytz==2025.2 \
    tqdm==4.67.1 \
    omegaconf==2.2.0 \
    antlr4-python3-runtime==4.8 \
    tabulate==0.9.0


# Install NVIDIA utils
RUN apt-get update && apt-get install -y nvidia-utils-530

# Set PYTHONPATH
ENV PYTHONPATH=/workspaces/scatr_its
COPY . /workspaces/scatr_its
WORKDIR /workspaces/scatr_its

# Install torch-graph-components
RUN pip install --no-build-isolation ./torch-graph-components

# Install torch-points-kernels and torch-cluster with CUDA support
RUN pip uninstall torch-points-kernels -y
ENV FORCE_CUDA=1
ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6"
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV CUDA_HOME=/usr/local/cuda
RUN FORCE_CUDA=1 pip install --no-deps --no-cache-dir --no-binary :all: torch-points-kernels==0.7.0
RUN pip uninstall torch-cluster
RUN pip install --no-deps --no-cache-dir torch-cluster==1.6.1 -f https://data.pyg.org/whl/torch-1.13.1+cu116.html

# Replace mmengine files
ENV MMENGINE_PATH=/usr/local/lib/python3.8/dist-packages/mmengine
ENV MMDET3D_PATH=/usr/local/lib/python3.8/dist-packages/mmdet3d
COPY ./overrides/loops.py ${MMENGINE_PATH}/runner/
COPY ./overrides/log_processor.py ${MMENGINE_PATH}/runner/
COPY ./overrides/base_model.py ${MMENGINE_PATH}/model/base_model/
COPY ./overrides/transforms_3d.py ${MMDET3D_PATH}/datasets/transforms/

# Add laz support
RUN pip install "laspy[laszip]"

CMD ["bash"]