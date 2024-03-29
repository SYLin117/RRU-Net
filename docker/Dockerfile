ARG CUDA="10.2"
ARG CUDNN="7"

FROM nvidia/cuda:${CUDA}-cudnn${CUDNN}-devel-ubuntu16.04

RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

# install basics
RUN apt-get update -y \
    && apt-get install -y apt-utils wget git curl ca-certificates bzip2 cmake tree htop bmon iotop gcc g++ \
    && apt-get install -y ffmpeg libglib2.0-0 libsm6 libxext6 libxrender-dev

# Install Miniconda
#RUN curl -so /miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
# && chmod +x /miniconda.sh \
# && /miniconda.sh -b -p /miniconda \
# && rm /miniconda.sh

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b -p /miniconda\
    && rm -f Miniconda3-latest-Linux-x86_64.sh

ENV PATH=/miniconda/bin:$PATH

# Create a Python 3.8 environment
RUN /miniconda/bin/conda install -y conda-build \
 && /miniconda/bin/conda create -y --name py38 python=3.8 \
 && /miniconda/bin/conda clean -ya

ENV CONDA_DEFAULT_ENV=py38
ENV CONDA_PREFIX=/miniconda/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH
ENV CONDA_AUTO_UPDATE_CONDA=false

RUN conda install -y ipython
RUN pip install requests ninja yacs cython matplotlib opencv-python tqdm pycocotools \
    && pip install self-attention-cv scikit-learn scikit-image imgaug torchinfo timm imutils \
    && pip install XlsxWriter efficientnet_pytorch

# Install PyTorch 1.0 Nightly
ARG CUDA
#RUN conda install pytorch-nightly cudatoolkit=${CUDA} -c pytorch \
##RUN conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=${CUDA} -c pytorch \
#RUN conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=${CUDA} -c pytorch \
RUN conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=${CUDA} -c pytorch \
  && conda clean -ya

# Install TorchVision master
#RUN git clone https://github.com/pytorch/vision.git \
# && cd vision \
# && python setup.py install

# install pycocotools
#RUN git clone https://github.com/cocodataset/cocoapi.git \
# && cd cocoapi/PythonAPI \
# && python setup.py build_ext install

# install apex
#ENV TORCH_CUDA_ARCH_LIST="7.0"
#RUN git clone https://github.com/NVIDIA/apex.git \
# && cd apex \
# && python setup.py install --cuda_ext --cpp_ext

#RUN #cd / \
# && git clone https://github.com/NVIDIA/apex.git apex \
# && cd apex \
# && git reset --hard 3fe10b5597ba14a748ebb271a6ab97c09c5701ac \
# && pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

## install PyTorch Detection
#ARG FORCE_CUDA="1"
#ENV FORCE_CUDA=${FORCE_CUDA}
#RUN git clone https://github.com/facebookresearch/maskrcnn-benchmark.git \
# && cd maskrcnn-benchmark \
# && python setup.py build develop

RUN mkdir /WD

WORKDIR /WD
