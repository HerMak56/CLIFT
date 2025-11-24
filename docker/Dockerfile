FROM nvidia/cuda:11.7.1-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG TZ=Etc/UTC
ARG USER=capitan
ARG UID=1000
ARG GID=1000

ENV TZ=${TZ}
RUN ln -snf /usr/share/zoneinfo/${TZ} /etc/localtime && echo "${TZ}" > /etc/timezone

# -------------------------------------------------------------------
# 1. БАЗОВЫЕ ПАКЕТЫ + PYTHON3 + pip
# -------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    sudo \
    locales \
    software-properties-common \
    ca-certificates \
    curl \
    gnupg2 \
    lsb-release \
    git \
    build-essential \
    cmake \
    ninja-build \
    pkg-config \
    python3 python3-dev python3-venv python3-distutils python3-pip \
    libsparsehash-dev \
    libopenblas-dev libblas-dev liblapack-dev \
 && rm -rf /var/lib/apt/lists/*

# -------------------------------------------------------------------
# 2. ЛОКАЛИ
# -------------------------------------------------------------------
RUN locale-gen en_US en_US.UTF-8 && \
    update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

# -------------------------------------------------------------------
# 3. УСТАНОВКА ROS2 HUMBLE (Ubuntu 22.04 = jammy)
# -------------------------------------------------------------------
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
      -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
      > /etc/apt/sources.list.d/ros2.list

RUN apt-get update && apt-get install -y --no-install-recommends \
      ros-humble-desktop \
      python3-colcon-common-extensions \
      ros-dev-tools \
      ros-humble-foxglove-bridge \
      ros-humble-tf-transformations \
    && rm -rf /var/lib/apt/lists/*

ENV ROS_DISTRO=humble
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> /etc/bash.bashrc

# -------------------------------------------------------------------
# 4. СОЗДАНИЕ ПОЛЬЗОВАТЕЛЯ + sudo БЕЗ ПАРОЛЯ
# -------------------------------------------------------------------
RUN groupadd --gid ${GID} ${USER} \
    && useradd --uid ${UID} --gid ${GID} -m -s /bin/bash ${USER} \
    && usermod -aG sudo ${USER} \
    && echo "${USER} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USER} \
    && chmod 0440 /etc/sudoers.d/${USER}

# -------------------------------------------------------------------
# 5. ПОДГОТОВКА VENV
# -------------------------------------------------------------------
RUN mkdir -p /opt/venv && chown -R ${USER}:${USER} /opt/venv

USER ${USER}
WORKDIR /home/${USER}/workspace

ENV VENV=/opt/venv
RUN python3 -m venv $VENV
ENV PATH="$VENV/bin:$PATH"

# -------------------------------------------------------------------
# 6. PYTORCH + ДЕПЕНДЕНСИ (CUDA 11.7, cu117)
# -------------------------------------------------------------------
RUN python -m pip install --no-cache-dir -U pip wheel setuptools

RUN python -m pip install --no-cache-dir \
    "numpy<1.24" \
    "setuptools<60"

RUN python -m pip install --no-cache-dir \
    torch==1.13.1+cu117 \
    torchvision==0.14.1+cu117 \
    torchaudio==0.13.1 \
    pyyaml \
    shapely \
    pyquaternion \
    python-lzf \
    matplotlib \
    tensorboardX \
    scikit-image \
    numba \
    nuscenes-devkit \
    open3d \
    matplotlib \
    ipykernel \
    transforms3d \
    --extra-index-url https://download.pytorch.org/whl/cu117

ENV CUDA_HOME=/usr/local/cuda
ENV FORCE_CUDA=1
ENV TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6"
ENV OMP_NUM_THREADS=32

# -------------------------------------------------------------------
# 7. TORCHSPARSE + MINKOWSKIENGINE
# -------------------------------------------------------------------
RUN python -m pip install --no-cache-dir --no-build-isolation \
  "git+https://github.com/mit-han-lab/torchsparse.git@v2.1.0"

RUN git clone https://github.com/NVIDIA/MinkowskiEngine.git /home/${USER}/MinkowskiEngine \
 && cd /home/${USER}/MinkowskiEngine \
 && python setup.py install --blas=openblas --force_cuda \
 && rm -rf /home/${USER}/.cache/pip
# -------------------------------------------------------------------
# 8. ENTRYPOINT
# -------------------------------------------------------------------

RUN mkdir -p ~/workspace/ros2_ws/src

CMD ["/bin/bash"]


# FROM nvidia/cuda:11.6.2-devel-ubuntu20.04

# ARG DEBIAN_FRONTEND=noninteractive
# ARG TZ=Etc/UTC
# ARG USER=oversir
# ARG UID=1000
# ARG GID=1000

# ENV TZ=${TZ}
# RUN ln -snf /usr/share/zoneinfo/${TZ} /etc/localtime && echo "${TZ}" > /etc/timezone

# # 1. Base packages + Python 3.9 + sparsehash
# # RUN apt-get update && apt-get install -y --no-install-recommends \
# #     software-properties-common \
# #     ca-certificates curl git \
# #     build-essential cmake ninja-build pkg-config \
# #     python3.9 python3.9-dev python3.9-distutils python3.9-venv \
# #     libsparsehash-dev \
# #     libopenblas-dev libblas-dev liblapack-dev \
# #     python3-distutils \ 
# #  && rm -rf /var/lib/apt/lists/*

# RUN apt-get update && apt-get install -y --no-install-recommends \
#     sudo \
#     locales \
#     software-properties-common \
#     ca-certificates \
#     curl \
#     gnupg2 \
#     lsb-release \
#     git \
#     build-essential \
#     cmake \
#     ninja-build \
#     pkg-config \
#     python3 python3-dev python3-venv python3-distutils \
#     python3-pip \
#     && rm -rf /var/lib/apt/lists/*

# # 2. Create user
# RUN groupadd --gid $USER_GID $USERNAME \
#     && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
#     && echo "$USERNAME ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/$USERNAME \
#     && chmod 0440 /etc/sudoers.d/$USERNAME

# # 3. Prepare venv
# RUN mkdir -p /opt/venv && chown -R ${USER}:${USER} /opt/venv

# # 4. Switch to user
# USER ${USER}
# WORKDIR /home/${USER}/workspace

# # 5. Create venv
# ENV VENV=/opt/venv
# RUN python3.9 -m venv $VENV
# ENV PATH="$VENV/bin:$PATH"

# # 6. pip and dependencis
# RUN python -m pip install --no-cache-dir -U pip wheel
# RUN pip install "numpy<1.24" "setuptools<60"

# RUN python -m pip install --no-cache-dir \
#     torch==1.13.1+cu116 \
#     torchvision==0.14.1+cu116 \
#     torchaudio==0.13.1 \
#     pyyaml \
#     shapely \
#     pyquaternion \  
#     python-lzf \
#     matplotlib \ 
#     tensorboardX \
#     scikit-image \
#     numba \
#     nuscenes-devkit \
#     --extra-index-url https://download.pytorch.org/whl/cu116

# ENV CUDA_HOME=/usr/local/cuda
# ENV TORCH_CUDA_ARCH_LIST="8.0;8.6"

# RUN python -m pip install --no-cache-dir --no-build-isolation \
#   "git+https://github.com/mit-han-lab/torchsparse.git@v2.1.0"

# RUN python -m pip install -U "git+https://github.com/NVIDIA/MinkowskiEngine" --no-deps --no-build-isolation


# # -------------------------------------------------------------------
# # 2. УСТАНОВКА ROS2 HUMBLE
# # -------------------------------------------------------------------

# RUN locale-gen en_US en_US.UTF-8 && \
#     update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
# ENV LANG=en_US.UTF-8
# ENV LC_ALL=en_US.UTF-8

# RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | apt-key add - && \
#     echo "deb [arch=amd64] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" \
#       > /etc/apt/sources.list.d/ros2-latest.list && \
#     apt-get update && apt-get install -y --no-install-recommends \
#       ros-humble-desktop \
#       python3-colcon-common-extensions \
#       ros-dev-tools \
#     && rm -rf /var/lib/apt/lists/*

# ENV ROS_DISTRO=humble
# RUN echo "source /opt/ros/humble/setup.bash" >> /etc/bash.bashrc

# CMD ["/bin/bash"]

# # # Docker/Dockerfile
# # FROM nvidia/cuda:11.6.2-devel-ubuntu20.04

# # ARG DEBIAN_FRONTEND=noninteractive
# # ARG TZ=Etc/UTC
# # ARG USER=oversir
# # ARG UID=1000
# # ARG GID=1000

# # ENV TZ=${TZ}
# # RUN ln -snf /usr/share/zoneinfo/${TZ} /etc/localtime && echo "${TZ}" > /etc/timezone

# # # Билд-зависимости + Python 3.9
# # RUN apt-get update && apt-get install -y --no-install-recommends \
# #     software-properties-common ca-certificates curl git \
# #     build-essential cmake ninja-build pkg-config \
# #     python3.9 python3.9-dev python3.9-distutils python3-pip \
# #  && ln -sf /usr/bin/python3.9 /usr/bin/python \
# #  && rm -rf /var/lib/apt/lists/*

# # # pip инструменты
# # RUN python -m pip install --no-cache-dir -U pip setuptools wheel

# # # 1) Ставим PyTorch 1.13.1 с CUDA 11.6 (ОБЯЗАТЕЛЬНО до torchsparse)
# # RUN pip install --no-cache-dir \
# #     torch==1.13.1+cu116 \
# #     torchvision==0.14.1+cu116 \
# #     torchaudio==0.13.1 \
# #     --extra-index-url https://download.pytorch.org/whl/cu116

# # # 2A) РЕКОМЕНДОВАНО: предсобранный wheel torchsparse v2.x, подхватит версию по установленным torch/cuda
# # # Если сервер PyPI недоступен, см. вариант 2B ниже.
# # RUN python -c "$(curl -fsSL https://raw.githubusercontent.com/mit-han-lab/torchsparse/master/install.py)"

# # # --- ВАРИАНТ 2B (из исходников) — оставьте закомментированным, если 2A проходит ---
# # # ENV CUDA_HOME=/usr/local/cuda
# # # # при желании зафиксируйте архитектуры, например Ampere:
# # # ENV TORCH_CUDA_ARCH_LIST="8.0;8.6"
# # # RUN pip install --no-cache-dir "git+https://github.com/mit-han-lab/torchsparse.git"

# # # Пользователь
# # RUN groupadd -g ${GID} ${USER} \
# #  && useradd -m -u ${UID} -g ${GID} -s /bin/bash ${USER}
# # WORKDIR /home/${USER}/workspace
# # RUN chown -R ${USER}:${USER} /home/${USER}
# # USER ${USER}

# # CMD ["/bin/bash"]



# # FROM nvidia/cuda:11.6.2-devel-ubuntu20.04

# # ARG DEBIAN_FRONTEND=noninteractive
# # ARG TZ=Etc/UTC
# # ARG USER=oversir
# # ARG UID=1000
# # ARG GID=1000

# # ENV TZ=${TZ}
# # RUN ln -snf /usr/share/zoneinfo/${TZ} /etc/localtime && echo "${TZ}" > /etc/timezone

# # # ----------------------------------------------------------------------
# # # 1. Базовые пакеты + Python3.9
# # # ----------------------------------------------------------------------
# # RUN apt-get update && apt-get install -y --no-install-recommends \
# #     software-properties-common \
# #     ca-certificates curl git \
# #     build-essential cmake ninja-build pkg-config \
# #     python3.9 python3.9-dev python3.9-distutils python3.9-venv \
# #  && rm -rf /var/lib/apt/lists/*

# # # ----------------------------------------------------------------------
# # # 2. Создаём изолированное venv для ВСЕГО ML-стека
# # # ----------------------------------------------------------------------
# # ENV VENV=/opt/venv
# # RUN python3.9 -m venv $VENV

# # # чтобы работать через "python" и "pip"
# # ENV PATH="$VENV/bin:$PATH"

# # # ----------------------------------------------------------------------
# # # 3. Устанавливаем pip/setuptools/wheel внутрь venv
# # # ----------------------------------------------------------------------
# # RUN pip install --no-cache-dir -U pip setuptools wheel

# # # ----------------------------------------------------------------------
# # # 4. Ставим PyTorch 1.13.1 + cu116 в venv
# # # ----------------------------------------------------------------------
# # RUN pip install --no-cache-dir \
# #     torch==1.13.1+cu116 \
# #     torchvision==0.14.1+cu116 \
# #     torchaudio==0.13.1 \
# #     --extra-index-url https://download.pytorch.org/whl/cu116

# # # ----------------------------------------------------------------------
# # # 5. TorchSparse (v1.2.0 или git)
# # # ----------------------------------------------------------------------
# # ENV CUDA_HOME=/usr/local/cuda
# # ENV TORCH_CUDA_ARCH_LIST="8.0;8.6"

# # RUN python -c "$(curl -fsSL https://raw.githubusercontent.com/mit-han-lab/torchsparse/master/install.py)"
# # # или
# # # RUN pip install --no-cache-dir "git+https://github.com/mit-han-lab/torchsparse.git"

# # # ----------------------------------------------------------------------
# # # 6. Создаём пользователя
# # # ----------------------------------------------------------------------
# # RUN groupadd -g ${GID} ${USER} \
# #  && useradd -m -u ${UID} -g ${GID} -s /bin/bash ${USER}

# # WORKDIR /home/${USER}/workspace
# # RUN chown -R ${USER}:${USER} /home/${USER}
# # USER ${USER}

# # CMD ["/bin/bash"]


# # FROM nvidia/cuda:11.6.2-devel-ubuntu20.04

# # ARG DEBIAN_FRONTEND=noninteractive
# # ARG TZ=Etc/UTC
# # ARG USER=oversir
# # ARG UID=1000
# # ARG GID=1000

# # ENV TZ=${TZ}
# # RUN ln -snf /usr/share/zoneinfo/${TZ} /etc/localtime && echo "${TZ}" > /etc/timezone

# # # 1. Базовые пакеты + Python 3.9 + sparsehash
# # RUN apt-get update && apt-get install -y --no-install-recommends \
# #     software-properties-common \
# #     ca-certificates curl git \
# #     build-essential cmake ninja-build pkg-config \
# #     python3.9 python3.9-dev python3.9-distutils python3.9-venv \
# #     libsparsehash-dev \
# #  && rm -rf /var/lib/apt/lists/*

# # # 2. venv
# # ENV VENV=/opt/venv
# # RUN python3.9 -m venv $VENV
# # ENV PATH="$VENV/bin:$PATH"

# # # 3. pip в venv
# # RUN python -m pip install --no-cache-dir -U pip setuptools wheel

# # # 4. PyTorch 1.13.1 + cu116
# # RUN python -m pip install --no-cache-dir \
# #     torch==1.13.1+cu116 \
# #     torchvision==0.14.1+cu116 \
# #     torchaudio==0.13.1 \
# #     pyyaml \
# #     --extra-index-url https://download.pytorch.org/whl/cu116

# # # 5. TorchSparse (из git, без изоляции)
# # ENV CUDA_HOME=/usr/local/cuda
# # ENV TORCH_CUDA_ARCH_LIST="8.0;8.6"

# # RUN python -m pip install --no-cache-dir --no-build-isolation \
# #     "git+https://github.com/mit-han-lab/torchsparse.git"

# # # 6. Пользователь
# # RUN groupadd -g ${GID} ${USER} \
# #  && useradd -m -u ${UID} -g ${GID} -s /bin/bash ${USER}

# # WORKDIR /home/${USER}/workspace
# # RUN chown -R ${USER}:${USER} /opt/venv
# # RUN chown -R ${USER}:${USER} /home/${USER}
# # USER ${USER}

# # CMD ["/bin/bash"]