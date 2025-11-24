FROM nvidia/cuda:11.6.2-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ARG TZ=Etc/UTC
ARG USER=oversirа 
ARG UID=1000
ARG GID=1000

ENV TZ=${TZ}
RUN ln -snf /usr/share/zoneinfo/${TZ} /etc/localtime && echo "${TZ}" > /etc/timezone

# 1. Base packages + Python 3.9 + sparsehash
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    ca-certificates curl git \
    build-essential cmake ninja-build pkg-config \
    python3.9 python3.9-dev python3.9-distutils python3.9-venv \
    libsparsehash-dev \
    libopenblas-dev libblas-dev liblapack-dev \
    python3-distutils \ 
 && rm -rf /var/lib/apt/lists/*

# 2. Create user
RUN groupadd -g ${GID} ${USER} \
 && useradd -m -u ${UID} -g ${GID} -s /bin/bash ${USER}

# 3. Prepare venv
RUN mkdir -p /opt/venv && chown -R ${USER}:${USER} /opt/venv

# 4. Switch to user
USER ${USER}
WORKDIR /home/${USER}/workspace

# 5. Create venv
ENV VENV=/opt/venv
RUN python3.9 -m venv $VENV
ENV PATH="$VENV/bin:$PATH"

# 6. pip and dependencis
RUN python -m pip install --no-cache-dir -U pip wheel
RUN pip install "numpy<1.24" "setuptools<60"

RUN python -m pip install --no-cache-dir \
    torch==1.13.1+cu116 \
    torchvision==0.14.1+cu116 \
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
    --extra-index-url https://download.pytorch.org/whl/cu116

ENV CUDA_HOME=/usr/local/cuda
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6"

RUN python -m pip install --no-cache-dir --no-build-isolation \
  "git+https://github.com/mit-han-lab/torchsparse.git@v2.1.0"

RUN python -m pip install -U "git+https://github.com/NVIDIA/MinkowskiEngine" --no-deps --no-build-isolation
CMD ["/bin/bash"]

# # Docker/Dockerfile
# FROM nvidia/cuda:11.6.2-devel-ubuntu20.04

# ARG DEBIAN_FRONTEND=noninteractive
# ARG TZ=Etc/UTC
# ARG USER=oversir
# ARG UID=1000
# ARG GID=1000

# ENV TZ=${TZ}
# RUN ln -snf /usr/share/zoneinfo/${TZ} /etc/localtime && echo "${TZ}" > /etc/timezone

# # Билд-зависимости + Python 3.9
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     software-properties-common ca-certificates curl git \
#     build-essential cmake ninja-build pkg-config \
#     python3.9 python3.9-dev python3.9-distutils python3-pip \
#  && ln -sf /usr/bin/python3.9 /usr/bin/python \
#  && rm -rf /var/lib/apt/lists/*

# # pip инструменты
# RUN python -m pip install --no-cache-dir -U pip setuptools wheel

# # 1) Ставим PyTorch 1.13.1 с CUDA 11.6 (ОБЯЗАТЕЛЬНО до torchsparse)
# RUN pip install --no-cache-dir \
#     torch==1.13.1+cu116 \
#     torchvision==0.14.1+cu116 \
#     torchaudio==0.13.1 \
#     --extra-index-url https://download.pytorch.org/whl/cu116

# # 2A) РЕКОМЕНДОВАНО: предсобранный wheel torchsparse v2.x, подхватит версию по установленным torch/cuda
# # Если сервер PyPI недоступен, см. вариант 2B ниже.
# RUN python -c "$(curl -fsSL https://raw.githubusercontent.com/mit-han-lab/torchsparse/master/install.py)"

# # --- ВАРИАНТ 2B (из исходников) — оставьте закомментированным, если 2A проходит ---
# # ENV CUDA_HOME=/usr/local/cuda
# # # при желании зафиксируйте архитектуры, например Ampere:
# # ENV TORCH_CUDA_ARCH_LIST="8.0;8.6"
# # RUN pip install --no-cache-dir "git+https://github.com/mit-han-lab/torchsparse.git"

# # Пользователь
# RUN groupadd -g ${GID} ${USER} \
#  && useradd -m -u ${UID} -g ${GID} -s /bin/bash ${USER}
# WORKDIR /home/${USER}/workspace
# RUN chown -R ${USER}:${USER} /home/${USER}
# USER ${USER}

# CMD ["/bin/bash"]



# FROM nvidia/cuda:11.6.2-devel-ubuntu20.04

# ARG DEBIAN_FRONTEND=noninteractive
# ARG TZ=Etc/UTC
# ARG USER=oversir
# ARG UID=1000
# ARG GID=1000

# ENV TZ=${TZ}
# RUN ln -snf /usr/share/zoneinfo/${TZ} /etc/localtime && echo "${TZ}" > /etc/timezone

# # ----------------------------------------------------------------------
# # 1. Базовые пакеты + Python3.9
# # ----------------------------------------------------------------------
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     software-properties-common \
#     ca-certificates curl git \
#     build-essential cmake ninja-build pkg-config \
#     python3.9 python3.9-dev python3.9-distutils python3.9-venv \
#  && rm -rf /var/lib/apt/lists/*

# # ----------------------------------------------------------------------
# # 2. Создаём изолированное venv для ВСЕГО ML-стека
# # ----------------------------------------------------------------------
# ENV VENV=/opt/venv
# RUN python3.9 -m venv $VENV

# # чтобы работать через "python" и "pip"
# ENV PATH="$VENV/bin:$PATH"

# # ----------------------------------------------------------------------
# # 3. Устанавливаем pip/setuptools/wheel внутрь venv
# # ----------------------------------------------------------------------
# RUN pip install --no-cache-dir -U pip setuptools wheel

# # ----------------------------------------------------------------------
# # 4. Ставим PyTorch 1.13.1 + cu116 в venv
# # ----------------------------------------------------------------------
# RUN pip install --no-cache-dir \
#     torch==1.13.1+cu116 \
#     torchvision==0.14.1+cu116 \
#     torchaudio==0.13.1 \
#     --extra-index-url https://download.pytorch.org/whl/cu116

# # ----------------------------------------------------------------------
# # 5. TorchSparse (v1.2.0 или git)
# # ----------------------------------------------------------------------
# ENV CUDA_HOME=/usr/local/cuda
# ENV TORCH_CUDA_ARCH_LIST="8.0;8.6"

# RUN python -c "$(curl -fsSL https://raw.githubusercontent.com/mit-han-lab/torchsparse/master/install.py)"
# # или
# # RUN pip install --no-cache-dir "git+https://github.com/mit-han-lab/torchsparse.git"

# # ----------------------------------------------------------------------
# # 6. Создаём пользователя
# # ----------------------------------------------------------------------
# RUN groupadd -g ${GID} ${USER} \
#  && useradd -m -u ${UID} -g ${GID} -s /bin/bash ${USER}

# WORKDIR /home/${USER}/workspace
# RUN chown -R ${USER}:${USER} /home/${USER}
# USER ${USER}

# CMD ["/bin/bash"]


# FROM nvidia/cuda:11.6.2-devel-ubuntu20.04

# ARG DEBIAN_FRONTEND=noninteractive
# ARG TZ=Etc/UTC
# ARG USER=oversir
# ARG UID=1000
# ARG GID=1000

# ENV TZ=${TZ}
# RUN ln -snf /usr/share/zoneinfo/${TZ} /etc/localtime && echo "${TZ}" > /etc/timezone

# # 1. Базовые пакеты + Python 3.9 + sparsehash
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     software-properties-common \
#     ca-certificates curl git \
#     build-essential cmake ninja-build pkg-config \
#     python3.9 python3.9-dev python3.9-distutils python3.9-venv \
#     libsparsehash-dev \
#  && rm -rf /var/lib/apt/lists/*

# # 2. venv
# ENV VENV=/opt/venv
# RUN python3.9 -m venv $VENV
# ENV PATH="$VENV/bin:$PATH"

# # 3. pip в venv
# RUN python -m pip install --no-cache-dir -U pip setuptools wheel

# # 4. PyTorch 1.13.1 + cu116
# RUN python -m pip install --no-cache-dir \
#     torch==1.13.1+cu116 \
#     torchvision==0.14.1+cu116 \
#     torchaudio==0.13.1 \
#     pyyaml \
#     --extra-index-url https://download.pytorch.org/whl/cu116

# # 5. TorchSparse (из git, без изоляции)
# ENV CUDA_HOME=/usr/local/cuda
# ENV TORCH_CUDA_ARCH_LIST="8.0;8.6"

# RUN python -m pip install --no-cache-dir --no-build-isolation \
#     "git+https://github.com/mit-han-lab/torchsparse.git"

# # 6. Пользователь
# RUN groupadd -g ${GID} ${USER} \
#  && useradd -m -u ${UID} -g ${GID} -s /bin/bash ${USER}

# WORKDIR /home/${USER}/workspace
# RUN chown -R ${USER}:${USER} /opt/venv
# RUN chown -R ${USER}:${USER} /home/${USER}
# USER ${USER}

# CMD ["/bin/bash"]