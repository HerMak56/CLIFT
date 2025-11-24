# CLIFT Startup Guide

This document describes the intermediate steps for preparing and launching the **CLIFT** project. The project uses the DCCLA pedestrian detection model, ROS 2, and a CUDA-based GPU container. The description uses the repository structure available on GitHub, so if you see differences or missing files, follow the actual content of your copy.

## 1. Cloning the repository and submodule

1. Clone the repository and download the DCCLA submodule. This can be done with a single command:

```bash
  git clone --recursive https://github.com/HerMak56/CLIFT.git
  cd CLIFT
  ```

   If the repository has already been cloned without the `--recursive` option, initialize and update the submodule manually:

```bash
  git submodule update --init --recursive
  ```


## 2. Building and installing DCCLA

The **DCCLA** submodule contains the model source code and libraries. Its README lists the required versions of Python and libraries. At the time of the 2025 commit, the requirements include Python 3.9, PyTorch 1.13.1, and CUDA 11.6 support. The installation procedure is as follows (perform these steps **inside** the container after it has been built, see below):

1. Go to the submodule directory:

```bash
   cd DCCLA
   ```

2. Set the package to development mode and build the auxiliary modules:

   ```bash
   python setup.py develop  # install DCCLA itself
   cd lib/iou3d
   python setup.py develop  # build the iou3d module
   cd ../jrdb_det3d_eval
   python setup.py develop  # build the JRDB evaluation module
   cd ../../..
   ```

3. Download the JRDB dataset to the `DCCLA/data` directory as described in the README submodule (the original suggests downloading it to `PROJECT/data`). When running in a container, the path to the dataset can be mounted externally (see the section on Docker).

## 3. Preparing the container

The project comes with a Dockerfile and a `compose.yaml` file in the `docker` directory. The base image uses NVIDIA CUDA 11.7.1 and installs ROS 2 Humble, colcon, and the necessary libraries. To run:

1. **Install** Docker and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/installation.html) to support GPU operation.
2. Go to the `docker` directory:

```bash
   cd docker
   ```

3. Build and run the container using Docker Compose. The `compose.yaml` file specifies the user's UID and GID, the image name, and the GPU parameters. The container mounts the project root directory and the JRDB dataset directory inside the container. Run:

```bash
  docker compose build
  docker compose up -d
  ```

   By default, the `..` directory (project root) is mounted in `/home/capitan/workspace` and the JRDB dataset from `/mnt/vol0/datasets/jrdb` is mounted in `/home/capitan/workspace/DCCLA/data/JRDB2019`. Change these paths in `compose.yaml` if your dataset is located elsewhere.


4. After starting the container, connect to it:

```bash
   docker exec -it dev-cuda116 /bin/bash
   ```

Here, `dev-cuda116` is the name of the container from `compose.yaml`.

## 4. Building the ROS 2 workspace

Inside the container, the project contains the ROS 2 workspace `ros2_ws`. Before building, make sure that the ROS 2 dependencies are installed and the virtual environment is created. The Dockerfile provides for the installation of ROS 2 Humble and the creation of the virtual environment `/opt/venv`. To build:

1. Load the ROS 2 environment and activate the virtual environment:

   ```bash
   source /opt/ros/humble/setup.bash
   source /opt/venv/bin/activate
   ```

2. Go to the workspace directory and build the packages:

```bash
   cd ~/workspace/ros2_ws
   colcon build --symlink-install
   ```

   The `--symlink-install` option is useful during development: instead of copying the source files to the `install` directory, symbolic links are created.

3. After building, activate the generated settings:

```bash
   source install/local_setup.bash
   ```

## 5. Changing the interpreter in the executable file

The `detector_node.py` file in the `dccla_detector` package has a hard-coded shebang string `#!/opt/venv/bin/python3`. If you are building the project outside of a container or using your own virtual environment, you should make sure that the path in the first line of the executable script matches your Python interpreter. After building the ROS 2 package, the executable script is usually located at:

```
ros2_ws/install/dccla_detector/lib/dccla_detector/detector_node
```

Open this file and, if necessary, replace the first line with the correct path, for example:

```bash
#!/usr/bin/env python3  # or the path to your venv, e.g. /home/USER/miniconda3/envs/ros/bin/python
```

## 6. Starting the DCCLA detector node

After completing all of the above steps, you can start the detection node. First, make sure that the DCCLA model is installed (see section 2) and that the control weight file is located at the path specified in the node parameters (`ckp/DCCLA_JRDB2019.pth` — the default path in the `detector_node.py` file ). To start, use the standard ROS 2 command:

```bash
ros2 run dccla_detector detector_node
```

If you want to change the parameters (input point cloud theme, path to the configuration file, or control weights), run the node with arguments, for example:

```bash
ros2 run dccla_detector detector_node --ros-args \
  -p config_file:=/home/capitan/workspace/DCCLA/bin/jrdb19.yaml \
  -p checkpoint_file:=/home/capitan/workspace/DCCLA/ckp/DCCLA_JRDB2019.pth \
  -p input_topic:=/ouster/points
```


## Notes

- The container installs the PyTorch, TorchSparse, and MinkowskiEngine libraries, so no additional dependencies are required for DCCLA.
- If you are running the project on your machine without Docker, make sure that the versions of Python, PyTorch, and CUDA meet the DCCLA requirements.
- The `compose.yaml` file sets a limit on the total memory (`shm_size`) and enables access to all GPUs on the host. Change these parameters if necessary.

This README is an intermediate version and describes the general structure of the project launch. Additional details on model training, dataset preparation, or data visualization can be found in the DCCLA submodule README and in the project source code.

