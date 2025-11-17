pip install datasets==2.19 # 解决 datasets 版本过高导致的问题
export HF_ENDPOINT=https://hf-mirror.com # 设置 Hugging Face 镜像

pip install -e ".[async]"
pip install -e ".[pi]"
pip install pytest # ==7.4.0 # 解决 pytest 版本过高导致的问题


# 数据采集经验
# 1.单手操作
# 2.夹爪回正
# 3.先对准，再夹取
# 4.下探过程缓慢

# todo:
# 用checkpoint推理
# 1.lerobot 0.4.1迁移
# 2.pi groot调试，训练
# 3.t700机器人数据采集系统搭建                  11.31开工
# 4.vr，视觉遥操系统搭建，关联仿真，产线数据采集，数据增强
# 5.openvla openpi opengalaxea框架
# 6.VLN


# pytorch编译
# 创建并激活 Conda 环境
conda create -n torch_build python=3.11 -y
conda activate torch_build

# 安装编译依赖
conda install -y cmake ninja git gcc_linux-64 gxx_linux-64
pip install -U pip typing_extensions setuptools wheel

# 获取 PyTorch 源码
git clone https://github.com/pytorch/pytorch.git
cd pytorch
git checkout v2.7.1
git submodule sync
git submodule update --init --recursive

# 设置编译参数
export CUDA_HOME=/usr/local/cuda-12.8
export TORCH_CUDA_ARCH_LIST="12.0"
export MAX_JOBS=$(nproc)
export USE_CUDA=1
export USE_CUDNN=1
export USE_NUMA=0
export USE_KINETO=0
export BUILD_TEST=0
export USE_NUMPY=1



# 2) 降低并行度（最关键）
#    这两项都会被 PyTorch/CMake/Ninja 识别，建议先用 2~4 试
export MAX_JOBS=6
export CMAKE_BUILD_PARALLEL_LEVEL=6
# # 3) 关闭非必须组件（减少编译量与内存）
# export USE_NUMA=0        # 免 libnuma
# export USE_KINETO=0      # 免 profiler
# export USE_NNPACK=0      # 免 peachpy/NNPACK
# export USE_QNNPACK=0
# export USE_XNNPACK=0
# export BUILD_TEST=0
# export USE_DISTRIBUTED=0 # 若单卡训练可关（DDP/NCCL不需要）
# # 可选：CPU侧库再缩一点
# export USE_MKLDNN=0      # 关 oneDNN（CPU算子变慢点，但能省内存/编译时间）

# 安装依赖以避免编译错误
conda install -y pyyaml six
sudo apt-get update
sudo apt-get install -y libnuma-dev


# 清理并重新安装 PyTorch
python setup.py clean
rm -rf build
python setup.py bdist_wheel

# 安装编译结果
pip uninstall -y torch
pip install dist/torch-2.7.1*.whl

# 编译 torchvision
cd ..
git clone https://github.com/pytorch/vision.git
cd vision
git checkout v0.22.0

# 安装依赖
conda install -y jpeg libpng libwebp ffmpeg pkg-config
python setup.py clean
rm -rf build
python setup.py bdist_wheel
pip uninstall -y torchvision
pip install dist/torchvision-0.22.0*.whl

# 安装 lerobot
cd ..
pip install lerobot==0.4.1

# 验证 GPU 架构是否支持 sm_120
python - << 'PY'
import torch
print("Torch:", torch.__version__, "| CUDA:", torch.version.cuda)
print("Arch list:", torch.cuda.get_arch_list())
print("Device capability:", torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None)
PY