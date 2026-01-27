# 错误排查指南

## 错误1: CUDA兼容性问题（主要问题）

### 错误信息
```
NVIDIA RTX PRO 6000 Blackwell Workstation Edition with CUDA capability sm_120 is not compatible with the current PyTorch installation.
The current PyTorch install supports CUDA capabilities sm_50 sm_60 sm_70 sm_75 sm_80 sm_86 sm_90.
```

### 根本原因
您的GPU是**Blackwell架构**（计算能力sm_120），这是NVIDIA最新的GPU架构。但是您当前安装的PyTorch版本只支持到sm_90（H100等），不支持Blackwell架构。

当DeepSpeed ZeRO-3尝试在GPU之间进行通信时，由于PyTorch无法正确识别GPU，导致CUDA调用失败，进而引发NCCL错误：
```
torch.distributed.DistBackendError: NCCL error
ncclUnhandledCudaError: Call to CUDA function failed.
Cuda failure 'invalid argument'
```

### 解决方案

#### 方案1：升级PyTorch（推荐）

Blackwell架构需要PyTorch 2.5+版本。请检查并升级：

```bash
# 检查当前PyTorch版本
python -c "import torch; print(torch.__version__)"

# 升级到支持Blackwell的PyTorch版本
# 注意：需要CUDA 12.4+支持Blackwell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 或者使用conda
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
```

**验证安装**：
```bash
python -c "import torch; print(torch.cuda.get_device_capability(0))"
# 应该显示 (12, 0) 或更高
```

#### 方案2：使用CPU Offload（临时方案）

如果无法升级PyTorch，可以修改DeepSpeed配置，将优化器和参数offload到CPU：

修改 `recipes/accelerate_config/deepspeed_zero3_2gpu.yaml`：
```yaml
deepspeed_config:
  offload_optimizer_device: cpu  # 已经是cpu
  offload_param_device: cpu      # 已经是cpu
```

但即使使用CPU offload，模型初始化时仍需要在GPU上创建张量，所以可能仍会失败。

#### 方案3：使用单GPU训练（临时方案）

如果多GPU训练失败，可以尝试单GPU训练：

1. 创建单GPU配置 `recipes/accelerate_config/deepspeed_zero3_1gpu.yaml`：
```yaml
compute_environment: LOCAL_MACHINE
debug: false
deepspeed_config:
  deepspeed_multinode_launcher: standard
  offload_optimizer_device: cpu
  offload_param_device: cpu
  zero3_init_flag: true
  zero3_save_16bit_model: true
  zero_stage: 3
distributed_type: DEEPSPEED
downcast_bf16: 'no'
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 1
rdzv_backend: static
same_network: true
use_cpu: false
```

2. 修改训练命令：
```bash
CUDA_VISIBLE_DEVICES=0 \
ACCELERATE_LOG_LEVEL=info \
DS_SKIP_CUDA_CHECK=1 \
python -m accelerate.commands.launch \
  --config_file recipes/accelerate_config/deepspeed_zero3_1gpu.yaml \
  scripts/run_sft.py \
  recipes/mistral-7b-deita/teacher_sft.yaml
```

**注意**：单GPU训练需要更大的gradient_accumulation_steps，可能需要调整batch size。

---

## 错误2: trust_remote_code警告（已修复）

### 错误信息
```
`trust_remote_code` is not supported anymore.
Please check that the Hugging Face dataset isn't based on a loading script and remove `trust_remote_code`.
```

### 原因
新版本的`datasets`库（>=2.14.0）移除了`trust_remote_code`参数，因为安全考虑。

### 解决方案
已修复：移除了`alignment/data.py`中的`trust_remote_code=True`参数。

---

## 推荐的解决步骤

1. **首先尝试升级PyTorch**（方案1）：
   ```bash
   # 检查CUDA版本
   nvcc --version
   
   # 升级PyTorch到支持Blackwell的版本
   pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
   ```

2. **如果升级后仍有问题，检查NCCL**：
   ```bash
   # 设置NCCL调试
   export NCCL_DEBUG=INFO
   export NCCL_DEBUG_SUBSYS=ALL
   
   # 重新运行训练，查看详细错误信息
   ```

3. **如果多GPU仍有问题，尝试单GPU训练**（方案3）

4. **验证修复**：
   ```bash
   python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count()); [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"
   ```

---

## 其他注意事项

1. **Blackwell架构要求**：
   - CUDA 12.4或更高版本
   - PyTorch 2.5+（或支持sm_120的版本）
   - 相应的cuDNN和NCCL版本

2. **如果使用conda环境**：
   ```bash
   conda activate handbook
   conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
   ```

3. **检查驱动版本**：
   ```bash
   nvidia-smi
   # 确保驱动版本支持Blackwell架构
   ```
