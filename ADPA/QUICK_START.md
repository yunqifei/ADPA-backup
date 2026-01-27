# 快速开始：训练教师模型（2张GPU）

## 📋 已创建的配置文件

1. ✅ `recipes/accelerate_config/deepspeed_zero3_2gpu.yaml` - 2张GPU的DeepSpeed配置
2. ✅ `recipes/mistral-7b-deita/teacher_sft.yaml` - SFT训练配置
3. ✅ `recipes/mistral-7b-deita/teacher_dpo.yaml` - DPO训练配置

## 🚀 执行步骤

### 步骤1：训练REF Teacher（SFT）

```bash
cd ~/ADPA

CUDA_VISIBLE_DEVICES=0,1 \
ACCELERATE_LOG_LEVEL=info \
DS_SKIP_CUDA_CHECK=1 \
python -m accelerate.commands.launch \
  --config_file recipes/accelerate_config/deepspeed_zero3_2gpu.yaml \
  scripts/run_sft.py \
  recipes/mistral-7b-deita/teacher_sft.yaml
```

**输出**：`~/ADPA/data/mistral-7b-deita/ref_teacher`

### 步骤2：训练DPO Teacher

```bash
cd ~/ADPA

CUDA_VISIBLE_DEVICES=0,1 \
ACCELERATE_LOG_LEVEL=info \
DS_SKIP_CUDA_CHECK=1 \
python -m accelerate.commands.launch \
  --config_file recipes/accelerate_config/deepspeed_zero3_2gpu.yaml \
  scripts/run_distill_dpo.py \
  recipes/mistral-7b-deita/teacher_dpo.yaml
```

**输出**：`~/ADPA/data/mistral-7b-deita/dpo_teacher`

## ⚙️ 关键修改说明

### 1. GPU数量调整
- **原配置**：8张GPU (`CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`)
- **新配置**：2张GPU (`CUDA_VISIBLE_DEVICES=0,1`)
- **Accelerate配置**：`num_processes: 2`

### 2. Batch Size调整
- **原配置**：`per_device_batch_size=1 × 8卡 × gradient_accumulation=16 = 有效batch=128`
- **新配置**：`per_device_batch_size=1 × 2卡 × gradient_accumulation=64 = 有效batch=128`
- **保持相同的有效batch size**，确保训练效果一致

### 3. 路径配置
- **模型路径**：`/home/yunbokun/Models/Teacher/mistralai/Mistral-7B-v0.1`
- **SFT数据集**：`/home/yunbokun/Datasets/sft/deita-10k-v0-sft`
- **DPO数据集**：`/home/yunbokun/Datasets/preference alignment/dpo-mix-7k`
- **输出目录**：`~/ADPA/data/mistral-7b-deita/`

## ⚠️ 注意事项

1. **数据集格式**：
   - SFT数据集需要 `train_sft` 和 `test_sft` split
   - DPO数据集需要 `train` split，包含 `chosen` 和 `rejected` 字段

2. **如果数据集路径不对**：
   - 检查配置文件中的 `dataset_mixer` 部分
   - 可以使用HuggingFace Hub路径（如 `HuggingFaceH4/deita-10k-v0-sft`）

3. **显存不足**：
   - 减小 `max_seq_length` 或 `max_length`
   - 增加 `gradient_accumulation_steps`

4. **训练时间**：
   - SFT：3个epoch，可能需要数小时
   - DPO：1个epoch，相对较快

## ⚠️ 常见错误及解决方案

### 错误1: CUDA兼容性问题（Blackwell GPU）

如果遇到以下错误：
```
NVIDIA RTX PRO 6000 Blackwell Workstation Edition with CUDA capability sm_120 is not compatible
torch.distributed.DistBackendError: NCCL error
```

**原因**：Blackwell架构（sm_120）需要PyTorch 2.5+版本支持。

**解决方案**：
```bash
# 升级PyTorch到支持Blackwell的版本
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 或使用conda
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
```

详细说明请参考：`TROUBLESHOOTING.md`

### 错误2: trust_remote_code警告

此问题已修复。如果仍看到警告，请确保使用最新代码。

## 📝 详细说明

更多详细信息请参考：
- `REPRODUCTION_STEPS.md` - 完整复现步骤
- `TROUBLESHOOTING.md` - 错误排查指南
