# ADPA 实验复现步骤指南

## 环境准备

### 1. 已准备的资源
- ✅ 初始教师模型：`/home/yunbokun/Models/Teacher/mistralai/Mistral-7B-v0.1`
- ✅ SFT数据集：`/home/yunbokun/Datasets/sft/deita-10k-v0-sft`
- ✅ 偏好对齐数据集：`/home/yunbokun/Datasets/preference alignment/dpo-mix-7k`
- ✅ ADPA代码：`~/ADPA`
- ✅ 硬件：2张GPU

### 2. 配置文件说明

我已经为你创建了以下配置文件：

1. **Accelerate配置**（2张GPU）：
   - `recipes/accelerate_config/deepspeed_zero3_2gpu.yaml`
   - 将 `num_processes` 设置为 2

2. **SFT训练配置**：
   - `recipes/mistral-7b-deita/teacher_sft.yaml`
   - 已配置模型路径、数据集路径、输出目录

3. **DPO训练配置**：
   - `recipes/mistral-7b-deita/teacher_dpo.yaml`
   - 已配置模型路径、数据集路径、输出目录

## 第一步：训练REF Teacher（SFT阶段）

### 目的
对Mistral-7B进行监督微调，使其能够遵循指令，这是后续DPO训练的基础。

### 执行命令

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

### 关键参数说明

1. **CUDA_VISIBLE_DEVICES=0,1**：
   - 指定使用2张GPU（GPU 0和GPU 1）
   - 根据你的实际GPU编号调整

2. **--config_file recipes/accelerate_config/deepspeed_zero3_2gpu.yaml**：
   - 使用2张GPU的DeepSpeed ZeRO-3配置
   - ZeRO-3会将模型参数、优化器状态和梯度分片到2张GPU上

3. **gradient_accumulation_steps: 64**：
   - 原配置（8卡）：`per_device_batch_size=1 × 8卡 × gradient_accumulation=16 = 有效batch=128`
   - 新配置（2卡）：`per_device_batch_size=1 × 2卡 × gradient_accumulation=64 = 有效batch=128`
   - 保持与原论文相同的有效batch size

4. **输出目录**：
   - 训练完成后，REF teacher模型会保存在：`~/ADPA/data/mistral-7b-deita/ref_teacher`

### 预期结果
- 训练3个epoch
- 模型保存在 `~/ADPA/data/mistral-7b-deita/ref_teacher`
- 训练日志会显示loss下降情况

## 第二步：训练DPO Teacher

### 目的
在REF teacher基础上，使用DPO方法进行偏好对齐训练，得到能够更好捕捉人类偏好的DPO teacher。

### 执行命令

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

### 关键参数说明

1. **model_name_or_path: ~/ADPA/data/mistral-7b-deita/ref_teacher**：
   - 指向第一步训练好的REF teacher
   - 确保路径正确

2. **dataset_mixer**：
   - 使用偏好对齐数据集（chosen/rejected pairs）
   - 路径：`/home/yunbokun/Datasets/preference alignment/dpo-mix-7k`

3. **dpo_weight: 1**：
   - 启用DPO损失
   - 其他蒸馏损失权重为0（这是纯DPO训练）

4. **beta: 0.01**：
   - DPO的温度参数
   - 控制与参考模型的KL散度约束强度

5. **输出目录**：
   - DPO teacher模型会保存在：`~/ADPA/data/mistral-7b-deita/dpo_teacher`

### 预期结果
- 训练1个epoch
- 模型保存在 `~/ADPA/data/mistral-7b-deita/dpo_teacher`
- 这是后续DCKD和ADPA训练所需的教师模型

## 注意事项

### 1. 数据集格式检查

确保你的数据集格式正确：

**SFT数据集**应包含 `messages` 字段，格式如下：
```python
{
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

**偏好对齐数据集**应包含 `chosen` 和 `rejected` 字段：
```python
{
  "chosen": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "rejected": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

### 2. 内存和显存

- Mistral-7B在2张GPU上使用DeepSpeed ZeRO-3应该可以运行
- 如果遇到OOM，可以：
  - 减小 `max_seq_length` 或 `max_length`
  - 增加 `gradient_accumulation_steps`
  - 使用更小的 `per_device_train_batch_size`

### 3. 路径问题

- 所有路径使用绝对路径或 `~` 开头的路径
- 确保输出目录有写入权限
- 如果数据集在HuggingFace Hub上，可以直接使用Hub路径（如 `HuggingFaceH4/deita-10k-v0-sft`）

### 4. 训练时间估算

- **SFT阶段**：3个epoch，根据数据集大小，可能需要数小时到一天
- **DPO阶段**：1个epoch，通常比SFT快一些

### 5. 检查训练是否正常

观察训练日志中的：
- Loss是否正常下降
- 没有OOM错误
- GPU利用率是否正常（应该接近100%）

## 下一步

完成这两个阶段后，你就有了：
1. **REF teacher**：`~/ADPA/data/mistral-7b-deita/ref_teacher`
2. **DPO teacher**：`~/ADPA/data/mistral-7b-deita/dpo_teacher`

这两个模型将用于后续的：
- 学生模型的DCKD训练
- 学生模型的ADPA训练

## 故障排查

### 问题1：找不到数据集
```bash
# 检查数据集路径
ls -la /home/yunbokun/Datasets/sft/deita-10k-v0-sft/
ls -la /home/yunbokun/Datasets/preference\ alignment/dpo-mix-7k/
```

### 问题2：CUDA out of memory
- 减小batch size或增加gradient accumulation
- 检查DeepSpeed配置是否正确加载

### 问题3：模型路径错误
```bash
# 检查模型是否存在
ls -la /home/yunbokun/Models/Teacher/mistralai/Mistral-7B-v0.1/
```

### 问题4：DeepSpeed配置问题
- 确保 `DS_SKIP_CUDA_CHECK=1` 已设置
- 检查 `deepspeed_zero3_2gpu.yaml` 中的 `num_processes: 2`
