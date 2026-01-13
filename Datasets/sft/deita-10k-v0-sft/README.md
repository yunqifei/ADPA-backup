---
license: mit
language:
- en
size_categories:
- 1K<n<10K
dataset_info:
  features:
  - name: prompt
    dtype: string
  - name: prompt_id
    dtype: string
  - name: messages
    list:
    - name: content
      dtype: string
    - name: role
      dtype: string
  splits:
  - name: train_sft
    num_bytes: 349335841.1
    num_examples: 9500
  - name: test_sft
    num_bytes: 18386096.9
    num_examples: 500
  - name: train_gen
    num_bytes: 336873383
    num_examples: 9500
  - name: test_gen
    num_bytes: 16979716
    num_examples: 500
  download_size: 289754284
  dataset_size: 721575037.0
configs:
- config_name: default
  data_files:
  - split: train_sft
    path: data/train_sft-*
  - split: test_sft
    path: data/test_sft-*
  - split: train_gen
    path: data/train_gen-*
  - split: test_gen
    path: data/test_gen-*
---


# Dataset Card for Deita 10k v0

This is a formatted version of [`hkust-nlp/deita-10k-v0`](https://huggingface.co/datasets/hkust-nlp/deita-10k-v0) to store the conversations in the same format as the OpenAI SDK.

## Citation
If you find this dataset useful, please cite the original dataset:

```
@misc{liu2023what,
      title={What Makes Good Data for Alignment? A Comprehensive Study of Automatic Data Selection in Instruction Tuning}, 
      author={Wei Liu and Weihao Zeng and Keqing He and Yong Jiang and Junxian He},
      year={2023},
      eprint={2312.15685},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
