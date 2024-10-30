# Part2： Finetune our GPT2 using LoRA

本项目使用https://github.com/microsoft/LoRA/作为参考库。

## 环境配置

参考https://github.com/microsoft/LoRA/blob/main/examples/NLG/README.md中的配置方法。

1. 下载所有需要的python库，注意torch版本必须小于1.13.0

```bash
pip install -r requirments.txt
```

2. 下载数据集以及预训练权重，这里预训练权重用于和我们自己训练出来的GPT2模型作对比

```bash
bash download_pretrained_checkpoints.sh
bash create_datasets.sh
cd ./eval
bash download_evalscript.sh
cd ..
```

## 代码以及实验细节

### 使用LoRA进行大模型微调

#### 代码介绍

1. beam test

在NLG/src中，gpt2_beam.py负责实现大模型的beam测试，在NLG/beam.sh中，用脚本实现了beam测试代码。

2. decoder

在NLG/src中，gpt2_decode.py负责实现对beam测试结果解码的工作。

3. Finetune

在NLG/src中，gpt2_ft.py负责实现对model.py中实现的llm模型的微调工作，对于GPT2来说，模型中的n_embd、n_layer、n_head是可以配置的。

4. 结果评估

eval/e2e/measure_scores.py实现了对decoder输出结果的评估，对于e2e数据集来说，评估标准为模型生成的句子和标准答案之间差距有多大。

#### 实验步骤

```bash
bash beam_hf_origin.sh # 对huggingface上的原始权重进行beam测试
bash beam_before_ft.sh # 对我们训练出的GPT2的原始权重进行beam测试
```

```bash
bash decode_hf_origin.sh # 对huggingface上的原始权重生成的结果解码
bash evaluation.sh > evaluation_hf_origin.log # 进行结果评估
bash decode_before_ft.sh # 对我们训练出的GPT2的原始权重生成的结果解码
bash evaluation.sh > evaluation_before_ft.log
```

```bash
bash ft_hg.sh # 对huggingface上的原始模型微调
bash ft.sh # 对训练出的GPT2的原始模型微调
```

```bash
bash beam_hf_ft.sh # 对huggingface微调后模型进行beam测试
bash beam_after_ft.sh # 对微调后的训练出的GPT2模型进行beam测试
```

```bash
bash decode_after_ft.sh
bash evaluation.sh > evaluation_after_ft.sh
bash decode_hf_ft.sh
bash evaluation.sh > evaluation_hf_ft.sh
```

#### 实验结果

|       model        |  BLEU  |  NIST  | METEOR | ROUGE_L | CIDEr  |
| :----------------: | :----: | :----: | :----: | :-----: | :----: |
|     hf_origin      | 0.0000 | 0.4170 | 0.0384 | 0.1638  | 0.0017 |
| custom_gpt2_origin | 0.0000 | 0.0590 | 0.0015 | 0.0000  | 0.0000 |
|    hf_finetune     | 0.6913 | 8.7298 | 0.4646 | 0.7140  | 2.5122 |
|   custom_gpt2_ft   | 0.3080 | 3.5745 | 0.1663 | 0.4115  | 0.3270 |

## 说明

* custom_gpt2 model: n_embd=768, n_layers=6, n_head=12
* hf_origin model: n_embd=1024, n_layers=24, n_head=16
* 由于checkpoints以及数据集太大，上传至网盘