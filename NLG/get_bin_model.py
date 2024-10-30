import torch
import transformers

model = transformers.GPT2LMHeadModel.from_pretrained("/lamport/shared/hzheng/workspace/NNs/finetune/quantized-training/examples/gpt2/output")
torch.save(model.state_dict(), 'model_6layer.bin')