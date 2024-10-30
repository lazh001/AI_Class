import argparse
import logging
import os

import torch
from accelerate.utils import get_max_memory
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from quantized_training import (
    add_qspec_args,
    get_default_quantizer,
    prepare_pt2e,
    quantize,
    plot_histogram,
    plot_layer_range,
    setup_logging,
)
from quantized_training.pt2e_utils import dispatch_model, get_device_map
from quantized_training.quantize_pt2e import convert_pt2e

logger = logging.getLogger(__name__)


# local model and dataset
model_id = "/lamport/shared/hzheng/workspace/NNs/finetune/quantized-training/examples/gpt2/output_e10"
#model_id = "/lamport/shared/hzheng/workspace/model/gpt2/"

def parse_args():
    parser = argparse.ArgumentParser(description="Process model parameters.")
    parser.add_argument('--max_length', type=int, default=1024, help='Maximum sequence length')
    parser.add_argument('--stride', type=int, default=512, help='Stride for processing the data')
    parser.add_argument('--output_dir', default=None, help='Output directory for histograms')
    parser.add_argument(
        '--torch_dtype',
        default="float32",
        choices=["auto", "bfloat16", "float16", "float32"],
        help=(
            "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
            "dtype will be automatically derived from the model's weights."
        )
    )
    add_qspec_args(parser)
    return parser.parse_args()

@setup_logging
def main(args):
    device = torch.device(f"cuda:{args.gpu}" if args.gpu is not None else "cuda")
    torch_dtype = (
        args.torch_dtype
        if args.torch_dtype in ["auto", None]
        else getattr(torch, args.torch_dtype)
    )
    print(f"Device: {device}, dtype: {torch_dtype}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
    )
    model.to(device)

    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    print("Model loaded.")
    text = ["The king says", 
            "Once upon a time, ", 
            "The weather is", 
            "New York is a city", 
            "The stock market", 
            "The best way to", 
            "Artificial intelligence is",
            "One of the most important things in life is",
            "I am a student", 
            "At the end of the year,",]
    print("Input text:", text)

    # 将每个输入文本单独编码并生成输出
    for input_text in text:
        print("\n\n\n------------------------------------\n")
        encoded_input = tokenizer.encode(input_text, return_tensors='pt').to(device)

        output = model.generate(
            encoded_input,
            max_length=50,
            num_return_sequences=1,
            repetition_penalty=1.15,
            do_sample=True,
            top_k=4,
            top_p=0.8
        )

        # 解码生成的文本
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        print("Generated text for '{}': \n{}".format(input_text, generated_text))


if __name__ == "__main__":
    args = parse_args()
    main(args)