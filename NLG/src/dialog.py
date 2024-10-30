# import argparse
# import time
# import math
# import os, sys
# import numpy as np
# import itertools

# import torch
# import random
# from torch.utils.data import DataLoader
# from data_utils import FT_Dataset
# from model import GPT2Config, GPT2LMModel
# torch.set_printoptions(threshold=100000)

# config = GPT2Config(
#     n_embd=1024, n_layer=24, n_head=16, 
#     lora_attn_dim=4, 
#     lora_attn_alpha=32, 
#     lora_dropout=0.1,
# )
# lm_net = GPT2LMModel(config)

# print('loading weights')
# lm_net.load_weight(torch.load('/lamport/makkapakka/jingxi_chen/GPT2/NLG/trained_models/GPT2_M/e2e/model.26290.pt'))
# lm_net = lm_net.cuda()


import json
import torch
import argparse
from gpt2_beam import _reorder_cache, _calc_banned_ngram_tokens, _enforce_repetition_penalty_, _postprocess_next_token_scores
import torch.utils.data.distributed
from encoder import get_encoder
from data_utils import FT_Dataset 
from torch.utils.data import DataLoader
from model import GPT2Config, GPT2LMModel
import torch.nn.functional as F
from gpu import (
    add_gpu_params, 
    parse_gpu, 
    distributed_opt, 
    distributed_gather, 
    distributed_sync, 
    cleanup
)

def _add_beam_candidate(
    best_score, 
    best_sequence, 
    batch_size, 
    num_beams, 
    beam_scores, 
    history, 
    eos_token_id=None
):
    last_tokens = history[:, -1]
    for _i in range(batch_size * num_beams):
        if eos_token_id is None or last_tokens[_i] in eos_token_id:
            cur_len = history.shape[-1]
            _score = beam_scores.view(-1)[_i] / cur_len ** 0.8

            batch_id = _i // num_beams

            if not batch_id in best_score or best_score[batch_id] < _score:
                best_score[batch_id] = _score
                best_sequence[batch_id][:cur_len] = history[_i]

            beam_scores.view(-1)[_i] = -float("inf")


parser = argparse.ArgumentParser(description='GPT Model Sentence Input-Output Script')
add_gpu_params(parser)
parser.add_argument('--vocab', type=str, required=True, help='Path to vocabulary file')
parser.add_argument('--model_card', default='gpt2.sm', choices=['gpt2.sm', 'gpt2.md', 'gpt2.lg'], help='Model size')
parser.add_argument('--init_checkpoint', type=str, required=True, help='Path to model checkpoint')
parser.add_argument('--seq_len', type=int, default=512, help='Maximum sequence length')
parser.add_argument('--output_file', type=str, default='output.json', help='Output file for generated responses')
parser.add_argument('--add_bos', type=bool, default=True, help='If add bos token or not')
parser.add_argument('--add_eos', type=bool, default=True, help='If add eos token or not')
parser.add_argument('--input_file', type=str, default='/lamport/makkapakka/jingxi_chen/GPT2/NLG/data/e2e/test_formatted.jsonl', help='input reference file')
parser.add_argument('--eval_len', type=int, default=64, help='eval_length')
parser.add_argument('--eos_token_id', action='append', type=int, default=[50256], 
                    help='eos token id')
args = parser.parse_args()

# Load encoder and model
encoder = get_encoder(args.vocab)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.model_card == 'gpt2.sm':
    config = GPT2Config(n_embd=768, n_layer=12, n_head=12)
elif args.model_card == 'gpt2.md':
    config = GPT2Config(n_embd=1024, n_layer=24, n_head=16)
else:
    config = GPT2Config(n_embd=1280, n_layer=36, n_head=20)

model = GPT2LMModel(config).to(device)
model.load_weight(torch.load(args.init_checkpoint))
model.eval()

# Function to process input and generate response
def generate_response(input_sentence):
# encode
    # Tokenize input
    bos = 50256
    eos = 50256

    context, _ = encoder.encode(input_sentence)
    context += [bos] if args.add_bos else []
    # input_tensor = torch.tensor([input_], device=device)
    completion = []
    completion += [eos] if args.add_eos else []

    ft_json = {}
    ft_json['context'] = context
    ft_json['completion'] = completion
    with open(args.output_file, 'w') as f:
        f.write(json.dumps(ft_json) + '\n')

    dialog_data = FT_Dataset(args.output_file, 1, args.seq_len, 64)
    dialog_loader = DataLoader(dialog_data, batch_size=1, num_workers=0)
    # print(context)

# inference
    all_predictions = {}
    with torch.no_grad():
        for idx, data in enumerate(dialog_loader):
            # print(data)
            data = {key: value for key, value in data.items()}
            _id = data['id'].to(device)
            _query = data['query'].to(device)
            _query_len = data['query_len'].to(device)
            output = None
            score = None
            batch_size = _id.size(0)
            num_beams = 1
            length_panalty = 0.8

            _batch = torch.arange(0, _id.size(0), device=device, dtype=torch.long)
            past = None
            len_past = None

            _query = _query.repeat(1, num_beams).view(batch_size * num_beams, -1)
            _query_len = _query_len.unsqueeze(-1).repeat(1, num_beams).view(-1)

            _bbatch = _batch.unsqueeze(-1).repeat(1, num_beams).view(-1)

            beam_scores = torch.zeros(
                (batch_size, num_beams), dtype=torch.float, device=_query.device
            )

            best_sequence = torch.zeros(
                (batch_size, args.eval_len), dtype=torch.long, device=_query.device
            )
            best_score = {}

            history = None
            with torch.no_grad():
                for i in range(0, args.eval_len):
                    if i == 0:
                        logits, past = model(_query)
                        logits = logits[_bbatch, (_query_len-1).long(), :]
                    else:
                        logits, past = model(token_id, past=past, len_past=len_past)
                        logits = logits[:, -1, :]
                    logits = _postprocess_next_token_scores(
                        logits,
                        history,
                        i,
                        batch_size,
                        num_beams,
                        repetition_penalty=1.0,
                        no_repeat_ngram_size=4,
                        min_length=0,
                        eos_token_id=eos
                    )
                    softmax_probs = F.softmax(logits, dim=-1)
                    vocab_size = softmax_probs.shape[-1]
                    _logprob = torch.log(softmax_probs)
                    if i == 0:
                        next_scores = _logprob.view(batch_size, num_beams, -1)[:, 0, :] # batch_size, vocab
                        
                    else:
                        next_scores = beam_scores.unsqueeze(-1) + _logprob.view(batch_size, num_beams, -1)
                        next_scores = next_scores.view(batch_size, -1) # batch_size, beam * vocab

                    next_scores, next_tokens = torch.topk(
                        next_scores, num_beams, dim=1, largest=True, sorted=True
                    )     # batch_size, num_beams
                    
                    beam_id = (next_tokens // vocab_size).view(-1)    # batch_size * num_beams
                    token_id = (next_tokens % vocab_size).view(-1).unsqueeze(-1) # batch_size, num_beams

                    beam_idx = beam_id.view(batch_size, num_beams) + (_batch * num_beams).unsqueeze(-1)
                    past = _reorder_cache(past, beam_idx.view(-1))                
                    beam_scores = next_scores # batch_size, num_beams
                    len_past = (_query_len + i).long()

                    if history is None:
                        history = token_id.detach()
                    else:
                        history = torch.cat((history[beam_idx.view(-1)], token_id.detach()), dim=1).detach()

                    _add_beam_candidate(
                        best_score, best_sequence, batch_size, num_beams, beam_scores, history, 
                        eos_token_id=args.eos_token_id
                    )

                _add_beam_candidate(
                best_score, best_sequence, batch_size, num_beams, beam_scores, history
                )
            # with torch.no_grad():
            #     _id = distributed_gather(args, _id)
            #     output = distributed_gather(args, best_sequence)
            #     #score = distributed_gather(args, score)
            #     distributed_sync(args)
            output = best_sequence
            # print(output)
            # if args.rank == 0:
            #     _id = _id.view(-1).cpu()
            #     output = output.view(-1, output.shape[-1]).cpu()
            #     #score = score.view(-1, score.shape[-1]).cpu()
            #     print("!")
            #     for _b in range(0, _id.shape[-1]):
            #         _i = int(_id[_b].item())
            #         all_predictions[_i] = {}
            #         all_predictions[_i]['id'] = _i
            #         all_predictions[_i]['predict'] = output[_b].tolist()
            output = output.view(-1, output.shape[-1]).cpu()
            all_predictions['predict'] = output
            
# decode
    enc = get_encoder(args.vocab)
    # _id = all_predictions['id']
    _pred_tokens = all_predictions['predict'][0].tolist()
    # print(_pred_tokens.shape)
    # with open(args.input_file, 'r', encoding='utf8') as input_reader:
    #     context_list = []
    #     for line in input_reader:
    #         items = json.load(line.strip())
    #         context = items['context']
    #         context_list.append(context)
    # _key = context_list[_id]


    return enc.decode(_pred_tokens).split('<|endoftext|>')[0].split('\n\n')[0].strip()

if __name__ == '__main__':
    print("Enter 'exit' to quit.")
    while True:
        sentence = input("Input sentence: ")
        if sentence.lower() == 'exit':
            break
        print("Generated response:", generate_response(sentence))
