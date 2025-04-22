import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import torch.nn.functional as F
import os
import multiprocessing as mp
import time
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    # llm_int8_threshold=6.0
)

def read_jsonl(path):
    with open(path, 'r') as f:
        return [json.loads(l) for l in f]

# device = 'cuda:0'
# model_name_1 = "model_hub/Qwen1.5-1.8B"
# model_name_2 = "model_hub/gpt2_120M"
# tokenizer_1 = AutoTokenizer.from_pretrained(model_name_1)
# tokenizer_1.pad_token = tokenizer_1.eos_token
# tokenizer_2 = AutoTokenizer.from_pretrained(model_name_2)
# tokenizer_2.pad_token = tokenizer_2.eos_token
# model_1 = AutoModelForCausalLM.from_pretrained(model_name_1, 
#                                                quantization_config=bnb_config, 
#                                                device_map=device,
#                                                torch_dtype=torch.float16)
# model_2 = AutoModelForCausalLM.from_pretrained(model_name_2, 
#                                                 # quantization_config=bnb_config,
#                                                 device_map=device,
#                                                 torch_dtype=torch.float16)

# prompt1 = "Tue dep trai qua di hehe \n\n\n hihiho    \n"
# prompt2 = "Tue qua    \n  dep trai di hehe \n hihihi    \n"

# ##################### START #####################
# t1 = time.time()

# tokenized_prompts_1 = tokenizer_1(prompt1, return_tensors="pt", padding=True).to(device)
# tokenized_prompts_2 = tokenizer_2(prompt2, return_tensors="pt", padding=True).to(device)

# with torch.no_grad():
#     logits_1 = model_1(**tokenized_prompts_1).logits
#     logits_2 = model_2(**tokenized_prompts_2).logits

# logits_1 = torch.log_softmax(logits_1, dim=-1).cpu().numpy()
# logits_2 = torch.log_softmax(logits_2, dim=-1).cpu().numpy()

# t2 = time.time()

# ##################### END #####################

# print(f'logits1 = {logits_1.shape}')
# print(f'logits2 = {logits_2.shape}')
# print(f'elapsed time = {t2 - t1}')
# print(f'logits2 = {logits_2[0]}')



model_name_2 = "model_hub/gpt2_120M"
tokenizer_2 = AutoTokenizer.from_pretrained(model_name_2)
tokenizer_2.pad_token = tokenizer_2.eos_token
data = read_jsonl('data/Ultrachat200k/drapt/0_train.jsonl')
print(data[0].keys())
for i in range(100):
    len_ch = len(data[id]['chosen_weight'])
    chosen = data[id]['chosen']
    token_chosen = tokenizer_2(chosen, return_tensors="pt", padding=True)
    print(f'shape = {token_chosen["input_ids"].shape} while len = {len_ch}')
