from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer

import argparse
import torch, time, json, os
from pathlib import Path
from tqdm import tqdm
from datetime import timedelta

import warnings

warnings.filterwarnings("ignore")
def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='model_hub/gpt2_120M')
    parser.add_argument('--data_frac', type=int, default=0)
    parser.add_argument('--frac_len', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default='data/Ultrachat200k/ite0')
    parser.add_argument('--world_size', type=int, default=1) # controls the number of gpus vLLM is allowed to use
    parser.add_argument('--input_dir', type=str, default='data/Ultrachat200k/trainSFT.jsonl')
    parser.add_argument('--split', type=str, default='train')
    return parser.parse_args()

def read_jsonl(input_path):
    with open(input_path, 'r') as f:
        data = [json.loads(line) for line in f]
    return data

def main():
    args = parse_arguments()
    model_path = args.model
    data_frac = args.data_frac
    world_size = args.world_size
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # load a base model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)   
    tokenizer.pad_token = tokenizer.eos_token

    llm = LLM(
        model=model_path,
        tensor_parallel_size=world_size,
    )

    sampling_params = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=256)

    # load data
    # data = load_dataset(args.input_dir, split=args.split)
    # data = data.shuffle(seed=42)
    # if args.frac_len > 0:
    #     sub_len = args.frac_len 
    #     if sub_len*(data_frac+1) > len(data):
    #         data = data[sub_len*data_frac:]['real']
    #     else:
    #         data = data[sub_len*data_frac:sub_len*(data_frac+1)]['real']
    # else:
    #     data = data[:]['real']

    # prompts_all = ["### Instruction: " + data[idx][0]['content'] + "\n\n### Response: " for idx in range(len(data))]
    # prompts_old = [data[idx][0]['content'] for idx in range(len(data))]
    # corrects_all = [data[idx][1]['content'] for idx in range(len(data))]
    data = read_jsonl(args.input_dir)
    if args.frac_len > 0:
        sub_len = args.frac_len 
        if sub_len*(data_frac+1) > len(data):
            data = data[sub_len*data_frac:]
        else:
            data = data[sub_len*data_frac:sub_len*(data_frac+1)]
    else:
        data = data[:]
    prompts_all = [data[idx]['prompt']for idx in range(len(data))]
    prompts_old = [data[idx]['prompt'] for idx in range(len(data))]
    corrects_all = [data[idx]['ground_truth'] for idx in range(len(data))]

    start=time.time()

    #run vllm
    results_gathered = list(map(lambda x: x.outputs[0].text, 
                                llm.generate(prompts_all, sampling_params)))

    results = [r.replace("</s>","").lstrip() for r in results_gathered]

    timediff=time.time()-start
    print(f"time elapsed: {timediff}")

    # collecting data
    for idx in tqdm(range(len(corrects_all))):
        # d = {"real": [{"role": "user", "content": prompts_old[idx]}, {"role": "assistant", "content": corrects_all[idx]}], "generated": [{"role": "user", "content": prompts_old[idx]}, {"role": "assistant", "content": results[idx]}]}
        d = {"prompt": data[idx]['prompt'], "chosen": data[idx]['ground_truth'], "rejected": results[idx]}
        if args.split == 'test':
            filename = f"{args.output_dir}/{args.data_frac}_test.jsonl"
        else:
            filename = f"{args.output_dir}/{args.data_frac}_train.jsonl"
        with open(filename, 'a') as f:
            json.dump(d, f)
            f.write('\n')


if __name__ == "__main__":
    main()
