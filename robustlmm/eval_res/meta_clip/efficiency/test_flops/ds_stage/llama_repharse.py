from vllm import LLM, SamplingParams
from mr_eval.utils.utils import *
import random
import os,sys
sys.path.append("/mnt/petrelfs/songmingyang/code/mm/robustLMM/robustlmm/model_inference/llama_inference/inference")
from prompts import prompt_dict
import torch
random.seed(42)

def convert_to_human_readable_size(num):
    if num / 1e18 > 1:
        return f"{num / 1e18:.2f} E"
    elif num / 1e15 > 1:
        return f"{num / 1e15:.2f} P"
    elif num / 1e12 > 1:
        return f"{num / 1e12:.2f} T"
    elif num / 1e9 > 1:
        return f"{num / 1e9:.2f} B"
    elif num / 1e6 > 1:
        return f"{num / 1e6:.2f} M"
    elif num / 1e3 > 1:
        return f"{num / 1e3:.2f} K"
    else:
        return f"{num}"

prompt_dict = prompt_dict["llava_caption_to_conversation"]
model_path = "/mnt/petrelfs/songmingyang/quxiaoye/models/Meta-Llama-3-70B-Instruct-back"
model_path = "/mnt/petrelfs/songmingyang/quxiaoye/models/Meta-Llama-3-8B-Instruct"
llm = LLM(model=model_path, tensor_parallel_size=8,)

input_data = "/mnt/petrelfs/songmingyang/songmingyang/data/llava_train/LLaVA-Instruct-150K/reformed_data/dr_algo/aug/llava_v1_5_aug200k_caps.jsonl"
input_data = process_jsonl(input_data)

input_data = random.sample(input_data, 1000)

messages  = [
    {
        "role": "system",
        "content": prompt_dict["system"]
    },
    {
        "role": "user",
        "content": prompt_dict["fewshot"][0][0]
    },
    {
        "role": "assistant",
        "content": prompt_dict["fewshot"][0][1]
    },
    {
        "role": "user",
        "content": prompt_dict["fewshot"][1][0]
    },
    {
        "role": "assistant",
        "content": prompt_dict["fewshot"][1][1]
    },
]

from tqdm import tqdm
from copy import deepcopy
target_convs = []
# for idx,item in enumerate(tqdm(input_data)):
#     current_message = deepcopy(messages)
#     current_message.append({"role": "user","content":item["caption"]})
#     target_convs.append(current_message)
current_message = deepcopy(messages)
current_message.append({"role": "user","content":input_data[0]["caption"]})
target_convs = current_message
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    with_flops=True,  # 启用 FLOPs 统计
    record_shapes=True,
    with_stack=True
) as prof:
    outputs = llm.chat(target_convs)

output_text = outputs[0].outputs[0].text
total_flops = sum(event.flops for event in prof.key_averages() if event.flops is not None)

print("total_flops:",convert_to_human_readable_size(total_flops))
