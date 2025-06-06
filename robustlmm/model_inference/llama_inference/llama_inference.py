from transformers import HfArgumentParser

from vllm import SamplingParams
from dataclasses import dataclass, field
from typing import Dict, Sequence, Optional, List
from robustlmm.model_inference.llama_inference.inference import inference_model_dict
        
if __name__ == "__main__":
        
    @dataclass
    class InferenceArguments:
        model_path: str = field(default="/mnt/petrelfs/songmingyang/quxiaoye/models/Meta-Llama-3-70B-Instruct-back")
        input_file_path: str = field(default="/mnt/petrelfs/songmingyang/songmingyang/data/llava_train/LLaVA-LCS-558K/blip_laion_cc_sbu_558k.json")
        output_file_path: str = field(default="/mnt/petrelfs/songmingyang/songmingyang/data/llava_train/LLaVA-LCS-558K/blip_laion_cc_sbu_558k_objects_llm.jsonl")
        tensor_parallel_size: int = field(default=8)
        function: str = field(default="existence_conversation")
        inference_type: str = field(default="pope_parse_objects_from_caption")
        max_tokens : int = field(default=1024)
        batch_size : int = field(default=32)
        synonym_file : Optional[str] = field(default=None)
        extra_caption_file : Optional[str] = field(default=None)
        extra_answer_file : Optional[str] = field(default=None)
        extra_subset_suffix : Optional[str] = field(default=None)
        
    parser = HfArgumentParser(InferenceArguments)
    args = parser.parse_args_into_dataclasses()[0]

    parser = inference_model_dict[args.inference_type](
        model_path=args.model_path,
        input_file_path=args.input_file_path,
        output_file_path=args.output_file_path,
        tensor_parallel_size=args.tensor_parallel_size,
        function=args.function,
        args = args
    )
    parser.process()