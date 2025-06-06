from robustlmm.model_inference.sharecaptioner_inference.captioner_infer import ShareCaptionerInference
from dataclasses import dataclass, field

from typing import Dict, Sequence, Optional,List
from transformers import HfArgumentParser
from transformers import AutoTokenizer,AutoModel
import torch 

if __name__ == "__main__":

    @dataclass
    class DataArguments:

        input_path: List[str] = field(default_factory=list)
        output_path: str = field(default=None)
        dataset_type: str = field(default="base")
        batch_size: int = field(default=32)
        num_workers: int = field(default=4)
        aug_target_dir_path: Optional[str] = field(default=None)
        image_rel_path: str = field(default="data")
        
        
    @dataclass
    class InferenceArguments:
        model_path: str = field(default="/mnt/petrelfs/songmingyang/songmingyang/model/mm/ShareCaptioner")
        float_type: str = field(default="float16")

            
    parser = HfArgumentParser(
    (InferenceArguments,DataArguments))
    
    
    inference_args, data_args = parser.parse_args_into_dataclasses()
    inference_module = ShareCaptionerInference(inference_args=inference_args,data_args=data_args)
    inference_module.inference()
    # tokenizer = AutoTokenizer.from_pretrained("/mnt/petrelfs/songmingyang/songmingyang/model/mm/ShareCaptioner", trust_remote_code=True)
    # model = AutoModel.from_pretrained("/mnt/petrelfs/songmingyang/songmingyang/model/mm/ShareCaptioner", trust_remote_code=True,device="cpu")
    # model.tokenizer = tokenizer
    # model=model.to(torch.float16)
    print("Done")