from mhr.utils.utils import load_json_file,process_jsonl,append_jsonl
import os
import logging
from robustlmm.analysis.longTail.language_level.parser.prompts import prompt_dict
from copy import deepcopy
from transformers import AutoTokenizer
from tqdm import tqdm
from typing import Dict

import numpy as np
import ray
from packaging.version import Version
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from vllm import LLM, SamplingParams


logger = logging.getLogger(__name__)
sampling_params = SamplingParams(temperature=0.0, top_p=0.95)

# Set tensor parallelism per instance.
tensor_parallel_size = 8

# Set number of instances. Each instance will use tensor_parallel_size GPUs.
num_instances = 1

class LLMPredictor:

    def __init__(self):
        # Create an LLM.
        self.llm = LLM(model="/mnt/petrelfs/songmingyang/quxiaoye/models/Meta-Llama-3-70B-Instruct-back",
                    tensor_parallel_size=tensor_parallel_size)

    def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, list]:
        # Generate texts from the prompts.
        # The output is a list of RequestOutput objects that contain the prompt,
        # generated text, and other information.
        outputs = self.llm.generate(batch["prompt"], sampling_params)
        prompt = []
        generated_text = []
        for output in outputs:
            prompt.append(output.prompt)
            generated_text.append(' '.join([o.text for o in output.outputs]))
        return {
            "prompt": prompt,
            "generated_text": generated_text,
            "id": batch["id"]
        }

class LlamaParser():
    def __init__(
        self,
        model_path: str = "/mnt/petrelfs/songmingyang/quxiaoye/models/Meta-Llama-3-70B-Instruct-back",
        input_file_path = "/mnt/petrelfs/songmingyang/songmingyang/data/llava_train/LLaVA-LCS-558K/blip_laion_cc_sbu_558k.json",
        output_file_path = "/mnt/petrelfs/songmingyang/songmingyang/data/llava_train/LLaVA-LCS-558K/blip_laion_cc_sbu_558k_objects_llm.jsonl",
        tesor_parallel_size = tensor_parallel_size,
        sampling_params = sampling_params,
        function = "existence_conversation",
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.input_file_path = input_file_path
        self.output_file_path = output_file_path
        self.sampling_params = sampling_params
        self.tensor_parallel_size = tesor_parallel_size
        self.function = function
        logger.info(f"selected function: {self.function}")
        
        self.resume_from_file()
        self.prepare_dataset()
        # self.llm = LLM(model=model_path,
        #                tensor_parallel_size=self.tensor_parallel_size)
    
    def resume_from_file(self):
        assert os.path.exists(self.input_file_path), f"Input dataset file {self.input_file_path} does not exist."
        if os.path.exists(self.output_file_path):
            logger.info(f"checkpoint detected! Resume from file: {self.output_file_path}")
            self.cache = process_jsonl(self.output_file_path)
            self.processed_id = {item["id"]:1 for item in self.cache}
        else:
            os.makedirs(os.path.dirname(self.output_file_path),exist_ok=True)
            self.cache = []
            self.processed_id = {}
        
    
    def prepare_dataset(self):
        # raw_data = load_json_file(self.input_file_path)[:100]
        raw_data = load_json_file(self.input_file_path)
        messages = self.prepare_prompt()
        self.data_dict = {i['id']:i for i in raw_data}
        self.meta_data = []
        logger.info(f"Start loading samples")
        for item in tqdm(raw_data):
            if self.processed_id.get(item['id'],False):
                continue
            conv = item["conversations"]
            res_str = ""
            for shot in conv:
                res_str += f"{shot['from']}: {shot['value']}\n"
            messages_iter = deepcopy(messages)
            messages_iter.append({"role": "user", "content": res_str})
            prompt = self.tokenizer.apply_chat_template(messages_iter,tokenize=False)

            self.meta_data.append({"id":item['id'],"prompt":prompt})
            
        self.data_length = len(self.meta_data)
        logger.info(f"Finish loading samples, total samples: {self.data_length}")
        self.meta_data = ray.data.from_items(self.meta_data)
    
    def scheduling_strategy_fn(self):
        # One bundle per tensor parallel worker
        pg = ray.util.placement_group(
            [{
                "GPU": 1,
                "CPU": 16
            }] * self.tensor_parallel_size,
            strategy="STRICT_PACK",
        )
        return dict(scheduling_strategy=PlacementGroupSchedulingStrategy(
            pg, placement_group_capture_child_tasks=True))
 
    def prepare_prompt(self):
        prompt = prompt_dict[self.function]
        messages=[{"role": "system", "content":prompt["system"]}]
        for shot in prompt["fewshot"]:
            messages.append({"role": "user", "content": shot[0]})
            messages.append({"role": "assistant", "content": shot[1]})
        logger.info(f"Prompt prepared: {prompt['system']}")
        return messages


        
    def process(self):
        resources_kwarg = {}
            
        if self.tensor_parallel_size == 1:
            # For tensor_parallel_size == 1, we simply set num_gpus=1.
            resources_kwarg["num_gpus"] = 1
        else:
            # Otherwise, we have to set num_gpus=0 and provide
            # a function that will create a placement group for
            # each instance.
            resources_kwarg["num_gpus"] = 0
            resources_kwarg["ray_remote_args_fn"] = self.scheduling_strategy_fn
        # Apply batch inference for all input data.
        self.ds = self.meta_data.map_batches(
            LLMPredictor,
            # Set the concurrency to the number of LLM instances.
            concurrency=num_instances,
            # Specify the batch size for inference.
            batch_size=32,
            **resources_kwarg,
        )
        # outputs = self.ds.take_all()
        structure="<|start_header_id|>assistant<|end_header_id|>\n\n"
        ds_iter=self.ds.iterator().iter_batches()
        # for output in outputs:
        #     prompt = output["prompt"]
        #     generated_text = output["generated_text"][len(structure):]
        #     id=output['id']
        #     # logger.info(f" Generated text: {generated_text!r}, id:{id}")
        #     item = self.data_dict[id]
        #     item["objects"] = generated_text
        #     append_jsonl(item,self.output_file_path)
        #     structure="<|start_header_id|>assistant<|end_header_id|>\n\n"

        for idx,batch in enumerate(tqdm(ds_iter)):
            prompt = batch["prompt"]
            generated_text=batch["generated_text"]
            ids = batch["id"]
            logger.info(f"Checkpoint reached, writing into file. ({(idx+1)*256}/{self.data_length} processed)")
            for i in range(len(ids)):
                logger.debug(f" Generated text: {generated_text[i][len(structure):]}, id:{ids[i]}")
                # logger.info(f" Generated text: {generated_text!r}, id:{id}")
                item = self.data_dict.get(ids[i],None)
                if item:
                    item["objects"] = generated_text[i][len(structure):]
                    append_jsonl(item,self.output_file_path)
                
        