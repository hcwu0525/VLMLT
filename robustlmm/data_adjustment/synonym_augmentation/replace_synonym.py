from transformers import HfArgumentParser

from dataclasses import dataclass, field
from typing import Dict, Sequence, Optional, List
from robustlmm.data_adjustment.synonym_augmentation.synonym_replacer import replacer_dict
        
if __name__ == "__main__":
        
    @dataclass
    class InferenceArguments:
        function:str = field(default="base_replace")
        input_path:str = field(default=None)
        output_path:str = field(default=None)
        synonym_path:str = field(default=None)
        threads:int = field(default=32)
        distribution_reverse_index_files: List[str] = field(default_factory=list)
        
        
    parser = HfArgumentParser(InferenceArguments)
    args = parser.parse_args_into_dataclasses()[0]

    parser = replacer_dict[args.function](args)
    # parser.sequential_process()
    parser.parallel_process(args.threads)
    # parser.sequential_process(args.threads)