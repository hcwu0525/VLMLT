from .augment_parser import *
from .benchmark_parser import *
from .object_parser import *

__all__ = [POPELlamaParser,AugCaptionLlamaParser,ShareGPT4VTrainsetParser,TokenAugmentParser,VAL2014ExtractParser,VQAV2WhatwordParser,SQAWhatwordParser,SQACaption2objectParser,PlainAugCaption2instanceParser]

inference_model_dict = {
    "pope_parse_objects_from_caption": POPELlamaParser,
    "llava_caption_to_conversation": AugCaptionLlamaParser,
    "sharegpt4_parse_objects": ShareGPT4VTrainsetParser,
    "llava_parse_objects": LLaVATrainsetParser,
    "token_augment": TokenAugmentParser,
    "val2014_extract": VAL2014ExtractParser,
    "vqav2_whatword": VQAV2WhatwordParser,
    "sqa_whatword": SQAWhatwordParser,
    "sqa_caption2object": SQACaption2objectParser,
    "plainaug_caption2instance": PlainAugCaption2instanceParser,
    "tail_token_lm_rephrase":TailTokenLMReParaphrase,
}