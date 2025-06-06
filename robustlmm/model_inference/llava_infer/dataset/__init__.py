from .benchmark_caption_datasets import *
from .augmentation_datasets import *


__all__ = ['POPEInferenceDataset', 'MMEInferenceDataset', 'AugmentationCaptionDataset', 'COCO14VALDataset', 'SQACaptionDataset']

dataset_dict = {
    "pope_infer":POPEInferenceDataset,
    "mme_infer":MMEInferenceDataset, 
    "aug_infer":AugmentationCaptionDataset,
    "val2014_infer":COCO14VALDataset,
    "sqa_infer":SQACaptionDataset,
    "plainaug_cap_infer":PlainAugmentationCaptionDataset,
}