from .nn_replacer import *

__all__ = [NNReplacer,ReplaceInfoExtractor]

replacer_dict=dict(
    base_replace=NNReplacer,
    extract_replace_info=ReplaceInfoExtractor,
)