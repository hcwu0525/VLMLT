from longtail_evaluator.base_evaluator import BaseEvaluator


import argparse
import yaml

# 读取YAML文件
with open('/mnt/petrelfs/songmingyang/code/mm/robustLMM/robustlmm/analysis/longTail/eval_longtail/scripts/configs/llava_ft.yaml', 'r') as f:
    parameters = yaml.safe_load(f)

# 创建命令行参数解析器
parser = argparse.ArgumentParser()

# 添加命令行参数
for key, value in parameters.items():
    parser.add_argument(f'--{key}', default=value)

# 解析命令行参数
args = parser.parse_args()

evaluator = BaseEvaluator(args)
evaluator.full_procedure()