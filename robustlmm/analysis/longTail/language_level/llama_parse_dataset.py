from robustlmm.analysis.longTail.language_level.parser.llama_parser import LlamaParser
import argparse
import logging
from vllm import LLM, SamplingParams

if __name__ == "__main__":
     # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description='Process some integers.')

    # 添加参数
    parser.add_argument('--input_file_path', type=str, help='input file path', default="lcs558k")
    parser.add_argument('--output_file_path', type=str, help='output file path', default="stanza")
    parser.add_argument("--model_path", type=str, help="model path", default="/mnt/petrelfs/songmingyang/quxiaoye/models/Meta-Llama-3-70B-Instruct-back")
    parser.add_argument("--verbosity", type=str, help="verbosity", default="info")
    parser.add_argument("--gpus", type=int, help="model type", default=8)
    parser.add_argument("--temperature", type=float, help="temperature", default=0.0)
    parser.add_argument("--top_p", type=float, help="top_p", default=0.95)
    parser.add_argument("--function",type=str,help="function",default="existence_conversation")

    # 解析参数
    args = parser.parse_args()
    verbosity_dict = {"info":logging.INFO,"debug":logging.DEBUG,"error":logging.ERROR,"warning":logging.WARNING,"critical":logging.CRITICAL}
    logging.basicConfig(level=verbosity_dict[args.verbosity])
    
    sampling_params = SamplingParams(temperature=args.temperature, top_p=args.top_p)
    parser = LlamaParser(
        model_path=args.model_path,
        input_file_path=args.input_file_path,
        output_file_path=args.output_file_path,
        tesor_parallel_size=args.gpus,
        sampling_params=sampling_params,
        function=args.function
    )
    parser.process()