from robustlmm.analysis.longTail.language_level.parser.lm_parser import LCS558KLMParser
import argparse
import logging

if __name__ == "__main__":
     # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description='Process some integers.')

    # 添加参数
    parser.add_argument('--input_file_path', type=str, help='input file path', default="lcs558k")
    parser.add_argument('--output_file_path', type=str, help='output file path', default="stanza")
    parser.add_argument("--num_workers",type=int,help="number of workers",default=64)
    parser.add_argument("--verbosity", type=str, help="verbosity", default="info")
    parser.add_argument("--model_type", type=str, help="model type", default="deepseek")
    parser.add_argument("--input_type", type=str, help="input type", default="existence_conversation")

    # 解析参数
    args = parser.parse_args()
    verbosity_dict = {"info":logging.INFO,"debug":logging.DEBUG,"error":logging.ERROR,"warning":logging.WARNING,"critical":logging.CRITICAL}
    logging.basicConfig(level=verbosity_dict[args.verbosity])
    logging.getLogger('httpx').setLevel(logging.CRITICAL)
    logging.getLogger('openai').setLevel(logging.CRITICAL)
    parser = LCS558KLMParser(
        model_type= args.model_type,
        input_type= args.input_type,
        input_dataset_file= args.input_file_path,
        output_file= args.output_file_path,
        model_args  = None,
        model_inference_args = None,
    )
    parser.parallel_process(args.num_workers)