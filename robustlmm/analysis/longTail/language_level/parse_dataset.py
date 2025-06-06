import argparse
import logging

from mhr.utils.utils import process_jsonl,load_json_file
from robustlmm.analysis.longTail.language_level.parser.base_parser import LCS558KParser,Instructmix665KParser,POPEParser, MMEParser

if __name__ == "__main__":
    
    

    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description='Process some integers.')

    # 添加参数
    parser.add_argument('--dataset', type=str, help='dataset file path', default="lcs558k")
    parser.add_argument('--parser', type=str, help='parser', default="stanza")
    parser.add_argument("--num_workers",type=int,help="number of workers",default=64)
    parser.add_argument("--verbosity", type=str, help="verbosity", default="info")
    parser.add_argument("--deduplicate", action="store_true", help="deduplicate")
    parser.add_argument("--function",type=str,help="function",default="count_num")
    parser.add_argument("--input_file_path",type=str,help="input file",default=None)
    parser.add_argument("--output_file_path",type=str,help="output file",default=None)
    
    verbosity_dict = {"info":logging.INFO,"debug":logging.DEBUG,"error":logging.ERROR,"warning":logging.WARNING}
    
    # 解析参数
    args = parser.parse_args()
    if args.deduplicate:
        deduplicate = True
    else:
        deduplicate = False
        
    logging.basicConfig(level=verbosity_dict[args.verbosity])
    if args.function == "count_num":
        if args.dataset == "lcs558k":
            d_parser = LCS558KParser(deduplicate=deduplicate,function=args.function)
            d_parser.parallel_process(args.num_workers)
        elif args.dataset == "instructmix665k":
            d_parser = Instructmix665KParser(deduplicate=deduplicate,function=args.function)
            d_parser.parallel_process(args.num_workers)
        elif args.dataset == "pope":
            d_parser = POPEParser(deduplicate=deduplicate,function=args.function)
            d_parser.parallel_process(args.num_workers)
        elif args.dataset == "mme":
            d_parser = MMEParser(deduplicate=deduplicate,function=args.function)
            d_parser.parallel_process(args.num_workers)
        else:
            raise ValueError("Invalid dataset")
    elif args.function == "object_extraction":
        if args.dataset == "lcs558k":
            d_parser = LCS558KParser(deduplicate=deduplicate,function=args.function,dataset_file=args.input_file_path,output_file_path=args.output_file_path)
            d_parser.parallel_extract(args.num_workers)
        elif args.dataset == "instructmix665k":
            d_parser = Instructmix665KParser(deduplicate=deduplicate,function=args.function,dataset_file=args.input_file_path,output_file_path=args.output_file_path)
            d_parser.parallel_extract(args.num_workers)
        else:
            raise ValueError("Invalid dataset")
    else:
        raise ValueError("Invalid function")