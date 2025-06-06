import neo4j
import argparse
import logging
from robustlmm.analysis.longTail.language_level.parser.co_occurrence_parser import CoOccurrenceBaseParser, CoOccurrenceVisualParser


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')

    # 添加参数
    parser.add_argument("--object_file_path",type=str,help="input file",default=None)
    parser.add_argument("--verbosity", type=str, help="verbosity", default="info")
    parser.add_argument("--num_workers",type=int,help="number of workers",default=64)
    # parser.add_argument('--dataset_type', type=str, help='dataset file path', default="lcs558k")
    # parser.add_argument('--parser_type', type=str, help='parser', default="pos")
    parser.add_argument('--database_name', type=str, help='parser', default="lcs558kpos")
    parser.add_argument('--dataset_uri', type=str, help='parser', default="neo4j://10.140.54.10:7687")
    parser.add_argument('--username', type=str, help='parser', default="neo4j")
    parser.add_argument('--password', type=str, help='parser', default="12345678")
    parser.add_argument('--ckpt_file_path', type=str, help='parser', default="ckpt/co_occurrence_parser.jsonl")
    parser.add_argument('--parser_type',default="base",type=str)
    
    verbosity_dict = {"info":logging.INFO,"debug":logging.DEBUG,"error":logging.ERROR,"warning":logging.WARNING}

    args = parser.parse_args()
    logging.basicConfig(level=verbosity_dict[args.verbosity]) 
    if args.parser_type == "base":
        database_dict = dict(uri=args.dataset_uri,username=args.username,password=args.password,database_name=args.database_name)
        co_occurrence_parser = CoOccurrenceBaseParser(neo4j_dict=database_dict,object_file_path=args.object_file_path, ckpt_file_path=args.ckpt_file_path)
    elif args.parser_type == "visual":
        database_dict = dict(uri=args.dataset_uri,username=args.username,password=args.password,database_name=args.database_name)
        co_occurrence_parser = CoOccurrenceVisualParser(neo4j_dict=database_dict,object_file_path=args.object_file_path, ckpt_file_path=args.ckpt_file_path)
    
    co_occurrence_parser.parallel_process(args.num_workers)
    
    

