from robustlmm.analysis.longTail.vision_level.parser.groundingDINO_parser import BaseGroudingDINOParser

import argparse
import logging

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description='Process some integers.')
    argparser.add_argument("--model_path",type=str,help="model path",default=None)
    argparser.add_argument("--input_file_path",type=str,help="input file",default=None)
    argparser.add_argument("--output_file_path",type=str,help="output file",default=None)
    argparser.add_argument("--image_dir_path",type=str,help="image dir",default=None)
    argparser.add_argument("--batch_size",type=int,help="batch size",default=64)
    
    args = argparser.parse_args()
    logging.basicConfig(level=logging.INFO)
    
    parser = BaseGroudingDINOParser(
        model_path=args.model_path,
        input_file_path=args.input_file_path,
        image_dir_path=args.image_dir_path,
        output_file_path=args.output_file_path,
        batch_size=args.batch_size
    )
    parser.parallel_process()
    