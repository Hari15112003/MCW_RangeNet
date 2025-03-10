import argparse
import os
import json
from infer import inference

def arguments(raw_args):
    """ Parse input arguments """
    parser = argparse.ArgumentParser(description='script for RangeNet model quantization')
    parser.add_argument('--config', help='model configuration to use', type=str, required=True)
    args = parser.parse_args(raw_args)
    print(vars(args))
    return args


def main(raw_args=None):
    """ Run evaluations """
    args = arguments(raw_args)
    if os.path.exists(args.config):
        with open(args.config) as f_in:
            config = json.load(f_in)

    os.environ["CUDA_VISIBLE_DEVICES"] = config["CUDA"]
    inference(config['dataset'], config["Result"], config['pretrained_model'], config)
 

if __name__ == '__main__':
    main()