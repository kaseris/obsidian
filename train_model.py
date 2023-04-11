import os
import argparse
import logging

from builder import build_trainer

parser = argparse.ArgumentParser(description='Train interface')


parser.add_argument('--wandb_api_key', type=str, required=True)
parser.add_argument('--config', type=str,
                  default='configs/resnet_cgd_base.json', required=True)
args = parser.parse_args()

def main():
    os.environ['WANDB_API_KEY'] = args.wandb_api_key
    logging.basicConfig(level=logging.DEBUG)
    trainer = build_trainer(args.config)
    trainer.train()

if __name__ == '__main__':
    main()
