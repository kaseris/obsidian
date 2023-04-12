import os
import argparse
import logging

parser = argparse.ArgumentParser(description='Train interface')


parser.add_argument('--wandb_api_key', type=str, required=True)
parser.add_argument('--config', type=str,
                  default='configs/resnet_cgd_base.json', required=True)
parser.add_argument('--dataset_path', type=str, required=True)
args = parser.parse_args()


def main():
    logging.basicConfig(level=logging.DEBUG)
    trainer = build_trainer(args.config)
    trainer.train()

if __name__ == '__main__':
    os.environ['DATASET_DIR'] = args.dataset_path
    os.environ['WANDB_API_KEY'] = args.wandb_api_key
    from builder import build_trainer
    main()
