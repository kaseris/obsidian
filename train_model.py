import os
import argparse
import logging

from builder import build_trainer

args = argparse.ArgumentParser()

args.add_argument('--wandb_api_key', type=str, required=True)
args.add_argument('--config', type=str, default='configs/resnet_cgd_base.json', required=True)

def main():
    os.environ['WANDB_API_KEY'] = args.wandb_api_key
    logging.basicConfig(level=logging.DEBUG)
    trainer = build_trainer(args.config)
    trainer.train()
    