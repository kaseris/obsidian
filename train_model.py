import argparse

from builder import build_trainer

args = argparse.ArgumentParser()

args.add_argument('--wandb_api_key', type=str, required=True)
args.add_argument('--config', type=str, default='configs/resnet_cgd_base.json', required=True)

def main():
    trainer = build_trainer(args.config)
    trainer.train()
    