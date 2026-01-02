import argparse

from omegaconf import OmegaConf

import torch
from torchsummary import summary

from src.loader import get_loader
from src.evaluating import evaluater
from src.models import build_model
from src.utils import setup_env

##################################################################################################

def main() -> None:

    # get args 
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    # set config
    cfg = OmegaConf.load(args.checkpoint+'/cfg.yml')

    # set seed and precision
    setup_env(seed=cfg.general.seed)

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # dataloader
    trainloader, valloader = get_loader(cfg) 

    # model
    model = build_model(cfg, trainloader, device, args.checkpoint)

    # info
    summary(model)
    print('\nEvaluating on', device, '\n')

    # evaluate model
    train_metrics  = evaluater(model, trainloader, device, 'train')
    val_metrics  = evaluater(model, valloader, device, 'val')

    # print evaluation for training and validation data
    for metric, train_val in train_metrics.items():
        val_val = val_metrics.get(metric, float('nan'))
        print(f"{metric.title():<10}  Train: {train_val:.3f}  Val: {val_val:.3f}")

##################################################################################################

if __name__ == "__main__":
    main()
