import argparse, time

from omegaconf import OmegaConf

import torch
from torchsummary import summary

from src.loader import get_loader
from src.training import trainer
from src.models import build_model
from src.utils import setup_env

##################################################################################################

def main() -> None:

    # get args 
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=-1)
    args = parser.parse_args()

    # set config
    if args.checkpoint is not None: cfg = OmegaConf.load(args.checkpoint+'/cfg.yml')
    elif args.config is not None:   cfg = OmegaConf.load('./configs/'+args.config)
    else: raise ValueError('No config or checkpoint was given.')

    # set seed and precision and get device
    setup_env(seed=cfg.general.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # dataloader 
    trainloader, valloader = get_loader(cfg, args.num_samples) 

    # model
    model = build_model(cfg, trainloader, device, args.checkpoint)
    summary(model)
    
    # start training
    print('\nTraining on', device, '\n')
    trainer(trainloader=trainloader,
            valloader=valloader,
            model=model,
            device=device,
            cfg=cfg)

##################################################################################################

if __name__ == "__main__":
    # get start time
    start_timestamp = time.time()

    # train model
    main()

    # info on training time
    time_passed = time.time()-start_timestamp
    print(f'\nTraining finished in {time_passed//3600}h {(time_passed%3600)//60}min {time_passed%60:.0f}s\n')