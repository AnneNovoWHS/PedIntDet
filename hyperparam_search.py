import argparse

import torch
import optuna
from omegaconf import OmegaConf

from sklearn.metrics import f1_score

from src.loader import get_loader
from src.models import build_model
from src.utils import setup_env

##################################################################################################

# get args 
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default=None)
parser.add_argument("--n-trials", type=int, default=100)
args = parser.parse_args()

# load config
cfg = OmegaConf.load('./configs/'+args.config)

##################################################################################################

def objective(trial):
    # training hyperparameters to tune
    dropout = trial.suggest_float("dropout", 0.0, 0.6, step=0.1)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    lr = trial.suggest_float("lr", 1e-6, 1e-2, log=True)
    model_size = trial.suggest_categorical('model_size', [128, 256, 512])

    # override base config
    cfg.model.dropout = dropout
    cfg.model.training.lr = lr
    cfg.model.training.weight_decay = weight_decay
    cfg.model.params.model_size = model_size

    #=================================================================================

    # device, dataloader and model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainloader, valloader = get_loader(cfg) 
    model = build_model(cfg, trainloader, device)

    #=================================================================================

    # train loop
    for epoch in range(model.epoch+1, model.cfg.training.epochs):

        # get model ready
        model.on_epoch_start()
        preds_list, labels_list = [], []

        # go through all data
        for batch in  trainloader:
            frames, labels = batch
            frames, labels = frames.to(device), labels.to(device) 

            # forward + step
            logits = model(frames)           
            model.train_step(logits, labels)

        #=================================================================================

        # evaluate model
        model.eval()
        preds_list, labels_list = [], []
        
        with torch.no_grad():
            for frames, labels in valloader:
                frames, labels = frames.to(device), labels.to(device)
                logits = model(frames)
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).long()
                
                preds_list.append(preds.cpu())
                labels_list.append(labels.cpu())
        
            y_pred_val = torch.cat(preds_list).view(-1).numpy()
            y_true_val = torch.cat(labels_list).view(-1).numpy()

        #=================================================================================
        
        f1_val = f1_score(y_true_val, y_pred_val, zero_division=0)

        #=================================================================================

        # check if prune trial
        trial.report(f1_val, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    
    return f1_val
        

##################################################################################################

def main() -> None:

    # set seed and precision
    setup_env(seed=cfg.general.seed)

    # Create & run study, maximizing validation F1
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    # Print out best trial
    print("Best trial:")
    print(f"  Value (val-F1): {study.best_value:.4f}")
    print("  Params:")
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")

##################################################################################################

if __name__ == "__main__":
    main()
