import os
import json
import torch
from datetime import datetime
from flatten_dict import flatten

from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter 

##################################################################################################

class Logger():
    def __init__(self, cfg):
        self.cfg = cfg

        # setup tensorboard writer if set in the config
        if cfg.general.logging:

            if hasattr(cfg.general, 'log_path'):
                # create writer and keep old log path
                self.log_path = cfg.general.log_path
                self.writer = SummaryWriter(self.log_path)
        
            else:
                # setup the path and name
                run_name = datetime.now().strftime("%Y-%m-%d_%H-%M")
                self.log_path = './logs/'+cfg.model.name+'/'+run_name
                os.makedirs(self.log_path, exist_ok=True) 

                # create writer and log hparams
                self.writer = SummaryWriter(self.log_path)
                self.log_hparams_tensorboard()
                self.log_config()

    #=================================================================================

    def log_hparams_tensorboard(self):
        if self.cfg.general.logging:
            # log hparams to tensorbaord (no metrics) -> nested config needs to be flattend and lists need ot be converted to string
            hparams = flatten(OmegaConf.to_container(self.cfg, resolve=True), 'dot')
            hparams = dict(map(lambda kv: (kv[0], json.dumps(kv[1]) if isinstance(kv[1], (list, tuple, dict)) else kv[1]), hparams.items()))
            self.writer.add_hparams(hparams, {})

    def log_config(self):
        if self.cfg.general.logging:
            # add indicator to continue logging if called again
            self.cfg.general['log_path'] = self.log_path
            # save config
            OmegaConf.save(config=self.cfg, f=self.log_path+'/cfg.yml')

    #=================================================================================

    def write_train_step_tensorboard(self, dict, split, step, phase):
        if self.cfg.general.logging:
            for k, v in dict.items():
                self.writer.add_scalars(phase+'/'+k, {split: v}, step)

    def write_eval_tensorboard(self, dict, split, step, phase):
        if self.cfg.general.logging:
            for key in dict.keys():
                for k, v in dict[key].items():
                    self.writer.add_scalars(phase+'/'+key+'/'+k, {split: v}, step)

    #=================================================================================

    def save_checkpoint(self, model):
        if self.cfg.general.logging:
            ckpt = {
                "epoch": model.epoch,
                "model": model.state_dict(),
                "optim": {name: g["optim"].state_dict() for name, g in model.param_groups.items()},
                "sched": {name: g["scheduler"].state_dict() for name, g in model.param_groups.items()},
            }
            torch.save(ckpt, self.log_path + "/ckpt.pth")
