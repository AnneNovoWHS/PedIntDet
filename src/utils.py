import os
import torch
import numpy as np
import random

##################################################################################################################

def setup_env(seed):

    # set python, numpy, torch random seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # when running on the CuDNN backend
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # set precision
    torch.set_float32_matmul_precision('high')      

##################################################################################################################

# funciton to seed the dataloader worker
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# get a generator for the dataloader for reprducability
def get_generator():
    g = torch.Generator()
    g.manual_seed(0)
    return g

##################################################################################################################

# function adapted from https://github.com/karpathy/nanoGPT/
def get_optimizer(params, weight_decay, lr):
        
        # filter out params that do not require grad
        params = [p for p in params if p.requires_grad]

        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for p in params if p.dim() >= 2]
        nodecay_params = [p for p in params if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]

        # Create AdamW optimizer 
        return torch.optim.AdamW(optim_groups, lr=lr, fused=True)
