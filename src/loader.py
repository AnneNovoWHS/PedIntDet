import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np

from src.pie_data import PIE
from src.pie_intent import PIEIntent

import torch
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms

from src.utils import seed_worker, get_generator

data_directory = os.path.join('E:/', 'PIE_dataset')


##################################################################################################

class PIEDataset(Dataset):
    def __init__(self, cfg, num_samples, split: str):
        """
        cfg   : config object with .path, .num_frames, .resize, .crop
        split : 'train' or 'val'
        """

        data_opts = {'fstride': 1,
                'sample_type': 'all', 
                'height_rng': [0, float('inf')],
                'squarify_ratio': 0,
                'data_split_type': 'default',  #  kfold, random, default
                'seq_type': 'intention', #  crossing , intention
                'min_track_size': 0, #  discard tracks that are shorter
                'max_size_observe': 15,  # number of observation frames
                'max_size_predict': 5,  # number of prediction frames
                'seq_overlap_rate': 0.5,  # how much consecutive sequences overlap
                'balance': True,  # balance the training and testing samples
                'crop_type': 'context',  # crop 2x size of bbox around the pedestrian
                'crop_mode': 'pad_resize',  # pad with 0s and resize to VGG input
                'encoder_input_type': [],
                'decoder_input_type': ['bbox'],
                'output_type': ['intention_binary']
                }
        

        t = PIEIntent()

        imdb = PIE(data_path=data_directory)

        beh_seq = imdb.generate_data_trajectory_sequence(split, **data_opts)
        beh_seq = imdb.balance_samples_count(beh_seq, label_type='intention_binary')

        
        data_type = {'encoder_input_type': data_opts['encoder_input_type'],
                     'decoder_input_type': data_opts['decoder_input_type'],
                     'output_type': data_opts['output_type']}

        seq_length = data_opts['max_size_observe']
        d = t.get_train_val_data(beh_seq, data_type, seq_length, data_opts['seq_overlap_rate'])
        
        # Retreive image array data in format S, T, H, W, C with S: # sample, T: time step, H: image height, W: image width, C: color channel
        self.img = t.load_images_and_process(d['images'],
                                             d['bboxes'],
                                             d['ped_ids'],
                                             num_samples,
                                             save_path=t.get_path(type_save='data',
                                                                            data_type='features'+'_'+data_opts['crop_type']+'_'+data_opts['crop_mode'], # images    
                                                                            model_name='vgg16_'+'none',
                                                                            data_subset = split))
        

        self.label = d['output'][:, 0] if num_samples==-1 else d['output'][:num_samples, 0]
        self.label = np.squeeze(self.label).astype(float)
        # get all filenames under root and get the file for the chosen split
    #     all_file_names = self.get_all_files(cfg.path)
    #     self.file_names = self.get_split_files(all_file_names, split)

    #     # init transform
    #     self.transform = self.get_transform(cfg)

    #     # get number of frames to extract from every video as well as the number of frames per input clip
    #     self.num_frames = cfg.num_frames
    #     self.clip_time = cfg.num_total_frames

    # #=================================================================================

    # @staticmethod
    # def get_transform(cfg):

    #     # create imagenet transform 
    #     transform = transforms.Compose([
    #         transforms.Resize(cfg.resize),                                          # (T,C,H,W) -> (T,C,H',W')
    #         transforms.CenterCrop(cfg.crop),                                        # (T,C,H',W') -> (T,C,crop,crop)
    #         transforms.Lambda(lambda x: x / 255.0),                                 # uint8 Tensor -> normalized float Tensor
    #         transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]) # IMAGENET1K_V2 mean and std
    #     ])

    #     return transform

    # #=================================================================================

    # @staticmethod
    # def get_split_files(all_files, split):
    #     if split == 'train':    return [str(file) for file in all_files if not 'gc_' in str(file)]
    #     if split == 'val':      return [str(file) for file in all_files if 'gc_' in str(file)]

    # #=================================================================================

    # @staticmethod
    # def get_all_files(root):
    #     return list(Path(os.path.join(root, 'processed')).rglob("*.mp4"))

    # #=================================================================================

    # def create_concepts(self, annotations):
    #     # get blinks and yawning intervalls from data
    #     blinks = torch.tensor(annotations['blinks'], dtype=torch.float32)
    #     yawns = torch.tensor(annotations['yawns'], dtype=torch.float32)

    #     # Yawns
    #     yawning = torch.tensor(1.0 if len(yawns)>0 else 0.0) 

    #     # PERCLOSE
    #     if len(blinks)>0:   perclose = (blinks[:, 1] - blinks[:, 0]).sum()/self.clip_time
    #     else:               perclose = torch.tensor(0.0)

    #     # blink rate
    #     if len(blinks)>0:   blink_rate = torch.tensor(len(blinks))/self.clip_time 
    #     else:               blink_rate = torch.tensor(0.0)

    #     # blink variation
    #     if len(blinks) >= 3:
    #         ibi = blinks[1:, 0] - blinks[:-1, 1]    # open time between blinks
    #         mean_ibi = ibi.mean()
    #         blink_variation = (ibi.std(unbiased=False) / mean_ibi) if mean_ibi > 0 else torch.tensor(0.0)  # population std
    #     else: blink_variation = torch.tensor(0.0)

    #     return {"yawning": yawning,          
    #             "perclose": perclose,     
    #             "blink_rate": blink_rate,
    #             "blink_variability": blink_variation}

    # #=================================================================================

    def __len__(self):
        return len(self.label)
    
    #=================================================================================

    def __getitem__(self, idx):

        return {"frames": self.img[idx]}, {"pred": self.label[idx]}
        
##################################################################################################

def get_loader(cfg, num_samples):
     
    # datasets
    if cfg.data.name == 'PIE':
        trainset = PIEDataset(cfg.data, num_samples, split='train')
        valset = PIEDataset(cfg.data, num_samples, split='val')

    # dataloader
    trainloader = DataLoader(trainset, 
                             batch_size=cfg.model.training.batch_size, 
                             num_workers=2,#os.cpu_count(), 
                             shuffle=True,
                             worker_init_fn=seed_worker,
                             generator=get_generator(), 
                            )
    
    valloader = DataLoader(valset, 
                           batch_size=cfg.model.training.batch_size, 
                           num_workers=2,#os.cpu_count(), 
                           worker_init_fn=seed_worker,
                        )

    return trainloader, valloader
