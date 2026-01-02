from abc import ABC, abstractmethod
from typing import Dict, Any
from copy import deepcopy

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LinearLR
from torchvision.models import get_model, get_model_weights

from src.utils import get_optimizer

##################################################################################################

class VideoClassifier(nn.Module, ABC):
    def __init__(self, cfg):
        super().__init__()
        self.cfg   = cfg
        self.epoch = -1

        # to be filled by subclasses
        self.param_groups: Dict[str, Dict[str, Any]] = {}   # {name: {params, lr, freeze_until_epoch}}

        # default BCE for "pred"
        self.loss_fns: Dict[str, nn.Module] = {"pred": nn.BCEWithLogitsLoss()}

    @abstractmethod
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        ...

    #=================================================================================

    def loss(self, outputs: Dict[str, torch.Tensor], labels: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out = {}
        for k, crit in self.loss_fns.items():
            if k in outputs and k in labels:
                out[k] = crit(outputs[k], labels[k])
        return out
    
    #=================================================================================

    def train_step(self, outputs: Dict[str, torch.Tensor], labels: Dict[str, torch.Tensor]) -> Dict[str, Any]:

        # calc losses
        loss_dict = self.loss(outputs, labels)
        step_report = {f"loss_{k}": v.detach().item() for k, v in loss_dict.items()}

        # calc total loss
        total = torch.stack([v for _, v in loss_dict.items()]).sum()
        step_report["loss_total"] = total.detach().item()

        # backward + grad cliping
        total.backward()
        norm = torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        step_report["norm"] = norm.detach().item()

        # updatee scheduler and lr for each param group
        for name, g in self.param_groups.items():
            if not self.epoch < int(g.get("freeze_until_epoch", 0)):
                g["optim"].step()
                g["optim"].zero_grad()
                g["scheduler"].step()
                step_report[f"lr_{name}"] = g["scheduler"].get_last_lr()[0]

        # return report
        return step_report
    
    #=================================================================================

    def on_epoch_start(self) -> None:

        # update epoch and put model into train mode
        self.epoch += 1
        self.train()
        
        # check unfreeze
        for _, g in self.param_groups.items():
            if self.epoch == int(g.get("freeze_until_epoch", 0)):
                for p in g["params"]:
                    p.requires_grad = True
                    g["optim"].zero_grad()
                    
    #=================================================================================

    def init_optimizer_and_scheduler(self, num_batches: int) -> None:

        # go through each param group and create an optimizer and scheduler
        for _, g in self.param_groups.items():
            g["optim"] = get_optimizer(g["params"], weight_decay=self.cfg.training.weight_decay, lr=float(g["lr"]))
            warm = int(g.get("freeze_until_epoch", 0))

            # create learning rate schedulers
            eff_epochs = max(1, self.cfg.training.epochs - warm) if warm > 0 else self.cfg.training.epochs
            total_iters = int(eff_epochs * num_batches)
            g["scheduler"] = LinearLR(g["optim"], start_factor=1.0, end_factor=0.0, total_iters=total_iters)

            # freeze param group
            if warm > 0:
                for p in g["params"]:
                    p.requires_grad = False

##################################################################################################

class ResNetVideoClassifier(VideoClassifier):
    def __init__(self, cfg):
        super().__init__(cfg)

        # ----- backbone -----
        backbone = get_model(
            name=cfg.variant,
            weights=(get_model_weights(cfg.variant).DEFAULT if getattr(cfg, "pretrained", False) else None)
        )
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])

        # ----- temporal attention pooling -----
        self.attn = nn.Sequential(
            nn.Linear(backbone.fc.in_features, cfg.params.pool_hidden_size),
            nn.Tanh(),
            nn.Linear(cfg.params.pool_hidden_size, 1),
        )

        # ----- classification head -----
        self.cls_head = nn.Sequential(
            nn.Linear(backbone.fc.in_features, cfg.params.cls_hidden_size),
            nn.BatchNorm1d(cfg.params.cls_hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.params.dropout),
            nn.Linear(cfg.params.cls_hidden_size, 1),
        )

        # ----- param groups -----
        self.param_groups = {
            "backbone": {
                "params": list(self.feature_extractor.parameters()),
                "lr": cfg.training.lr_backbone,
                "freeze_until_epoch": cfg.training.warmup_epochs,  # freeze during warmup
            },
            "head": {
                "params": list(self.attn.parameters()) + list(self.cls_head.parameters()),
                "lr": cfg.training.lr_head,
                "freeze_until_epoch": 0,
            },
        }

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, T, C, H, W = x['frames'].shape

        # frame-wise feature extraction with 2D backbone
        x = x['frames'].view(B * T, C, H, W)    # (B*T, C, H, W)
        x = self.feature_extractor(x)           # (B*T, F, 1, 1)
        x = x.view(B, T, -1)                    # (B, T, F)

        # attention pooling over time
        scores = self.attn(x)               # (B, T, 1)
        weights = torch.softmax(scores, dim=1)
        pooled = (x * weights).sum(dim=1)   # (B, F)

        # final logits
        logits = self.cls_head(pooled).squeeze(-1)  # (B,)
        return {"pred": logits}

##################################################################################################

class ResNet3DVideoClassifier(ResNetVideoClassifier):
    def __init__(self, cfg):
        # build the 2D version first to inherit loss_fns and head config
        super().__init__(cfg)

        # ----- 3D backbone (r3d_18, mc3_18, r2plus1d_18) -----
        backbone = get_model(
            name=cfg.variant,
            weights=(get_model_weights(cfg.variant).DEFAULT if getattr(cfg, "pretrained", False) else None)
        )

        # keep everything up through avgpool, drop the final fc
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])

        # attention is not used for 3D backbones
        if hasattr(self, "attn"):
            del self.attn

        # ----- param groups (no attention params here) -----
        self.param_groups = {
            "backbone": {
                "params": list(self.feature_extractor.parameters()),
                "lr": cfg.training.lr_backbone,
                "freeze_until_epoch": cfg.training.warmup_epochs,
            },
            "head": {
                "params": list(self.cls_head.parameters()),
                "lr": cfg.training.lr_head,
                "freeze_until_epoch": 0,
            },
        }

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # x: (B, T, C, H, W) -> 3D models expect (B, C, T, H, W)
        x = x['frames'].permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
        x = self.feature_extractor(x)           # (B, F, 1, 1, 1)
        x = x.flatten(1)                        # (B, F)
        logits = self.cls_head(x).squeeze(-1)   # (B,)
        return {"pred": logits}

##################################################################################################

class ResNetCBMVideoClassifier(VideoClassifier):
    def __init__(self, cfg):
        super().__init__(cfg)

        # concepts
        self.concepts = cfg.params.concepts
        num_concepts = len(self.concepts)

        # ----- backbone (2D ResNet-style, frame-wise features) -----
        backbone = get_model(
            name=cfg.variant,
            weights=(get_model_weights(cfg.variant).DEFAULT if getattr(cfg, "pretrained", False) else None)
        )
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])

        # ----- temporal attention pooling over frames -----
        self.attn = nn.Sequential(
            nn.Linear(backbone.fc.in_features, cfg.params.pool_hidden_size),
            nn.Tanh(),
            nn.Linear(cfg.params.pool_hidden_size, 1),
        )

        # ----- concept heads  -----
        self.concept_heads = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(backbone.fc.in_features, cfg.params.concept_pred_size),
                nn.BatchNorm1d(cfg.params.concept_pred_size),
                nn.ReLU(inplace=True),
                nn.Dropout(cfg.params.dropout),
                nn.Linear(cfg.params.concept_pred_size, 1),
            ) for name in self.concepts
        })

        # ----- final prediction head -----
        self.pred_head = nn.Sequential(
            nn.Linear(num_concepts, cfg.params.head_size),
            nn.BatchNorm1d(cfg.params.head_size),
            nn.ReLU(inplace=True),

            nn.Linear(cfg.params.head_size, cfg.params.head_size // 2),
            nn.BatchNorm1d(cfg.params.head_size // 2),
            nn.ReLU(inplace=True),

            nn.Dropout(cfg.params.dropout),
            nn.Linear(cfg.params.head_size // 2, 1),
        )

        # ----- param groups -----
        self.param_groups = {
            "concept": {
                "params": list(self.feature_extractor.parameters())
                        + list(self.attn.parameters())
                        + list(self.concept_heads.parameters()),
                "lr": cfg.training.lr_concept_predictor,
                "freeze_until_epoch": cfg.training.concept_freeze_epochs,
            },
            "head": {
                "params": list(self.pred_head.parameters()),
                "lr": cfg.training.lr_head,
                "freeze_until_epoch": cfg.training.predictor_freeze_epochs, 
            },
        }

        # ----- losses -----
        self.loss_fns = {c: nn.MSELoss() for c in self.concepts}
        if "yawning" in self.loss_fns: self.loss_fns["yawning"] = nn.BCEWithLogitsLoss() # override binary concepts to BCE
        self.loss_fns["pred"] = nn.BCEWithLogitsLoss()

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # x: (B, T, C, H, W) -> frame-wise 2D backbone
        B, T, C, H, W = x['frames'].shape
        x = x['frames'].view(B * T, C, H, W)    # (B*T, C, H, W)
        x = self.feature_extractor(x)           # (B*T, F, 1, 1)
        x = x.view(B, T, -1)                    # (B, T, F)

        scores = self.attn(x)               # (B, T, 1)
        weights = torch.softmax(scores, dim=1)
        pooled = (x * weights).sum(dim=1)   # (B, F)

        # concept logits dict
        outs = {}
        cat = []
        for name, head in self.concept_heads.items():
            logit = head(pooled).squeeze(-1) 
            outs[name] = logit
            cat.append(logit.unsqueeze(1))

        concepts_cat = torch.cat(cat, dim=1)                        # (B, num_concepts)
        outs["pred"] = self.pred_head(concepts_cat).squeeze(-1)
        return outs

    #=================================================================================

    def train_step(self, outputs: Dict[str, torch.Tensor], labels: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        # calc losses
        loss_dict = self.loss(outputs, labels)

        # select losses to include based on freeze state
        losses = {f"loss_{k}": v for k, v in loss_dict.items()
                    if self.epoch >= self.param_groups.get("head" if k == "pred" else "concept", {}).get("freeze_until_epoch", 0)}
                
        # calc total loss 
        losses["loss_total"] = torch.stack(list(losses.values())).sum()
        
        # backward + grad clipping
        losses["loss_total"].backward()
        norm = torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)

        # add losses and norm to step report
        step_report = {k: v.detach().item() for k, v in losses.items()} 
        step_report["norm"] = float(norm)

        # update scheduler and lr for each param group
        for name, g in self.param_groups.items():
            if not self.epoch < int(g.get("freeze_until_epoch", 0)):
                g["optim"].step()
                g["optim"].zero_grad()
                g["scheduler"].step()
                step_report[f"lr_{name}"] = g["scheduler"].get_last_lr()[0]

        # return report
        return step_report

##################################################################################################

class CBMHeadOnly(ResNetCBMVideoClassifier):
    def __init__(self, cfg):
        # build a temp CBM to get the exact head architecture
        tmp = ResNetCBMVideoClassifier(cfg)
        # init only the VideoClassifier base on *this* instance
        VideoClassifier.__init__(self, cfg)

        self.concepts = tmp.concepts
        self.pred_head = deepcopy(tmp.pred_head)
        del tmp  # free memory

        # single param group (head only)
        self.param_groups = {
            "head": {
                "params": list(self.pred_head.parameters()),
                "lr": float(cfg.training.lr_head),
                "freeze_until_epoch": 0,
            }
        }
        # base already has BCE("pred"); keep it
        self.loss_fns = {"pred": nn.BCEWithLogitsLoss()}

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # x["concepts"]: (B, len(self.concepts))
        return {"pred": self.pred_head(x["concepts"]).squeeze(-1)}

##################################################################################################

# map model to name
MODEL_REGISTRY = {
    "resnet": ResNetVideoClassifier,
    "resnet3d": ResNet3DVideoClassifier,
    "resnetCBM": ResNetCBMVideoClassifier,
    "resnetCBMhead": CBMHeadOnly,
}

#=================================================================================

def build_model(cfg, trainloader, device, checkpoint=None):

    # load model
    ctor = MODEL_REGISTRY[cfg.model.name]

    # throw error and show available models
    if ctor is None:
        raise KeyError(
            f"Unknown model {cfg.model.name}."
            f"Available: {list(MODEL_REGISTRY)}"
        )
    
    # create model
    model = ctor(cfg.model)
    model.to(device)

    # init model
    model.init_optimizer_and_scheduler(num_batches=len(trainloader))
    
    # if model should be loaded from a checkpoint, load all params 
    if checkpoint != None:
        ckpt = torch.load(checkpoint+'/ckpt.pth', map_location=device)

        # load weights and last epoch
        model.load_state_dict(ckpt["model"])
        model.epoch = ckpt.get("epoch", -1)

        # restore per-group optimizer states
        if "optim" in ckpt:
            for name, sd in ckpt["optim"].items():
                if name in model.param_groups:
                    model.param_groups[name]["optim"].load_state_dict(sd)

        # restore per-group scheduler states
        if "sched" in ckpt:
            for name, sd in ckpt["sched"].items():
                if name in model.param_groups:
                    model.param_groups[name]["scheduler"].load_state_dict(sd)

    return model
