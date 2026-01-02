import torch

from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, root_mean_squared_error

##################################################################################################

def evaluater(model, dataloader, device, split):
    # disable grad and set to eval mode
    model.eval()
    with torch.no_grad():

        # running containers
        all_preds  = {}   # key -> [Tensor...], for classification these are ints, for regression floats
        all_labels = {}   # key -> [Tensor...]
        loss_sum   = {}   # key -> float
        loss_count = {}   # key -> int

        #=================================================================================
        # go through all data in test loader
        for inputs, labels in tqdm(dataloader, desc=f'evaluating {split}set'):
            # move everything to device (inputs is dict of tensors)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = {k: v.to(device) for k, v in labels.items()}

            # forward + per-key losses
            outputs = model(inputs)                 # dict[str, Tensor]
            losses  = model.loss(outputs, labels)   # dict[str, Tensor]

            # for each key decide whether to threshold (classification) or keep raw (regression)
            for k, out in outputs.items():
                # unify shapes to 1-D
                y_pred_raw = out.detach().cpu().view(-1)

                # get label (must exist to evaluate)
                if k not in labels:
                    continue
                y_true = labels[k].detach().cpu().view(-1)

                # classification if the registered loss is BCEWithLogitsLoss
                is_binary = isinstance(model.loss_fns.get(k, None), torch.nn.BCEWithLogitsLoss)

                if is_binary:
                    probs = torch.sigmoid(y_pred_raw)
                    preds = (probs > 0.5).long()
                    all_preds.setdefault(k, []).append(preds)
                    all_labels.setdefault(k, []).append(y_true.long())
                else:
                    # regression: keep float predictions
                    all_preds.setdefault(k, []).append(y_pred_raw.float())
                    all_labels.setdefault(k, []).append(y_true.float())

                # accumulate loss sums (if present)
                if k in losses:
                    loss_sum[k]   = loss_sum.get(k, 0.0) + float(losses[k].item())
                    loss_count[k] = loss_count.get(k, 0) + 1

        #=================================================================================
        # compute metrics per key
        results = {}
        for k in all_labels.keys():
            preds_cat = torch.cat(all_preds[k], dim=0).numpy()
            labs_cat  = torch.cat(all_labels[k], dim=0).numpy()

            loss_avg = float(loss_sum.get(k, 0.0) / max(1, loss_count.get(k, 0)))

            loss_fn = model.loss_fns.get(k, None)
            is_binary = isinstance(loss_fn, torch.nn.BCEWithLogitsLoss)

            if is_binary:
                # ensure integer arrays for sklearn metrics
                y_pred = preds_cat.astype(int)
                y_true = labs_cat.astype(int)

                results[k] = {
                    "accuracy":  float(accuracy_score(y_true, y_pred)),
                    "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                    "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
                    "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
                    "loss":      loss_avg,
                }
            else:
                # regression metrics
                rmse = root_mean_squared_error(labs_cat, preds_cat) if preds_cat.size > 0 else 0.0
                mae = mean_absolute_error(labs_cat, preds_cat) if preds_cat.size > 0 else 0.0

                results[k] = {
                    "rmse":  float(rmse),
                    "mae":   float(mae),
                    "loss":  loss_avg,
                }

        return results