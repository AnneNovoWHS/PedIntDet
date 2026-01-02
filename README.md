# Driver Drowsiness Detection

A video-based drowsiness detection framework using both 2D ResNet and 3D ResNet backbones, currently supporting the YaWDD and DMD dataset. A concept bottleneck variant (available only for DMD) provides interpretable intermediate predictions.

## 📋 Overview

* **Models**:
    * ResNet-18 + attention pooling over time dimension
    * 3D 18-Layer Mixed Convolution Network (mc3_18)
    * ResNet-18 Concept Bottleneck Model (CBM) (DMD only) that first predicts concepts (yawning, PERCLOSE, blink rate, blink variability) and then classifies drowsiness from those concepts.
* **Dataset**:
    * YaWDD (Mirror subset)
    * Driver Monitoring Dataset (drowsiness data (s5) + in car recordings of gaze data (s6) as control)

## ⚙️ Requirements

**Python Version**: 3.10.18

Install all the dependencies in an enviroment.

    pip install -r requirements.txt

## 💾 Data

### YaWDD

Download and unpack the [YaWDD dataset](https://ieee-dataport.org/open-access/yawdd-yawning-detection-dataset). Before training or evaluating, change the path that points to the dataset in the respective config files.

### DMD

Download the drowsiness and gaze dataset from the [DMD dataset](https://dmd.vicomtech.org/) website and unpack them into the same folder.  
Change the path to the folder in the preprocessing script, then run it.

    python preprocess_dmd.py

Before training or evaluating, also change the path that points to the DMD dataset in the respective config files.

### Concept annotations for the CBM

The Concept Bottleneck Model uses four interpretable concepts derived from the DMD annotations:

* **Yawning**: whether any yawn is present in the clip.
* **PERCLOSE**: proportion of time the eyes are closed.
* **Blink rate**: blinks per frame over the clip.
* **Blink variability**: variation in inter-blink intervals.

These values are computed automatically inside the DMD dataloader from the preprocessed `*_ann.pt` files; no extra labeling is required.

## 🚀 Training

Train any model using the appropriate config file:

    python train.py --config configs/<config>.yml

Available configs:

* **YaWDD**
  * `resnet18_yawdd.yml` (ResNet-18 + attention pooling)
  * `resnet3d_yawdd.yml` (mc3_18)
* **DMD**
  * `resnet18_dmd.yml` (ResNet-18 + attention pooling)
  * **Concept Bottleneck**
    * `resnet18_dmd_cbm.yml` – full CBM (concept predictors + drowsiness head)
    * `resnet18_dmd_cbm_head_only.yml` – train only the CBM head on ground-truth concepts (oracle upper bound)

> Use `--checkpoint logs/<model>/<timestamp>` instead of `--config` to continue from a checkpoint. The config will be loaded from the logs in the given checkpoint folder.

## 📊 Evaluation

Evaluate any trained model by pointing to its checkpoint directory (e.g., `logs/resnet/2025-08-04_22-47`):

    python evaluate.py --checkpoint logs/<model>/<timestamp>

> Reports: Loss, Accuracy, Precision, Recall, F1-score

## 🔍 Hyperparameter Search (Work in Progress)

To staart a hyperparameter search with optuna run

    python hyperparam_search.py --config <some default config>

> Replace the config and adjust the hyperparameters and their range in the *hyperparam_search.py* scripts

## 📈 Results

### YaWDD

| Model                             | F1-score | Precision | Recall | Accuracy |
| --------------------------------- | -------- | --------- | ------ | -------- |
| ResNet-18 + attention pooling     | 0.960    | 0.923     | 1.000  | 0.971    |
| 3D 18-Layer Mixed Conv Net (mc_18)| 0.979    | 0.958     | 1.000  | 0.985    |

> *Metrics are reported on a 20% stratified validation split of the Mirror subset of the YaWDD dataset.*

### DMD

| Model                         | F1-score | Precision | Recall | Accuracy |
| ----------------------------- | -------- | --------- | ------ | -------- |
| ResNet-18 + attention pooling | 0.814    | 0.782     | 0.850  | 0.822    |

| Model / Output             | Metric(s)      | Score        |
| --------------------------- | -------------- | ------------ |
| CBM – blink rate           | MAE / RMSE     | 0.049 / 0.068|
| CBM – blink variability    | MAE / RMSE     | 0.335 / 0.386|
| CBM – PERCLOS              | MAE / RMSE     | 0.327 / 0.145|
| CBM – yawning              | Accuracy / F1  | 0.982 / 0.847|
| Head-only (GT concepts)    | Accuracy / F1  | 0.781 / 0.736|
| ResNet-18-CBM            | Accuracy / F1  | 0.695 / 0.776|

> *Prediction metrics are reported on Group C of the DMD data, with in car recordings of gaze data (s6) as control class.*
