# Suppressing Bird-Induced False Drone Alarms Using YOLO Segmentation and Mask-Derived Geometry

This repository contains the code and experimental pipeline for a lightweight drone detection framework under bird interference. The project is built around a YOLO-based instance segmentation baseline and a bird-specific geometric post-processing module designed to suppress bird-induced false drone alarms.

## Overview

Detecting small drones in visible-spectrum surveillance imagery is challenging because drones often appear as tiny, low-contrast objects and are easily confused with birds. In many challenge-style drone-versus-bird benchmarks, the task is primarily formulated as **drone detection under bird interference**, rather than balanced supervised drone/bird classification.

This project follows that setting. The main idea is:

1. Use **YOLO11n-seg** as a lightweight instance segmentation backbone.
2. Predict aerial-object bounding boxes and instance masks.
3. Extract **mask-derived geometric descriptors** from predicted drone instances.
4. Apply a **bird-specific post-processing filter** to suppress bird-induced false drone alarms.

## Main Contributions

- Lightweight **YOLO11n-seg** baseline for drone detection.
- Mask-derived geometric feature extraction:
  - area
  - aspect ratio
  - extent
  - solidity
  - circularity
  - compactness
  - rectangularity
  - eccentricity
  - vertex count
  - bounding-box width/height
- Rule-based and learned post-processing filters.
- Bird-specific **Random Forest** filter for suppressing bird-induced false drone alarms.

## Project Structure
``` text
.
├── data/
│   ├── dronebird_seg_raw/
│   ├── dronebird_seg_clean_v2/
│   └── dronebird_det_clean/
├── runs/
├── scripts/
├── tools/
├── *.tex
└── README.md
```

## Key folders

- data/: datasets and processed label splits
- tools/: Python utilities for data conversion, feature extraction, and filtering
- scripts/: PowerShell scripts for training and evaluation
- runs/: model outputs, checkpoints, metrics, and intermediate results

## Datasets

This project uses public drone-versus-bird datasets, especially the YOLO-based segmented drone-vs-bird dataset, which provides manually annotated segmentation masks for drones and birds.

**Dataset notes**

The raw segmented dataset may contain mixed annotation formats:

- _standard YOLO bounding boxes_
- _polygon-style segmentation labels_

Therefore, preprocessing is required before training.

**Data Preprocessing**

Two processed datasets are created:

**1. Segmentation dataset**

Used for YOLO11n-seg.
polygon annotations are preserved
box-only annotations are converted to rectangular polygons
Output:
data/dronebird_seg_clean_v2/

**2. Detection dataset**

Used for box-only baseline training.

box annotations are preserved

polygon annotations are converted to bounding boxes

Output:

data/dronebird_det_clean/

## Baseline Models
- YOLO11n: Box-only drone detection baseline.
- YOLO11n-seg: Instance segmentation baseline used as the main backbone of the proposed method.

## Proposed Method

The proposed pipeline consists of two stages:

**Stage 1: YOLO11n-seg baseline**

The model predicts:

- bounding boxes
- confidence scores
- instance masks

**Stage 2: Bird-specific post-processing**

For each predicted drone instance, the pipeline extracts mask-derived geometric descriptors and applies a lightweight bird-specific filter. 
The best-performing variant in our experiments is a Random Forest post-processing filter trained to identify bird-induced false drone alarms.

## Experimental Pipeline
**A. Prepare data**

- convert raw mixed-format labels
- create detection and segmentation datasets

**B. Train baselines**
- train YOLO11n on detection labels
- train YOLO11n-seg on segmentation labels

**C. Post-processing experiments**

- rule-based shape filters

- learned keep/reject geometric filters

- bird-specific learned filters

**D. Evaluate**

Metrics include:

- box precision / recall / mAP
- mask precision / recall / mAP
- drone precision / recall / F1
- bird-induced false drone alarms

## Main Scripts
Data preparation

- scripts/00_prepare_data.ps1

Baseline training

- scripts/10_train_baseline_det.ps1

- scripts/11_train_baseline_seg.ps1

Full baseline pipeline

- scripts/run_baselines.ps1

Rule-based and learned filters

scripts/20_eval_proposed_false_alarm_filter.ps1

scripts/21_eval_proposed_false_alarm_filter_v2.ps1

Bird-specific valid-to-test pipeline

scripts/22_run_bird_fp_valid_to_test.ps1

## Main Python Tools
- Dataset conversion
  - tools/convert_raw_to_yolodet.py
  - tools/convert_mixed_to_yoloseg_v2.py
- Rule-based filtering
  - tools/eval_false_alarm_shape_filter.py
  - tools/eval_false_alarm_shape_filter_v2.py
- Learned geometric filtering
  - tools/build_shape_training_data.py
  - tools/train_shape_classifier.py
  - tools/eval_false_alarm_shape_filter_ml.py
- Bird-specific learned filtering
  - tools/build_bird_fp_training_data.py
  - tools/train_bird_fp_classifier.py
  - tools/eval_false_alarm_shape_filter_bird_ml.py

## Example Commands
Train segmentation baseline
powershell -ExecutionPolicy Bypass -File D:\chim\scripts\11_train_baseline_seg.ps1 -Epochs 40 -Batch 12 -Workers 4 -Imgsz 640 -RunName baseline_seg_rgb
Evaluate bird-specific Random Forest filter
python D:\chim\tools\eval_false_alarm_shape_filter_bird_ml.py `
  --model "D:\chim\runs\baseline_seg_rgb\weights\best.pt" `
  --classifier "D:\chim\runs\bird_fp_ml\bird_fp_filter_rf.joblib" `
  --data "D:\chim\data\dronebird_seg_clean_v2\data.yaml" `
  --split test `
  --outdir "D:\chim\runs\bird_fp_ml_eval_rf" `
  --imgsz 640 `
  --conf 0.25 `
  --iou-nms 0.5 `
  --iou-match 0.5 `
  --device 0
## Representative Results

The YOLO11n-seg baseline achieves strong drone detection performance for a lightweight model. In our exploratory experiments, the best bird-specific Random Forest filter substantially reduced bird-induced false drone alarms while improving drone precision and overall F1-score relative to the raw segmentation baseline.
Important Note on Evaluation Protocol: The project also explores bird-specific learned filtering under an exploratory protocol. In the current dataset split, the validation set does not contain sufficient positive bird-induced false-drone samples to support a fully clean validation-to-test training protocol for the bird-specific learner.

Therefore:

- the baseline detector results are standard
- the strongest bird-specific filtering results should be interpreted as proof-of-concept exploratory results

## Environment

Recommended environment:
- Python 3.11
- PyTorch
- Ultralytics
- OpenCV
- pandas
- scikit-learn
- joblib

Install Dependencies
- pip install ultralytics opencv-python pandas scikit-learn joblib
Notes

Datasets, model weights, and training outputs are typically not included in the repository.

Please prepare the required dataset folders manually before running the scripts.

Some scripts assume Windows PowerShell paths and local folder organization.

## Citation

If you use this repository in your research, please cite the corresponding paper/report when available.

## Contact

Maintained by congpx.



