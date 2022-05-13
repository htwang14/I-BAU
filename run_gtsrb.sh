#!/bin/bash

# python run.py --ds gtsrb --target 1 --pattern badnet_grid --lr 1e-4 --model_path target1-ratio0.1_e200-b256-adam-lr0.001-wd0.0005-cos-holdout0.05
python run.py --ds gtsrb --target 1 --pattern badnet_sq --lr 1e-4 --model_path target1-ratio0.1_e200-b256-adam-lr0.001-wd0.0005-cos-holdout0.05
python run.py --ds gtsrb --target 1 --pattern blend --lr 1e-4 --model_path target1-ratio0.1_e200-b256-adam-lr0.001-wd0.0005-cos-holdout0.05
python run.py --ds gtsrb --target 1 --pattern l0_inv --lr 1e-4 --model_path target1-ratio0.1_e200-b256-adam-lr0.001-wd0.0005-cos-holdout0.05
python run.py --ds gtsrb --target 1 --pattern l2_inv --lr 1e-4 --model_path target1-ratio0.1_e200-b256-adam-lr0.001-wd0.0005-cos-holdout0.05
python run.py --ds gtsrb --target 1 --pattern sig --lr 1e-4 --model_path target1-ratio0.1_e200-b256-adam-lr0.001-wd0.0005-cos-holdout0.05
python run.py --ds gtsrb --target 1 --pattern trojan_3x3 --lr 1e-4 --model_path target1-ratio0.1_e200-b256-adam-lr0.001-wd0.0005-cos-holdout0.05
python run.py --ds gtsrb --target 1 --pattern trojan_wm --lr 1e-4 --model_path target1-ratio0.1_e200-b256-adam-lr0.001-wd0.0005-cos-holdout0.05
python run.py --ds gtsrb --target 1 --pattern smooth --lr 1e-4 --model_path target1-ratio0.1_e200-b256-adam-lr0.001-wd0.0005-cos-holdout0.05