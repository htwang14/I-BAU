#!/bin/bash

python run.py --ds cifar10 --target 0 --pattern badnet_grid --model_path target0-ratio0.1_e200-b256-adam-lr0.001-wd0.0005-cos-holdout0.05
python run.py --ds cifar10 --target 0 --pattern badnet_sq --lr 1e-4 --model_path target0-ratio0.1_e200-b256-adam-lr0.001-wd0.0005-cos-holdout0.05
python run.py --ds cifar10 --target 0 --pattern blend --model_path target0-ratio0.1_e200-b256-adam-lr0.001-wd0.0005-cos-holdout0.05
python run.py --ds cifar10 --target 0 --pattern l0_inv --model_path target0-ratio0.1_e200-b256-adam-lr0.001-wd0.0005-cos-holdout0.05
python run.py --ds cifar10 --target 0 --pattern l2_inv --lr 5e-4 --model_path target0-ratio0.1_e200-b256-adam-lr0.001-wd0.0005-cos-holdout0.05
python run.py --ds cifar10 --target 0 --pattern sig --lr 1e-4 --model_path target0-ratio0.1_e200-b256-adam-lr0.001-wd0.0005-cos-holdout0.05
python run.py --ds cifar10 --target 0 --pattern trojan_3x3 --lr 1e-4 --model_path target0-ratio0.1_e200-b256-adam-lr0.001-wd0.0005-cos-holdout0.05
python run.py --ds cifar10 --target 0 --pattern trojan_wm --lr 1e-4 --model_path target0-ratio0.1_e200-b256-adam-lr0.001-wd0.0005-cos-holdout0.05
python run.py --ds cifar10 --target 0 --pattern smooth --lr 5e-4 --model_path target0-ratio0.1_e200-b256-adam-lr0.001-wd0.0005-cos-holdout0.05