#!/bin/bash

eval $(conda shell.bash hook)
conda activate mmdetection

python tools/get_flops.py configs/cascade_mask_rcnn_r101_fpn_1x.py
