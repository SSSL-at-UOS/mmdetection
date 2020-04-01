import os



os.system('./tools/dist_test.sh project_work_dirs/multiple_damage_detection/configs/val_config/mask_rcnn_r50_fpn_1x.py project_work_dirs/multiple_damage_detection/work_dirs/mask_rcnn_r50_fpn_1x/epoch_19.pth 3 --out results.pkl --eval bbox segm')
