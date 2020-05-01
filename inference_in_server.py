
# import required libraries

from mmdet.apis import init_detector, inference_detector

import numpy as np
import mmcv
import cv2
import os 
import glob
import sys
import datetime
import argparse

from skimage.measure import  label, regionprops_table, find_contours
from skimage.morphology import medial_axis, skeletonize
import json

sys.path.append('..')
from shm_tools.shm_utils import imread, imwrite, inference_detector_sliding_window, connect_cracks, remove_cracks


# read inference configuration json file 
parser = argparse.ArgumentParser()
parser.add_argument("inference_config")
args = parser.parse_args()

with open(args.inference_config) as f:
    inference_config = json.load(f)
    
# Set color mask
color_mask = np.array([[255, 0, 0],
                       [0, 255, 0],
                       [0, 255, 255],
                       [255, 0, 255],
                      ], dtype=np.uint8)

# set img path list 
img_folder =inference_config['anlyTargetPath']
img_path_list = glob.glob(os.path.join(img_folder, '*.jpg')) + glob.glob(os.path.join(img_folder, '*.JPG'))
img_path_list = sorted(img_path_list)

# set result path?
result_save_folder = inference_config["anlyResultPath"]

damage_detection_output = {}
damage_detection_output["ptanFcltsCd"] = inference_config["ptanFcltsCd"]
damage_detection_output["anlyDataId"] = inference_config["anlyDataId" ]
damage_detection_output["pctrList"] = {}
damage_detection_output["pctrList"]["dfctCnt"] = str(len(img_path_list))

# loop through imgs in the list 
# detection and post processing for crack 

# Load trained damage detection model here
if inference_config["ptanFcltsCd"] == "TN" :
    config = 'project_work_dirs/express_cor/configs/cascade_mask_rcnn_x101_32x4d_dcn_fpn_carafe_1x_road_crack_200423.py'
    checkpoint = 'project_work_dirs/express_cor/work_dirs/cascade_mask_rcnn_x101_32x4d_dcn_fpn_carafe_1x_road_crack_200423/epoch_26.pth'

elif inference_config["ptanFcltsCd"] == "BR" :
    config = 'project_work_dirs/express_cor/configs/cascade_mask_rcnn_x101_32x4d_dcn_fpn_carafe_1x_road_crack_200423.py'
    checkpoint = 'project_work_dirs/express_cor/work_dirs/cascade_mask_rcnn_x101_32x4d_dcn_fpn_carafe_1x_road_crack_200423/epoch_26.pth'

elif inference_config["ptanFcltsCd"] == "BP" :
    config = 'project_work_dirs/express_cor/configs/cascade_mask_rcnn_x101_32x4d_dcn_fpn_carafe_1x_road_crack_200423.py'
    checkpoint = 'project_work_dirs/express_cor/work_dirs/cascade_mask_rcnn_x101_32x4d_dcn_fpn_carafe_1x_road_crack_200423/epoch_26.pth'

# load the model on GPU
device = 'cuda:0'
model = init_detector(config, checkpoint, device=device)

# inference for crack
for num, img_path in enumerate(img_path_list) :

    pctrList = {}

    pctrList["strtDttm"] = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    pctrList["anlyPctrId"] = num+1
    pctrList["anlyPctrNm"] = os.path.basename(img_path)
    
    _, mask_output = inference_detector_sliding_window(model, img_path, color_mask[0], score_thr = 0.1, window_size = 1024, overlap_ratio = 0.3)

    if np.sum(mask_output) > 0:

        mask_output = connect_cracks(mask_output)
        mask_output = connect_cracks(mask_output)
        mask_output = remove_cracks(mask_output)

        skel, distance = medial_axis(mask_output, return_distance=True)
        dist_on_skel = distance * skel

        labels = label(mask_output)

        damage_region_prop = regionprops_table(labels,  properties=('label', 'centroid'))

        pctrList['cordList'] = []

        for label_num in range(np.max(labels)) :

            a_label = labels == label_num+1

            contours, _ = cv2.findContours(a_label.astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

            imprXcord = str()
            imprYcord = str()

            for countour in contours[0]:
                imprYcord = imprYcord + str(countour[0][0]) + ','
                imprXcord = imprXcord + str(countour[0][1]) + ','

            imprYcord = imprYcord[:-1]
            imprXcord = imprXcord[:-1]

            imprCnterCord = str(damage_region_prop['centroid-0'][label_num])+ ',' + str(damage_region_prop['centroid-1'][label_num])

            dist_label = dist_on_skel[a_label]

            cordList = {}

            cordList["cordTypeCd"]  = "2"
            cordList["imprXcord"]  = imprXcord
            cordList["imprYcord"]  = imprYcord
            cordList["imprCnterCord"]  = imprCnterCord
            cordList["imprTypeCd"] = "01"
            cordList["imprWdth"] = str(dist_label[np.nonzero(dist_label)].mean()*0.2)
            cordList["imprLnth"] = str(np.sum(skel[a_label])*0.2)
            cordList["imprBrdthVal"] = ""
            cordList["imprQntt"] = ""
            pctrList['cordList'].append(cordList)

        pctrList["endDttm"] = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        damage_detection_output["pctrList"].append(pctrList)
        

    elif np.sum(mask_output) == 0:
        print(img_path + ' has no crack detection result')

# Load trained damage detection model here
if inference_config["ptanFcltsCd"] == "TN":
    config = 'project_work_dirs/express_cor/configs/cascade_mask_rcnn_x101_32x4d_dcn_fpn_carafe_1x_road_crack_200423.py'
    checkpoint = 'project_work_dirs/express_cor/work_dirs/cascade_mask_rcnn_x101_32x4d_dcn_fpn_carafe_1x_road_crack_200423/epoch_26.pth'

elif inference_config["ptanFcltsCd"] == "BR":
    config = 'project_work_dirs/express_cor/configs/cascade_mask_rcnn_x101_32x4d_dcn_fpn_carafe_1x_road_crack_200423.py'
    checkpoint = 'project_work_dirs/express_cor/work_dirs/cascade_mask_rcnn_x101_32x4d_dcn_fpn_carafe_1x_road_crack_200423/epoch_26.pth'

elif inference_config["ptanFcltsCd"] == "BP":
    config = 'project_work_dirs/express_cor/configs/cascade_mask_rcnn_x101_32x4d_dcn_fpn_carafe_1x_road_crack_200423.py'
    checkpoint = 'project_work_dirs/express_cor/work_dirs/cascade_mask_rcnn_x101_32x4d_dcn_fpn_carafe_1x_road_crack_200423/epoch_26.pth'

# load the model on GPU
device = 'cuda:0'
model = init_detector(config, checkpoint, device=device)

# inference for crack
for pctrList in damage_detection_output["pctrList"]:

    pctrList = {}

    pctrList["strtDttm"] = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    pctrList["anlyPctrId"] = num + 1
    pctrList["anlyPctrNm"] = os.path.basename(img_path)

    _, mask_output = inference_detector_sliding_window(model, img_path, color_mask[0], score_thr=0.1, window_size=1024,
                                                       overlap_ratio=0.3)

    if np.sum(mask_output) > 0:

        mask_output = connect_cracks(mask_output)
        mask_output = connect_cracks(mask_output)
        mask_output = remove_cracks(mask_output)

        skel, distance = medial_axis(mask_output, return_distance=True)
        dist_on_skel = distance * skel

        labels = label(mask_output)

        damage_region_prop = regionprops_table(labels, properties=('label', 'centroid'))

        pctrList['cordList'] = []

        for label_num in range(np.max(labels)):

            a_label = labels == label_num + 1

            contours, _ = cv2.findContours(a_label.astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

            imprXcord = str()
            imprYcord = str()

            for countour in contours[0]:
                imprYcord = imprYcord + str(countour[0][0]) + ','
                imprXcord = imprXcord + str(countour[0][1]) + ','

            imprYcord = imprYcord[:-1]
            imprXcord = imprXcord[:-1]

            imprCnterCord = str(damage_region_prop['centroid-0'][label_num]) + ',' + str(
                damage_region_prop['centroid-1'][label_num])

            dist_label = dist_on_skel[a_label]

            cordList = {}

            cordList["cordTypeCd"] = "2"
            cordList["imprXcord"] = imprXcord
            cordList["imprYcord"] = imprYcord
            cordList["imprCnterCord"] = imprCnterCord
            cordList["imprTypeCd"] = "01"
            cordList["imprWdth"] = str(dist_label[np.nonzero(dist_label)].mean() * 0.2)
            cordList["imprLnth"] = str(np.sum(skel[a_label]) * 0.2)
            cordList["imprBrdthVal"] = ""
            cordList["imprQntt"] = ""
            pctrList['cordList'].append(cordList)

        pctrList["endDttm"] = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        damage_detection_output["pctrList"].append(pctrList)


    elif np.sum(mask_output) == 0:
        print(img_path + ' has no crack detection result')

# detection and post processing for efflorescence

with open('person.txt', 'w') as json_file:
    json.dump(person_dict, json_file)