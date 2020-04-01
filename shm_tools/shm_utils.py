
import mmcv
import cv2
import os
import glob
import numpy as np
import slidingwindow as sw
import pycocotools.mask as maskUtils

from mmdet.apis import init_detector, inference_detector
from skimage.measure import label, regionprops_table


def imread(filename, flags=cv2.IMREAD_COLOR, dtype=np.uint8):
    try:
        n = np.fromfile(filename, dtype)
        imageBGR = cv2.imdecode(n, flags)
        return cv2.cvtColor(imageBGR, cv2.COLOR_BGR2RGB)

    except Exception as e:
        print(e)
        return None


def imwrite(filename, imageRGB, params=None):
    try:
        ext = os.path.splitext(filename)[1]
        imageBGR = cv2.cvtColor(imageRGB, cv2.COLOR_RGB2BGR)
        result, n = cv2.imencode(ext, imageBGR, params)
        if result:
            with open(filename, mode='w+b') as f:
                n.tofile(f)
                return True
        else:
                return False

    except Exception as e:
        print(e)
        return False


def inference_detector_sliding_window(model, input_img, color_mask,
                                      score_thr = 0.1, window_size = 1024, overlap_ratio = 0.5,):

    '''
    :param model: is a mmdetection model object
    :param input_img : str or numpy array
                    if str, run imread from input_img
    :param score_thr: is float number between 0 and 1.
                   Bounding boxes with a confidence higher than score_thr will be displayed,
                   in 'img_result' and 'mask_output'.
    :param window_size: is a subset size to be detected at a time.
                        default = 1024, integer number
    :param overlap_ratio: is a overlap size.
                        If you overlap sliding windows by 50%, overlap_ratio is 0.5.

    :return: img_result
    :return: mask_output

    '''

    # color mask has to be updated for multiple-class object detection
    if isinstance(input_img, str) :
        img = imread(input_img)
    else :
        img = input_img

    # Generate the set of windows, with a 256-pixel max window size and 50% overlap
    windows = sw.generate(img, sw.DimOrder.HeightWidthChannel, window_size, overlap_ratio)
    mask_output = np.zeros((img.shape[0], img.shape[1]), dtype=np.bool)

    for window in windows:
        # Add print option for sliding window detection
        img_subset = img[window.indices()]
        results = inference_detector(model, img_subset)
        bbox_result, segm_result = results
        mask_sum = np.zeros((img_subset.shape[0], img_subset.shape[1]), dtype=np.bool)
        bboxes = np.vstack(bbox_result)  # bboxes

        # draw segmentation masks
        if segm_result is not None:
            segms = mmcv.concat_list(segm_result)
            inds = np.where(bboxes[:, -1] > score_thr)[0]

            for i in inds:
                mask = maskUtils.decode(segms[i]).astype(np.bool)
                mask_sum = mask_sum + mask

        mask_output[window.indices()] = mask_sum

    mask_output = mask_output.astype(np.uint8)
    mask_output[mask_output > 1] = 1

    mask_output_bool = mask_output.astype(np.bool)

    # Add colors to detection result on img
    img_result = img
    img_result[mask_output_bool, :] = img_result[mask_output_bool,:] * 0.3 + color_mask * 0.6

    return img_result, mask_output


def connect_cracks(mask_output, epsilon = 200):
    '''
    :param mask_output: a numpy uint8 variable
    :param epsilon: distance between cracks to be connected
    :return: connect_mask_output : crack-connection result
    '''

    '''
    To-dos : 
    1 . Add iteration option
    2 . Add connection option considering a direction of a crack
    with the direction of ellipse of each crack 
    '''

    # label each crack
    labels, num = label(mask_output, connectivity=2, return_num=True)
    # get information of each crack area
    crack_region_table = regionprops_table(labels, properties=('label', 'bbox', 'coords', 'orientation'))

    width = crack_region_table['bbox-3'] - crack_region_table['bbox-1']
    height = crack_region_table['bbox-2'] - crack_region_table['bbox-0']

    crack_region_table['is_horizontal'] = width > height

    e2_list = []
    e1_list = []

    for crack_num, crack_region in enumerate(crack_region_table['label']):

        min_row = crack_region_table['bbox-0'][crack_num]
        min_col = crack_region_table['bbox-1'][crack_num]
        max_row = crack_region_table['bbox-2'][crack_num]-1
        max_col = crack_region_table['bbox-3'][crack_num]-1

        if crack_region_table['is_horizontal'][crack_num]:
            # max col / min col
            col = crack_region_table['coords'][crack_num][:, 1]

            e2 = crack_region_table['coords'][crack_num][np.argwhere(col == max_col), :][-1][0]
            e1 = crack_region_table['coords'][crack_num][np.argwhere(col == min_col), :][0][0]

            e2_list.append(e2)
            e1_list.append(e1)

        else:
            # max row / min row
            row = crack_region_table['coords'][crack_num][:, 0]

            e2 = crack_region_table['coords'][crack_num][np.argwhere(row == max_row), :][-1][0]
            e1 = crack_region_table['coords'][crack_num][np.argwhere(row == min_row), :][0][0]

            e2_list.append(e2)
            e1_list.append(e1)

    crack_region_table['e2'] = e2_list
    crack_region_table['e1'] = e1_list

    connect_line_img = np.zeros_like(mask_output, dtype=np.uint8)
    n = len(crack_region_table['label'])
    color = (1) # binary image

    for i in range(n):  # scan through all of crack area
        k_list = []
        for k in range(n):
            if not k == i: # compare with all the other cracks
                distance = []
                e_list = []

                # compare every direction
                # close-close / far-close / close-far / far-far
                for e1 in ['e1', 'e2']:
                    for e2 in ['e1', 'e2']:
                        d = np.subtract(crack_region_table[e1][k], crack_region_table[e2][i])
                        distance.append(np.sqrt(d[0] ** 2 + d[1] ** 2))
                        e_list.append([e1, e2])

                if not k_list: # when k_list is empty,
                    k_list.append([distance, e_list, k])

                elif np.min(k_list[0][0]) > np.min(distance):
                    k_list = []
                    k_list.append([distance, e_list, k])

        if k_list:

            if np.min(k_list[0][0]) < epsilon:
                dist_idx = np.argmin(k_list[0][0])
                e1, e2 = k_list[0][1][dist_idx]
                k_ = k_list[0][2]

                start_point = crack_region_table[e1][k_][::-1]
                end_point = crack_region_table[e2][i][::-1]

                connect_line_img = cv2.line(connect_line_img, tuple(start_point), tuple(end_point), color, 8)

    mask_output = mask_output + connect_line_img
    mask_output[mask_output > 1] = 1

    return mask_output

def remove_cracks(mask_output, threshold = 300):
    '''
    :param mask_output: a numpy uint8 variable
    :param threshold: cracks of which length is under thershold will be removed.
    :return: mask_output : crack mask after thresholding
    '''

    labels, num = label(mask_output, connectivity=2, return_num=True)
    crack_region_table = regionprops_table(labels, properties=('label', 'bbox', 'coords'))

    width = crack_region_table['bbox-3'] - crack_region_table['bbox-1']
    height = crack_region_table['bbox-2'] - crack_region_table['bbox-0']
    crack_region_table['diagonal_length'] = np.sqrt(height**2 + width**2)

    for crack_num in range(len(crack_region_table['label'])):
        if crack_region_table['diagonal_length'][crack_num] < threshold :
            for c in crack_region_table['coords'][crack_num]:
                mask_output[c[0], c[1]] = 0

    return mask_output


class AverageMeter(object):
    # This function is imported from https://github.com/hszhao/semseg/blob/master/util/util.py
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def comparison_operator(img, thr, ind='>'):
    import operator
    if ind == '==':
        return operator.eq(img, thr)
    elif ind == '<':
        return operator.lt(img, thr)
    elif ind == '>':
        return operator.gt(img, thr)
    elif ind == '!=':
        return operator.ne(img, thr)


def intersectionAndUnion(output, target, K, ignore_index=255):
    # This function is imported from https://github.com/hszhao/semseg/blob/master/util/util.py
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.ndim in [1, 2, 3])
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = ignore_index
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K + 1))
    area_output, _ = np.histogram(output, bins=np.arange(K + 1))
    area_target, _ = np.histogram(target, bins=np.arange(K + 1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


def cal_acc(data_list, pred_folder, classes, names):
    # This function is imported from https://github.com/hszhao/semseg/blob/master/util/util.py
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    for i, (image_path, target_path) in enumerate(data_list):
        image_name = image_path.split('/')[-1].split('.')[0]
        pred = cv2.imread(os.path.join(pred_folder, image_name + '.png'), cv2.IMREAD_GRAYSCALE)
        target = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)
        intersection, union, target = intersectionAndUnion(pred, target, classes)
        intersection_meter.update(intersection)
        union_meter.update(union)
        target_meter.update(target)
        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        logger.info(
            'Evaluating {0}/{1} on image {2}, accuracy {3:.4f}.'.format(i + 1, len(data_list), image_name + '.png',
                                                                        accuracy))

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    logger.info('Eval result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
    for i in range(classes):
        logger.info('Class_{} result: iou/accuracy {:.4f}/{:.4f}, name: {}.'.format(i, iou_class[i], accuracy_class[i],
                                                                                    names[i]))




