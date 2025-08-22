# -*- coding: utf-8 -*-
"""
Created on Aug,2025

@author: Lujia Lv
"""
import os
import numpy as np
import torch
import mAP
def bb_intersection_over_union(A, B):
    xA = max(A[0], B[0])
    yA = max(A[1], B[1])
    xB = min(A[2], B[2])
    yB = min(A[3], B[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0
    boxAArea = (A[2] - A[0]) * (A[3] - A[1])
    boxBArea = (B[2] - B[0]) * (B[3] - B[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def get_weighted_box(boxes):
    box = np.zeros(6, dtype=np.float32)
    conf = 0
    conf_list = []
    for b in boxes:
        box[:4] += (b[4] * b[:4])
        conf += b[4]
    box[4], box[5] = boxes[0][4], boxes[0][5]
    box[:4] /= conf
    return box

def find_matching_box(boxes_list, new_box, match_iou):
    best_iou = match_iou
    best_index = -1
    for i in range(len(boxes_list)):
        box = boxes_list[i]
        iou = bb_intersection_over_union(box[:4], new_box[:4])
        if iou > best_iou:
            best_index = i
            best_iou = iou
    return best_index, best_iou

def weighted_boxes_fusion(
        boxes,
        iou_thr=0.55,
        allows_overflow=False):

    clustered_boxes = []
    weighted_boxes = []

    for box in boxes:
        index, best_iou = find_matching_box(weighted_boxes, box, iou_thr)
        if index != -1:
            clustered_boxes[index].append(box.copy())
            weighted_boxes[index] = get_weighted_box(clustered_boxes[index])
        else:
            clustered_boxes.append([box.copy()])
            weighted_boxes.append(box[:6].copy())

    for index in range(len(weighted_boxes)):
        conf = sum(clustered_boxes[index][i][4] for i in range(len(clustered_boxes[index]))) / len(clustered_boxes[index])
        weighted_boxes[index][4] = conf

        pr_list = []
        for j in range(6, 17):
            sum_pr = 0
            sum_weight = 0
            for i in range(len(clustered_boxes[index])):
                sum_pr += clustered_boxes[index][i][j] * clustered_boxes[index][i][-1]
                sum_weight += clustered_boxes[index][i][-1]

            pr_list.append(sum_pr / (sum_weight + 1e-10))
        pr_list = np.array(pr_list)
        class_index = pr_list.argmax()
        weighted_boxes[index][5] = class_index

    for index in range(len(weighted_boxes)):
        if not allows_overflow:
             weighted_boxes[index][4] = weighted_boxes[index][4] * min(3, len(clustered_boxes[index])) / 3
        else:
             weighted_boxes[index][4] = weighted_boxes[index][4] * len(clustered_boxes[index]) / 3

    weighted_boxes = np.array(weighted_boxes)
    if weighted_boxes.shape[0] != 1 and weighted_boxes.shape[0] != 0:
        weighted_boxes = weighted_boxes[weighted_boxes[:, 4].argsort()[::-1]]

    return weighted_boxes

def run(v, dataset = 'valid'):
    if dataset == 'valid':
        model1Path = './Output result file for model1.'
        model2Path = './Output result file for model2.'
        model3Path = './Output result file for model3.'
        labelPath = './Label file.'

    weight_model1 = v[0]
    weight_model2 = v[1]
    weight_model3 = v[2]

    meanAP = mAP.MeanAveragePrecison()

    labeltxts = os.listdir(labelPath)
    for labeltxt in labeltxts:
        boxes = []
        labels = []
        labeltxt_path = os.path.join(labelPath, labeltxt)
        with open(labeltxt_path, "r") as file:
            for line in file:
                line = line.strip()
                contline = line.split(' ')
                data = [float(contline[0]), float(contline[1]), float(contline[2]), float(contline[3]),
                        float(contline[4])]
                labels.append(data)
        labels = torch.as_tensor(labels)

        model1txt_path = os.path.join(model1Path, labeltxt)
        if os.path.isfile(model1txt_path):
            with open(model1txt_path, "r") as file:
                for line in file:
                    line = line.strip()
                    contline = line.split(' ')
                    data = [float(contline[i]) for i in range(0, 17)]
                    data.append(weight_model1)
                    boxes.append(data)

        model2txt_path = os.path.join(model2Path, labeltxt)
        if os.path.isfile(model2txt_path):
            with open(model2txt_path, "r") as file:
                for line in file:
                    line = line.strip()
                    contline = line.split(' ')
                    data = [float(contline[i]) for i in range(0, 17)]
                    data.append(weight_model2)
                    boxes.append(data)

        model3txt_path = os.path.join(model3Path, labeltxt)
        if os.path.isfile(model3txt_path):
            with open(model3txt_path, "r") as file:
                for line in file:
                    line = line.strip()
                    contline = line.split(' ')
                    data = [float(contline[i]) for i in range(0, 17)]
                    data.append(weight_model3)
                    boxes.append(data)

        boxes = np.array(boxes)
        if boxes.shape[0] != 1 and boxes.shape[0] != 0:
            boxes = boxes[boxes[:, 4].argsort()[::-1]]
        weighted_boxes = weighted_boxes_fusion(boxes)
        if len(weighted_boxes) == 0:
            detections = torch.zeros((0, 6))
        else:
            detections = torch.as_tensor(weighted_boxes)
        meanAP.process_batch(detections, labels)
    tp, fp, p, r, f1, ap, ap_class = meanAP.calculate_ap_per_class()
    ap50, ap = ap[:, 0], ap.mean(1)
    mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
    return mp, mr, map50, map

