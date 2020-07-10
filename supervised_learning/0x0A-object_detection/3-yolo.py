#!/usr/bin/env python3
""" This module has the Yolo class """

import tensorflow.keras as K
import tensorflow as tf
import numpy as np

def _sigmoid(x):
    """ This method calculates sigmoid function"""
    return 1. / (1. + np.exp(-x))


class Yolo():
    """
    This is a class to use the Yolo v3 algorithm to perform object detection
    """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """ Everything begins here """
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            data = f.read()
        self.class_names = data.split()
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """ This method process output """

        boxes, box_confidences, box_class_probs = [], [], []

        for i in range(len(outputs)):
            ih, iw = image_size
            t_xy, t_wh, objectness, classes = np.split(outputs[i],
                                                       (2, 4, 5),
                                                       axis=-1)

            # print(t_xy.shape, t_wh.shape, objectness.shape, classes.shape)

            box_confidences.append(_sigmoid(objectness))
            box_class_probs.append(_sigmoid(classes))

            grid_size = np.shape(outputs[i])[1]
            # bh bw debe ser normalizado dividiendose por el input shape
            # x = bx -bw / 2
            # y = by -bh /2
            C_xy = np.meshgrid(range(grid_size), range(grid_size))

            #
            # [[[0, 0], [1, 0], [2, 0]],
            #  [[0, 1], [1, 1], [2, 1]],
            #  [[0, 2], [1, 2], [2, 2]]]
            #
            C_xy = np.stack(C_xy, axis=-1)
            # [[[[0, 0]], [[1, 0]], [[2, 0]]],
            #  [[[0, 1]], [[1, 1]], [[2, 1]]],
            #  [[[0, 2]], [[1, 2]], [[2, 2]]]]
            #

            C_xy = np.expand_dims(C_xy, axis=2)  # [gx, gy, 1, 2]

            # YoloV2, YoloV3:
            # bx = sigmoid(tx) + Cx
            # by = sigmoid(ty) + Cy
            #
            #
            # [[[[0.1, 0.2]], [[1.1, 0.2]], [[2.1, 0.2]]],
            #  [[[0.1, 1.2]], [[1.1, 1.2]], [[2.1, 1.2]]],
            #  [[[0.1, 2.2]], [[1.1, 2.2]], [[2.1, 2.2]]]]
            #
            b_xy = _sigmoid(t_xy) + C_xy

            b_wh = (np.exp(t_wh) / 416) * self.anchors[i]

            bx = b_xy[:, :, :, :1]
            by = b_xy[:, :, :, 1:2]
            bw = b_wh[:, :, :, :1]
            bh = b_wh[:, :, :, 1:2]

            x1 = (bx - bw / 2) * image_size[1]
            y1 = (by - bh / 2) * image_size[0]
            x2 = (bx + bw / 2) * image_size[1]
            y2 = (by + bh / 2) * image_size[0]

            boxes.append(np.concatenate([x1, y1, x2, y2], axis=-1))

        return (boxes, box_confidences, box_class_probs)

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """ This method filter the boxes """
        v_boxes, v_labels, v_scores = [], [], []
        for i in range(len(boxes)):
            a, b, c, d = boxes[i].shape
            resha = boxes[i].reshape(a * b * c, d)
            a, b, c, d = box_confidences[i].shape
            resha_conf = box_confidences[i].reshape(a * b * c, d)
            a, b, c, d = box_class_probs[i].shape
            resha_probs = box_class_probs[i].reshape(a * b * c, d)
            for box in range(len(resha)):
                pos = np.argmax(resha_probs[box])
                score = resha_probs[box][pos] * resha_conf[box]
                if score > self.class_t:
                    v_boxes = np.concatenate([v_boxes, resha[box]])
                    v_labels = np.concatenate([v_labels, [pos]], axis=-1)
                    v_scores = np.concatenate([v_scores, score])
        filas = v_boxes.shape[0] // 4
        v_boxes = v_boxes.reshape(filas, 4)
        return v_boxes, v_labels.astype(int), v_scores

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """ This method return the real boxes, scores, classes predicted """
        if len(filtered_boxes) == 0:
            return []
        x1 = filtered_boxes[:, 0]
        y1 = filtered_boxes[:, 1]
        x2 = filtered_boxes[:, 2]
        y2 = filtered_boxes[:, 3]
        ind = np.lexsort((-box_scores, box_classes))
        _, class_count = np.unique(box_classes, return_counts=True)
        i = 0
        keep_i = []
        for c in class_count:
            c_boxes = ind[i:i + c]
            while len(c_boxes):
                fix = c_boxes[0]
                keep_i += [fix]
                c_boxes = c_boxes[1:]
                keep_tmp = []
                for b in c_boxes:
                    xA = max(x1[fix], x1[b])
                    yA = max(y1[fix], y1[b])
                    xB = min(x2[fix], x2[b])
                    yB = min(y2[fix], y2[b])
                    interArea = max(0, xB - xA) * max(0, yB - yA)
                    boxAArea = (x2[fix] - x1[fix]) * (y2[fix] - y1[fix])
                    boxBArea = (x2[b] - x1[b]) * (y2[b] - y1[b])
                    overlap = interArea / ((boxAArea + boxBArea) - interArea)
                    if overlap > self.nms_t:
                        pass
                    else:
                        keep_tmp += [b]
                c_boxes = keep_tmp
            i += c
        return filtered_boxes[keep_i], box_classes[keep_i], box_scores[keep_i]
