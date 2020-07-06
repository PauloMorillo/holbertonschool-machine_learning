#!/usr/bin/env python3
""" This module has the Yolo class """

import tensorflow.keras as K
import tensorflow as tf


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
        """
        This method return the real boxes, scores, classes predicted
        """
        if len(filtered_boxes) == 0:
            return []
        pick = []
        x1 = filtered_boxes[:, 0]
        y1 = filtered_boxes[:, 1]
        x2 = filtered_boxes[:, 2]
        y2 = filtered_boxes[:, 3]
        area = (x2 - x1) * (y2 - y1)
        # idxs = np.argsort(y2)
        ind = np.lexsort((box_scores, -box_classes))

        keep_idx = []
        del_idx = []
        classes = np.unique(box_classes)
        for c in classes:
            class_boxes = np.where(box_classes == c)[0]
            idxs = np.argsort(box_scores[class_boxes])[::-1]
            while len(idxs) > 0:
                keep_idx.append(class_boxes[idxs[0]])
                keep_tmp = []
                i = idxs[0]
                del_idx = [0]
                aux = 0
                for j in idxs[1:]:
                    xA = max(x1[i], x1[j])
                    yA = max(y1[i], y1[j])
                    xB = min(x2[i], x2[j])
                    yB = min(y2[i], y2[j])

                    interArea = max(0, xB - xA) * max(0, yB - yA)
                    boxAArea = (x2[i] - x1[i]) * (y2[i] - y1[i])
                    boxBArea = (x2[j] - x1[j]) * (y2[j] - y1[j])

                    overlap = interArea / ((boxAArea + boxBArea) - interArea)

                    if overlap > self.nms_t:
                        aux += 1
                        del_idx.append(aux)
                        # keep_tmp.append(j)

                # idxs = keep_tmp
                idxs = np.delete(idxs, aux)
                # keep_idx += list(class_boxes[keep_tmp])
        return (filtered_boxes[keep_idx],
                box_classes[keep_idx],
                box_scores[keep_idx])
