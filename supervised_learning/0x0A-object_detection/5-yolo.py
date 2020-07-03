#!/usr/bin/env python3
""" This module has the Yolo class """

import tensorflow.keras as K
import numpy as np
import cv2
import glob


def _sigmoid(x):
    """ This method calculates sigmoid function"""
    return 1. / (1. + np.exp(-x))


class Yolo():
    """
    This is a class to use the Yolo v3 algorithm to perform object detection
    model_path is the path to where a Darknet Keras model is stored
    classes_path is the path to where the list of class names used for the
    Darknet model, listed in order of index, can be found
    class_t is a float representing the box score threshold for the initial
    filtering step nms_t is a float representing the IOU threshold for non-max
    suppression anchors is a numpy.ndarray of shape (outputs, anchor_boxes, 2)
    containing all of the anchor boxes:
    outputs is the number of outputs (predictions) made by the Darknet model
    anchor_boxes is the number of anchor boxes used for each prediction
    2 => [anchor_box_width, anchor_box_height] public method
    def filter_boxes(self, boxes, box_confidences, box_class_probs):
    boxes: a list of numpy.ndarrays of shape (grid_height, grid_width,
    anchor_boxes, 4) containing the processed boundary boxes for each output,
    respectively box_confidences: a list of numpy.ndarrays of shape
    (grid_height, grid_width, anchor_boxes, 1) containing the processed box
    confidences for each output, respectively box_class_probs: a list of
    numpy.ndarrays of shape (grid_height, grid_width, anchor_boxes, classes)
    containing the processed box class probabilities for each output,
    respectively Returns a tuple of (filtered_boxes, box_classes, box_scores):
    filtered_boxes:
    a numpy.ndarray of shape (?, 4) containing all of the filtered bounding
    boxes:
    box_classes: a numpy.ndarray of shape (?,) containing the class number
    that each box in filtered_boxes predicts, respectively box_scores:
    a numpy.ndarray of shape (?) containing the box scores for each box in
    filtered_boxes, respectively
    """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """ Everything begins here """
        # self.model = K.models.load_model(model_path)
        # with open(classes_path, 'r') as f:
        #    data = f.read()
        # self.class_names = data.split()
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """ This method process output """

        boxes, box_confidences, box_class_probs = [], [], []

        for i in range(len(outputs)):
            ih, iw = image_size
            t_xy, t_wh, objectness, classes = np.split(outputs[i],
                                                       (2, 4, 5), axis=-1)

            # print(t_xy.shape, t_wh.shape, objectness.shape, classes.shape)

            box_confidences.append(_sigmoid(objectness))
            box_class_probs.append(_sigmoid(classes))

            grid_size = np.shape(outputs[i])[1]
            # bh bw debe ser normalizado dividiendose por el input shape
            # x = bx -bw / 2
            # y = by -bh /2
            # meshgrid generates a grid that repeats by given range. It's the
            # Cx and Cy in YoloV3 paper.
            # for example, tf.meshgrid(tf.range(3), tf.range(3)) will generate
            # a list with two elements
            # note that in real code, the grid_size should be something like
            # 13, 26, 52 for examples here and below
            #
            # [[0, 1, 2],
            #  [0, 1, 2],
            #  [0, 1, 2]]
            #
            # [[0, 0, 0],
            #  [1, 1, 1],
            #  [2, 2, 2]]
            #
            C_xy = np.meshgrid(range(grid_size), range(grid_size))

            # next, we stack two items in the list together in the last
            # dimension,
            # so that
            # we can interleve these elements together and become this:
            #
            # [[[0, 0], [1, 0], [2, 0]],
            #  [[0, 1], [1, 1], [2, 1]],
            #  [[0, 2], [1, 2], [2, 2]]]
            #
            C_xy = np.stack(C_xy, axis=-1)

            # let's add an empty dimension at axis=2 to expand the tensor
            # to this:
            #
            # [[[[0, 0]], [[1, 0]], [[2, 0]]],
            #  [[[0, 1]], [[1, 1]], [[2, 1]]],
            #  [[[0, 2]], [[1, 2]], [[2, 2]]]]
            #
            # at this moment, we now have a grid, which can always give
            # us (y, x)
            # if we access grid[x][y]. For example, grid[0][1] == [[1, 0]]

            C_xy = np.expand_dims(C_xy, axis=2)  # [gx, gy, 1, 2]

            # YoloV2, YoloV3:
            # bx = sigmoid(tx) + Cx
            # by = sigmoid(ty) + Cy
            #
            # for example, if all elements in b_xy are (0.1, 0.2), the
            # result will be
            #
            # [[[[0.1, 0.2]], [[1.1, 0.2]], [[2.1, 0.2]]],
            #  [[[0.1, 1.2]], [[1.1, 1.2]], [[2.1, 1.2]]],
            #  [[[0.1, 2.2]], [[1.1, 2.2]], [[2.1, 2.2]]]]
            #
            b_xy = _sigmoid(t_xy) + C_xy

            # finally, divide this absolute box_xy by grid_size, and then
            # we will get the normalized bbox centroids
            # for each anchor in each grid cell. b_xy is now in shape
            # (batch_size, grid_size, grid_size, num_anchor, 2)
            #
            # [[[[0.1/3, 0.2/3]], [[1.1/3, 0.2/3]], [[2.1/3, 0.2/3]]],
            #  [[[0.1/3, 1.2/3]], [[1.1/3, 1.2]/3], [[2.1/3, 1.2/3]]],
            #  [[[0.1/3, 2.2/3]], [[1.1/3, 2.2/3]], [[2.1/3, 2.2/3]]]]
            #
            b_xy = b_xy / grid_size

            # YoloV2:
            # "If the cell is offset from the top left corner of the image
            # by (cx , cy)
            # and the bounding box prior has width and height pw , ph ,
            # then the predictions correspond to: "
            #
            # https://github.com/pjreddie/darknet/issues/568#issuecomment-469600294
            # "It’s OK for the predicted box to be wider and/or taller
            # than the original image, but
            # it does not make sense for the box to have a negative width
            # or height. That’s why
            # we take the exponent of the predicted number."
            b_wh = (np.exp(t_wh) /
                    self.model.input_shape[1:3]) * self.anchors[i]

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

    @staticmethod
    def load_images(folder_path):
        """ This method load an image """
        images = [cv2.imread(file)
                  for file in glob.glob(folder_path + '/*.jpg')]
        paths = [file for file in glob.glob(folder_path + '/*.jpg')]
        return images, paths

    def preprocess_images(self, images):
        """ Preprocess images to posterior use with darknet"""
        resized = []
        dims = self.model.input_shape[1:3]
        for img in images:
            resized.append(cv2.resize(img, dims,
                                      interpolation=cv2.INTER_CUBIC) / 255)
        shapes = []
        for img in images:
            shapes.append(img.shape[:2])
        return np.array(resized), np.array(shapes)
