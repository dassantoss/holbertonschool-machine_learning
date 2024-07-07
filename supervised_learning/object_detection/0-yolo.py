#!/usr/bin/env python3
"""
Initialize Yolo
"""
from tensorflow import keras as K


class Yolo:
    """Class that uses the Yolo v3 algorithm to perform object detection"""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Initializes the Yolo instance.

        Parameters:
        - model_path: path to where a Darknet Keras model is stored
        - classes_path: path to where the list of class names used for
            the Darknet model is found
        - class_t: float representing the box score threshold for the
            initial filtering step
        - nms_t: float representing the IOU threshold for non-max
            suppression
        - anchors: numpy.ndarray of shape (outputs, anchor_boxes, 2)
            containing all of the anchor boxes
        """
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as file:
            self.class_names = file.read().strip().split('\n')
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
