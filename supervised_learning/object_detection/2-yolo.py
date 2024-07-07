#!/usr/bin/env python3
"""
Initialize Yolo and process outputs
"""
import numpy as np
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

    def sigmoid(self, x):
        """Applies sigmoid function"""
        return 1 / (1 + np.exp(-x))

    def process_outputs(self, outputs, image_size):
        """
        Process the outputs of the Darknet model

        Parameters:
        - outputs: list of numpy.ndarrays containing the predictions
            from the Darknet model for a single image
        - image_size: numpy.ndarray containing the image’s original
            size [image_height, image_width]

        Returns:
        - boxes: list of numpy.ndarrays containing the processed
            boundary boxes for each output
        - box_confidences: list of numpy.ndarrays containing the box
            confidences for each output
        - box_class_probs: list of numpy.ndarrays containing the box’s
            class probabilities for each output
        """
        boxes = []
        box_confidences = []
        box_class_probs = []

        image_height, image_width = image_size

        for i, output in enumerate(outputs):
            grid_height, grid_width, anchor_boxes, _ = output.shape

            # Extract box confidence, class probabilities, and coordinates
            box_confidence = self.sigmoid(output[..., 4:5])
            box_class_prob = self.sigmoid(output[..., 5:])
            box_xy = self.sigmoid(output[..., 0:2])  # Center coordinates
            box_wh = np.exp(output[..., 2:4])  # Width and height
            anchors = self.anchors[i].reshape((1, 1, len(self.anchors[i]), 2))
            box_wh *= anchors

            # Create grid to map coordinates
            col = np.tile(np.arange(0, grid_width),
                          grid_height).reshape(-1, grid_width)
            row = np.tile(np.arange(0, grid_height),
                          grid_width).reshape(-1, grid_width).T

            col = col.reshape(grid_height, grid_width, 1, 1).repeat(3, axis=-2)
            row = row.reshape(grid_height, grid_width, 1, 1).repeat(3, axis=-2)

            # Adjust coordinates
            box_xy += np.concatenate((col, row), axis=-1)
            box_xy /= (grid_width, grid_height)
            box_wh /= (self.model.input.shape[1], self.model.input.shape[2])

            # Convert from center coordinates to corner coordinates
            box_xy -= (box_wh / 2)
            boxes.append(np.concatenate((box_xy, box_xy + box_wh), axis=-1))

            box_confidences.append(box_confidence)
            box_class_probs.append(box_class_prob)

        # Adjust boxes to the original image size
        for i in range(len(boxes)):
            boxes[i][..., 0] *= image_width
            boxes[i][..., 1] *= image_height
            boxes[i][..., 2] *= image_width
            boxes[i][..., 3] *= image_height

        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        Filter the bounding boxes using the threshold.

        Parameters:
        - boxes: list of numpy.ndarrays containing the processed boundary
            boxes for each output
        - box_confidences: list of numpy.ndarrays containing the processed
            box confidences for each output
        - box_class_probs: list of numpy.ndarrays containing the processed
            box class probabilities for each output

        Returns:
        - filtered_boxes: numpy.ndarray containing all of the filtered
            bounding boxes
        - box_classes: numpy.ndarray containing the class number that
            each box in filtered_boxes predicts
        - box_scores: numpy.ndarray containing the box scores for each
            box in filtered_boxes
        """
        filtered_boxes = []
        box_classes = []
        box_scores = []

        for box, box_conf, box_class_prob in zip(boxes, box_confidences,
                                                 box_class_probs):
            # Compute box scores
            scores = box_conf * box_class_prob
            max_scores = np.max(scores, axis=-1)
            max_classes = np.argmax(scores, axis=-1)

            # Filter boxes by score threshold
            mask = max_scores >= self.class_t
            filtered_boxes.append(box[mask])
            box_classes.append(max_classes[mask])
            box_scores.append(max_scores[mask])

        filtered_boxes = np.concatenate(filtered_boxes)
        box_classes = np.concatenate(box_classes)
        box_scores = np.concatenate(box_scores)

        return filtered_boxes, box_classes, box_scores
