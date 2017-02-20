import cv2
import numpy as np
import random

from rect import Rect, matrix_iou

import chainer


def crop(image, boxes, classes):
    height, width, _ = image.shape

    while True:
        mode = random.choice((
            None,
            (0.1, None),
            (0.3, None),
            (0.7, None),
            (0.9, None),
            (None, None),
        ))

        if mode is None:
            return image, boxes, classes

        min_iou, max_iou = mode
        if min_iou is None:
            min_iou = float('-inf')
        if max_iou is None:
            max_iou = float('inf')

        for _ in range(50):
            w = random.randrange(int(0.3 * width), width)
            h = random.randrange(int(0.3 * height), height)

            if h / w < 0.5 or 2 < h / w:
                continue

            rect = Rect.LTWH(
                random.randrange(width - w),
                random.randrange(height - h),
                w,  h)

            iou = matrix_iou(boxes, np.array((rect,)))
            if iou.min() < min_iou and max_iou < iou.max():
                continue

            image = image[rect.top:rect.bottom, rect.left:rect.right]

            centers = (boxes[:, :2] + boxes[:, 2:]) / 2
            mask = np.logical_and(
                (rect.left, rect.top) < centers,
                centers < (rect.right, rect.bottom)).all(axis=1)
            boxes = boxes[mask].copy()
            classes = classes[mask]

            boxes[:, :2] = np.maximum(boxes[:, :2], (rect.left, rect.top))
            boxes[:, :2] -= (rect.left, rect.top)
            boxes[:, 2:] = np.minimum(boxes[:, 2:], (rect.right, rect.bottom))
            boxes[:, 2:] -= (rect.left, rect.top)

            return image, boxes, classes

        print('over')


def mirror(image, boxes, classes):
    _, w, _ = image.shape
    if random.randrange(2):
        image = image[:, ::-1]
        boxes = boxes.copy()
        boxes[:, 0::2] = w - boxes[:, 2::-2]
    return image, boxes, classes


def augment(image, boxes, classes):
    image, boxes, classes = crop(image, boxes, classes)
    image, boxes, classes = mirror(image, boxes, classes)
    return image, boxes, classes


class SSDLoader(chainer.dataset.DatasetMixin):

    def __init__(self, dataset, size, mean, encoder):
        super().__init__()

        self.dataset = dataset
        self.size = size
        self.mean = mean
        self.encoder = encoder

    def __len__(self):
        return len(self.dataset)

    def get_example(self, i):
        image = self.dataset.image(i)

        boxes, classes = zip(*self.dataset.annotations(i))
        boxes = np.array(boxes)
        classes = np.array(classes)

        image, boxes, classes = augment(image, boxes, classes)

        h, w, _ = image.shape
        image = cv2.resize(image, (self.size, self.size)).astype(np.float32)
        image -= self.mean
        image = image.transpose(2, 0, 1)
        boxes[:, 0::2] /= w
        boxes[:, 1::2] /= h
        loc, conf = self.encoder.encode(boxes, classes)

        return image, loc, conf
