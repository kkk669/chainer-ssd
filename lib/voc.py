#import cv2
from PIL import Image
import numpy as np
import os
import xml.etree.ElementTree as ET


class VOCDataset(object):

    labels = (
        'aeroplane',
        'bicycle',
        'bird',
        'boat',
        'bottle',
        'bus',
        'car',
        'cat',
        'chair',
        'cow',
        'diningtable',
        'dog',
        'horse',
        'motorbike',
        'person',
        'pottedplant',
        'sheep',
        'sofa',
        'train',
        'tvmonitor',
    )

    def __init__(self, root, year, subset):
        self.root = os.path.join(root, 'VOC' + year)

        path = os.path.join(self.root, 'ImageSets', 'Main', subset + '.txt')
        self.images = [line.strip() for line in open(path)]

    def __len__(self):
        return len(self.images)

    def name(self, i):
        return self.images[i]

    def image(self, i):
        pil_img = Image.open(os.path.join(self.root, 'JPEGImages', self.images[i] + '.jpg')).convert('RGB')
        np_array = np.array(pil_img)
        return np_array[:, :, ::-1].copy()
        #return cv2.imread(
        #    os.path.join(self.root, 'JPEGImages', self.images[i] + '.jpg'),
        #    cv2.IMREAD_COLOR)

    def annotations(self, i):
        tree = ET.parse(os.path.join(
            self.root, 'Annotations', self.images[i] + '.xml'))

        boxes = list()
        labels = list()
        for child in tree.getroot():
            if not child.tag == 'object':
                continue
            bndbox = child.find('bndbox')
            box = tuple(
                float(bndbox.find(t).text) - 1
                for t in ('xmin', 'ymin', 'xmax', 'ymax'))
            label = self.labels.index(child.find('name').text)

            boxes.append(box)
            labels.append(label)

        return np.array(boxes), np.array(labels)
