#! /usr/bin/env python3

import argparse
#import cv2
from PIL import Image
import matplotlib.pyplot as plot
import numpy as np

import chainer
from chainer import serializers

from lib import MultiBoxEncoder
from lib import preproc_for_test
from lib import SSD300
from lib import SSD512
from lib import VOCDataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', choices=('300', '512'), default='300')
    parser.add_argument('model')
    parser.add_argument('image')
    args = parser.parse_args()

    if args.arch == '300':
        model = SSD300(20)
    elif args.arch == '512':
        model = SSD512(20)
    serializers.load_npz(args.model, model)

    multibox_encoder = MultiBoxEncoder(model)

    src = Image.open(args.image).convert('RGB')
    src = np.array(src)
    src = src[:, :, ::-1].copy()
    #src = cv2.imread(args.image, cv2.IMREAD_COLOR)
    image = preproc_for_test(src, model.insize, model.mean)

    loc, conf = model(image[np.newaxis])
    loc = chainer.cuda.to_cpu(loc.data)
    conf = chainer.cuda.to_cpu(conf.data)
    boxes, labels, scores = multibox_encoder.decode(
        loc[0], conf[0], 0.45, 0.01)

    figure = plot.figure()
    ax = figure.add_subplot(111)
    ax.imshow(src[:, :, ::-1])

    for box, label, score in zip(boxes, labels, scores):
        box[:2] *= src.shape[1::-1]
        box[2:] *= src.shape[1::-1]
        box = box.astype(int)
        
        label = label[0]
        score = score[0]

        print(label + 1, score, *box)

        if score > 0.6:
            ax.add_patch(plot.Rectangle(
                (box[0], box[1]), box[2] - box[0], box[3] - box[1],
                fill=False, edgecolor='red', linewidth=3))
            ax.text(
                box[0], box[1],
                '{:s}: {:0.2f}'.format(VOCDataset.labels[label], score),
                bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 10})

    plot.show()
