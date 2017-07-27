import numpy as np
import re
import sys

import chainer.links.caffe.caffe_function as caffe

from lib.ssd import _Normalize


def _rename(name):
    m = re.match(r'^conv(\d+)_([123])$', name)
    if m:
        i, j = map(int, m.groups())
        if i >= 6:
            i += 2
        return 'conv{:d}_{:d}'.format(i, j), True

    m = re.match(r'^fc([67])$', name)
    if m:
        return 'conv{:d}'.format(int(m.group(1))), True

    if name == r'conv4_3_norm':
        return 'norm4', True

    m = re.match(r'^conv4_3_norm_mbox_(loc|conf)$', name)
    if m:
        return '{:s}/0'.format(m.group(1)), True

    m = re.match(r'^fc7_mbox_(loc|conf)$', name)
    if m:
        return ('{:s}/1'.format(m.group(1))), True

    m = re.match(r'^conv(\d+)_2_mbox_(loc|conf)$', name)
    if m:
        i, type_ = int(m.group(1)), m.group(2)
        if i >= 6:
            return '{:s}/{:d}'.format(type_, i - 4), True

    return name, False


class _CaffeFunction(caffe.CaffeFunction):

    def __init__(self, model_path, verbose=False):
        self.verbose = verbose
        if self.verbose:
            print(
                'loading weights from {:s} ... '.format(model_path),
                file=sys.stderr)
        super().__init__(model_path)

    def __setattr__(self, name, link):
        new_name, match = _rename(name)
        if match and hasattr(self, 'verbose') and self.verbose:
            print('{:s} -> {:s}'.format(name, new_name), file=sys.stderr)
        super().__setattr__(new_name, link)

    @caffe._layer('Normalize', None)
    def _setup_normarize(self, layer):
        blobs = layer.blobs
        func = _Normalize(caffe._get_num(blobs[0]))
        func.scale.data[:] = np.array(blobs[0].data)
        with self.init_scope():
            setattr(self, layer.name, func)

    @caffe._layer('AnnotatedData', None)
    @caffe._layer('Flatten', None)
    @caffe._layer('MultiBoxLoss', None)
    @caffe._layer('Permute', None)
    @caffe._layer('PriorBox', None)
    def _skip_layer(self, _):
        pass


def load_caffe(model_path, verbose=False):
    return _CaffeFunction(model_path, verbose)
