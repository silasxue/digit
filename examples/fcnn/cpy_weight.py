# -*- coding: utf-8 -*-
from __future__ import division
caffe_root = '../../'
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe
import numpy as np
base_weights = 'fcn-32s-pascalcontext.caffemodel'