#!/usr/bin/env python
# encoding: utf-8

"""
@version: ??
@author: r.li
@license: Apache Licence 
@contact: r.li@bmi-tech.com
@site: 
@software: PyCharm
@file: sfd.py
@time: 18-5-21 下午5:43
"""

import numpy as np
import cv2

import time

caffe_ssd_root = '/home/lirui/packages/caffe_ssd'  # this file is expected to be in {sfd_root}/sfd_test_code/AFW
import os
import sys

sys.path.insert(0, os.path.join(caffe_ssd_root, 'python'))
import caffe


class SFD:
    def __init__(self, model_def, model_weights, conf_thresh=0.8, img_max_side=480.0, gpu_id=0):
        caffe.set_device(gpu_id)
        caffe.set_mode_gpu()

        self.net = caffe.Net(model_def, model_weights, caffe.TEST)
        self.conf_thresh = conf_thresh
        self.img_max_side = img_max_side

    def set_cfgs(self, conf_thresh, img_max_side):
        self.conf_thresh = conf_thresh
        self.img_max_side = img_max_side

    def detect(self, img_arr):
        heigh = img_arr.shape[0]
        width = img_arr.shape[1]

        im_shrink = self.img_max_side / max(img_arr.shape[0], img_arr.shape[1])
        im_shrink = im_shrink if im_shrink < 1 else 1
        image = cv2.resize(img_arr, None, None, fx=im_shrink, fy=im_shrink, interpolation=cv2.INTER_LINEAR)

        # print("in size: ", image.shape)
        self.net.blobs['data'].reshape(1, 3, image.shape[0], image.shape[1])

        self.transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
        self.transformer.set_transpose('data', (2, 0, 1))
        self.transformer.set_mean('data', np.array([104, 117, 123]))
        self.transformer.set_raw_scale('data', 255)
        self.transformer.set_channel_swap('data', (2, 1, 0))

        transformed_image = self.transformer.preprocess('data', image.astype(np.float32))

        # print("preprocess time: ", time.time() - tic)
        self.net.blobs['data'].data[...] = transformed_image

        detections = self.net.forward()['detection_out']

        # print("forward time: ", time.time() - tic)

        # filer by confidence
        detections_out = detections[0, 0, :, 2:7]
        detections_out = detections_out[detections_out[:, 0].argsort()[::-1], :]
        thresh_pos = np.where(detections_out[:, 0] >= self.conf_thresh)[0]
        detections_out = detections_out[thresh_pos, :]

        if detections_out.shape[0] > 4:
            print("qqqqqqqqqqq")
        detections_out[:, 1:5] = np.maximum(detections_out[:, 1:5], 0)
        detections_out[:, 1:5] = np.minimum(detections_out[:, 1:5], 1)
        detections_out[:, (1, 3)] *= width
        detections_out[:, (2, 4)] *= heigh

        return detections_out

    def detect_batch(self, imgs):
        pass


if __name__ == '__main__':
    pass
