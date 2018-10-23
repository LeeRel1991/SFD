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

import os
os.environ['GLOG_minloglevel'] = '2'

import sys

try:
    import caffe
except ImportError:
    import traceback

    traceback.print_exc()
    print("Error: please add $CAFFE_ROOT/python into sys.path or PYTHONPATH first")
    exit()


class SFD:
    """
    implementation of S3FD class for face detect
    """

    def __init__(self, model_def, model_weights, img_max_side=480.0, conf_thresh=0.8, gpu_id=0):
        """
        initialize
        Args:
            model_def: .prototxt
            model_weights:  .caffemodel
            img_max_side: 给网络输入的图像的最长边 
            conf_thresh: 置信概率的阈值
            gpu_id: gpu显卡号
        """
        caffe.set_mode_gpu()
        caffe.set_device(gpu_id)

        self.net = caffe.Net(model_def, model_weights, caffe.TEST)
        self.transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
        self.transformer.set_transpose('data', (2, 0, 1))
        self.transformer.set_mean('data', np.array([104, 117, 123]))
        self.transformer.set_raw_scale('data', 255)
        self.transformer.set_channel_swap('data', (2, 1, 0))

        self.conf_thresh = conf_thresh
        self.img_max_side = img_max_side

    def set_cfgs(self, img_max_side, conf_thresh):
        """
        设置相关参数， 在detect前调用
        Args:
            img_max_side: 给网络输入的图像的最长边 
            conf_thresh: 置信概率的阈值

        Returns:

        """
        self.conf_thresh = conf_thresh
        self.img_max_side = img_max_side

    def detect(self, img_arr):
        """
        detect faces on an image
        Args:
            img_arr: input， np.array，rgb图像，

        Returns:
            out： 检测结果 [[x1,y1,x2,y2, confidence], ... ]
        """
        tic = time.time()
        heigh = img_arr.shape[0]
        width = img_arr.shape[1]

        im_shrink = self.img_max_side / max(img_arr.shape[0], img_arr.shape[1])
        im_shrink = im_shrink if im_shrink < 1 else 1
        image = cv2.resize(img_arr, None, None, fx=im_shrink, fy=im_shrink, interpolation=cv2.INTER_LINEAR)

        # print("in size: ", image.shape)
        # print("data shape: ", self.net.blobs['data'].data.shape)
        self.net.blobs['data'].reshape(1, 3, image.shape[0], image.shape[1])
        # print("data shape: ", self.net.blobs['data'].data.shape)

        self.transformer.inputs['data'] = self.net.blobs['data'].data.shape
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

        detections_out[:, 1:5] = np.maximum(detections_out[:, 1:5], 0)
        detections_out[:, 1:5] = np.minimum(detections_out[:, 1:5], 1)

        # remove extremely small faces of 0 width or height, which possibly are mistakes
        need_rm_idx = [id for id, x in enumerate(detections_out[:, 1:]) if x[2] - x[0] <= 0 or x[3] - x[1] <= 0]
        detections_out = np.delete(detections_out, need_rm_idx, 0)
        detections_out[:, (1, 3)] *= width
        detections_out[:, (2, 4)] *= heigh

        # print("total time: ", time.time() - tic)
        # [score, x1,y1,x2,y2] -> [x1,y1,x2,y2, score]
        a = detections_out[:, 1:]
        b = detections_out[:, 0]
        out = np.hstack((a, b.reshape(a.shape[0], 1)))
        return out

    def detect_batch(self, imgs):
        pass


if __name__ == '__main__':
    pass
