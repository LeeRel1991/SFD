#!/usr/bin/env python
# encoding: utf-8

"""
@version: ??
@author: r.li
@license: Apache Licence 
@contact: r.li@bmi-tech.com
@site: 
@software: PyCharm
@file: demo.py
@time: 18-5-16 下午2:57
"""
from glob import glob

import numpy as np
import cv2

# Make sure that caffe is on the python path:
import skimage

caffe_ssd_root = '/home/lirui/packages/caffe_ssd'  # this file is expected to be in {sfd_root}/sfd_test_code/AFW
import os

# os.chdir(caffe_ssd_root)
import sys

sys.path.insert(0, os.path.join(caffe_ssd_root, 'python'))
import caffe

caffe.set_device(0)
caffe.set_mode_gpu()
model_def = './models/VGGNet/WIDER_FACE/SFD_trained/deploy.prototxt'
model_weights = './models/VGGNet/WIDER_FACE/SFD_trained/SFD.caffemodel'
sfd = caffe.Net(model_def, model_weights, caffe.TEST)
conf_thresh = 0.8

filter1_time = 0
filter2_time = 0
cnt = 0

import time


def handle_image(net, image_src):
    tic = time.time()
    # image_src = image_src.astype(np.float32)

    print("resize time: ", time.time() - tic)
    heigh = image_src.shape[0]
    width = image_src.shape[1]

    im_shrink = 480.0 / max(image_src.shape[0], image_src.shape[1])
    im_shrink = im_shrink if im_shrink < 1 else 1
    image = cv2.resize(image_src, None, None, fx=im_shrink, fy=im_shrink, interpolation=cv2.INTER_LINEAR)

    # print("in size: ", image.shape)
    net.blobs['data'].reshape(1, 3, image.shape[0], image.shape[1])
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_mean('data', np.array([104, 117, 123]))
    transformer.set_raw_scale('data', 255)
    transformer.set_channel_swap('data', (2, 1, 0))
    transformed_image = transformer.preprocess('data', image.astype(np.float32))

    print("preprocess time: ", time.time() - tic)
    net.blobs['data'].data[...] = transformed_image

    detections = net.forward()['detection_out']

    print("forward time: ", time.time() - tic)

    # filer by confidence
    detections_out = detections[0, 0, :, 2:7]
    detections_out = detections_out[detections_out[:, 0].argsort()[::-1], :]
    thresh_pos = np.where(detections_out[:, 0] >= conf_thresh)[0]
    detections_out = detections_out[thresh_pos, :]

    detections_out[:, 1:3] = np.maximum(detections_out[:, 1:3], 0)
    detections_out[:, 3:5] = np.minimum(detections_out[:, 3:5], 1)
    detections_out[:, (1, 3)] *= width
    detections_out[:, (2, 4)] *= heigh

    bbox_width = detections_out[:, 3] - detections_out[:, 1]
    bbox_height = detections_out[:, 4] - detections_out[:, 2]

    toc = time.time()
    print("total time: ", toc - tic)

    if len(bbox_height):
        print("min width: {:.2f}, min height: {:.2f}".format(min(bbox_width), min(bbox_height)))
    for line in detections_out:
        score, x1, y1, x2, y2 = line
        # print("bbox2 ", x1, y1, x2, y2)
        cv2.rectangle(image_src, (x1, y1), (x2, y2), (0, 255, 0), 4)
        cv2.putText(image_src, "score: {:.2f}, size: {:.2f} x {:.2f}".format(score, x2 - x1, y2 - y1), (x1, y1),
                    cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 0, 0), 2)

    cv2.imshow("out", cv2.cvtColor(image_src.astype(np.uint8), cv2.COLOR_RGB2BGR))
    cv2.waitKey(1)


def demo_image_folder(image_folder):
    types = ('*.jpg', '*.png', '*.jpeg')
    image_path_list = []
    for files in types:
        image_path_list.extend(glob(os.path.join(image_folder, files)))

    for filename in image_path_list:
        # read image
        print("image file: ", filename)
        image = cv2.imread(filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        handle_image(sfd, image)


def demo_video(videofile):
    cap = cv2.VideoCapture()
    r = cap.open(videofile)

    if not r:
        return

    while True:
        r, img = cap.read()
        if not r:
            break
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        handle_image(sfd, img)


def demo_path(imgpath):
    pass


if __name__ == '__main__':
    # demo_image_folder("/media/lirui/Personal/DeepLearning/FaceRec/PRNet/TestImages/test")
    # handle_image(sfd, "/media/lirui/Personal/DeepLearning/FaceRec/PRNet/TestImages/test/timg7.jpeg")
    demo_video("/media/lirui/Program/Datas/Videos/Face201701052.mp4")
