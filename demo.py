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

import os

# os.chdir(caffe_ssd_root)
import sys


from sfd.sfd import SFD

# caffe.set_mode_cpu()
sfd = SFD("./data/SFD_deploy.prototxt",
          "./data/SFD_weights.caffemodel")
conf_thresh = 0.8

filter1_time = 0
filter2_time = 0
cnt = 0

import time


def handle_image(net, bgr_img):
    tic = time.time()
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    dets = sfd.detect(rgb_img)

    toc = time.time()
    print("total time: ", toc - tic)

    bbox_width = dets[:, 2] - dets[:, 0]
    bbox_height = dets[:, 3] - dets[:, 1]

    if len(bbox_height):
        print("min width: {:.2f}, min height: {:.2f}".format(min(bbox_width), min(bbox_height)))

    # ---------- optional Plot
    from sfd.utils.cv_plot import plot_detections
    plot_detections(bgr_img, dets)
    cv2.imshow("out", bgr_img)
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
        handle_image(sfd, img)


if __name__ == '__main__':
    print(sys.argv)
    if os.path.isfile(sys.argv[1]):
        demo_video(sys.argv[1])
    else:
        demo_image_folder(sys.argv[1])
    # demo_image_folder("/media/lirui/Personal/DeepLearning/FaceRec/PRNet/TestImages/test")
    # demo_video("/media/lirui/Program/Datas/Videos/Face201701052.mp4")
