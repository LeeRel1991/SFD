#!/usr/bin/env python
# encoding: utf-8

"""
@version: ??
@author: r.li
@license: Apache Licence 
@contact: r.li@bmi-tech.com
@site: 
@software: PyCharm
@file: cv_plot.py
@time: 18-6-18 上午10:55
"""

import cv2
import numpy as np

def plot_detections(image, dets):
    """
    在图片上绘制检测的矩形框，及confidence
    Args:
        image: 
        dets: 

    Returns:
        
    """
    for line in dets:
        score = line[4]
        x1, y1, x2, y2 = list(map(int, line[:4]))
        # print("bbox ", x1, y1, x2, y2)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 4)
        cv2.rectangle(image, (x1, y1), (x2+150, y1-20), (0, 255, 0), -1)
        cv2.putText(image, "{} x {} @{:.2f}".format(x2 - x1, y2 - y1, score), (x1, y1),
                    cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 0), 2)



class Main():
    def __init__(self):
        pass


if __name__ == '__main__':
    pass
