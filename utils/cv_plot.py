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


def plot_detections(image, dets):
    """
    在图片上绘制检测的矩形框，及confidence
    Args:
        image: 
        dets: 

    Returns:
        
    """
    for line in dets:
        x1, y1, x2, y2, score = line
        # print("bbox2 ", x1, y1, x2, y2)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 4)
        cv2.putText(image, "score: {:.2f}, size: {:.2f} x {:.2f}".format(score, x2 - x1, y2 - y1), (x1, y1),
                    cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 0, 0), 2)



class Main():
    def __init__(self):
        pass


if __name__ == '__main__':
    pass
