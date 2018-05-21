#!/usr/bin/env python
# encoding: utf-8

"""
@version: ??
@author: r.li
@license: Apache Licence 
@contact: r.li@bmi-tech.com
@site: 
@software: PyCharm
@file: sfd_api.py
@time: 18-5-21 下午5:55
"""
import cv2

from sfd import SFD
import numpy as np
import os


def func():
    pass


class SDF_API:
    def __init__(self, model_path):
        self.detector = SFD(os.path.join(model_path, "deploy.prototxt"),
                            os.path.join(model_path, "SFD.caffemodel"))
        self.dets = None
        self.img = None
        self.pid = 0

    def draw_detect_result(self):
        for line in self.dets:
            score, x1, y1, x2, y2 = line
            # print("bbox2 ", x1, y1, x2, y2)
            cv2.rectangle(self.img, (x1, y1), (x2, y2), (0, 255, 0), 4)
            cv2.putText(self.img, "score: {:.2f}, size: {:.2f} x {:.2f}".format(score, x2 - x1, y2 - y1), (x1, y1),
                        cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 0, 0), 2)

        cv2.imshow("out", cv2.cvtColor(self.img.astype(np.uint8), cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)

    def process_image(self, image_file):
        pass

    def process_video(self, video_file):
        cap = cv2.VideoCapture()
        r = cap.open(video_file)

        # fp = open("sfd_detects.txt", 'w')

        if not r:
            return

        while True:
            r, image = cap.read()
            if not r:
                break
            self.img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.dets = self.detector.detect(self.img)
            self.draw_detect_result()

            # 将检测结果按"frameId faceCnt x1 y1 w h x1 y1 w h ... "的格式写入文档
            # bboxs = self.dets[:, 1:]
            # bboxs[:, 2:] = bboxs[:, 2:] - bboxs[:, 0:2]
            #
            # rm_idx = [id for id, x in enumerate(bboxs) if x[2] <= 0 or x[3] <= 0]
            # bboxs = np.delete(bboxs, rm_idx, 0)
            # out_str = "{} {}".format(self.pid, bboxs.shape[0])
            # if bboxs.shape[0] > 0:
            #     out_str += " " + " ".join(str(int(v)) for bbox in bboxs for v in bbox)
            # out_str += "\n"
            # print(out_str)
            # fp.write(out_str)

            self.pid += 1

        # fp.close()


if __name__ == '__main__':
    api = SDF_API("./models/VGGNet/WIDER_FACE/SFD_trained")
    api.process_video("/media/lirui/Program/Datas/Videos/Face201701052.mp4")
