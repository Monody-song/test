
#encoding:utf-8
from __future__ import division

import numpy

'''
功能： 人脸识别摄像头视频流数据实时检测模块
'''

from PIL import Image, ImageDraw, ImageFont
import os
import cv2
from train_model import Model

threshold=0.7  # 如果模型认为概率高于70%则显示为模型中已有的人物



# 新的名字列表
new_names = ["张三", "李四"]






# 解决cv2.putText绘制中文乱码
def cv2ImgAddText(img2, text, left, top, textColor=(0, 0, 255), textSize=20):
    if isinstance(img2, numpy.ndarray):  # 判断是否OpenCV图片类型
        img2 = Image.fromarray(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img2)
    # 字体的格式
    fontStyle = ImageFont.truetype(r"C:\WINDOWS\FONTS\MSYH.TTC", textSize, encoding="utf-8")
    # 绘制文本
    draw.text((left, top), text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(numpy.asarray(img2), cv2.COLOR_RGB2BGR)


class Camera_reader(object):
    def __init__(self):
        self.model=Model()
        self.model.load()
        self.img_size=128


    def build_camera(self):
        '''
        调用摄像头来实时人脸识别
        '''
        face_cascade = cv2.CascadeClassifier('config/haarcascade_frontalface_alt.xml')

        cameraCapture=cv2.VideoCapture(0)
        success, frame=cameraCapture.read()
        while success and cv2.waitKey(1)==-1:
            success,frame=cameraCapture.read()
            gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces=face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in faces:
                ROI=gray[x:x+w,y:y+h]
                ROI=cv2.resize(ROI, (self.img_size, self.img_size),interpolation=cv2.INTER_LINEAR)
                label,prob=self.model.predict(ROI)
                print(label)

                if prob > threshold:
                    show_name = new_names[label]
                else:
                    show_name = "陌生人"
               # cv2.putText(frame, show_name, (x,y-20),cv2.FONT_HERSHEY_SIMPLEX,1,255,2)
                # 在图像上绘制中文字符
                # 解决cv2.putText绘制中文乱码
                frame = cv2ImgAddText(frame, show_name, x + 5, y - 30,)

                frame=cv2.rectangle(frame,(x,y), (x+w,y+h),(255,0,0),2)
            cv2.imshow("Camera", frame)
        else:
            cameraCapture.release()
            cv2.destroyAllWindows()


if __name__ == '__main__':
    camera=Camera_reader()
    camera.build_camera()


