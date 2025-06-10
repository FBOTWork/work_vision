#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from std_msgs.msg import String
from ultralytics import YOLO
import cv2
import math 
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from PIL import Image as IMG
import numpy as np

bridge = CvBridge()

pub = rospy.Publisher('recog_imgs', Image, queue_size=1)
#cria um publisher que fica publicando mensagens no tópico 'recog_imgs' do tipo image

pub2 = rospy.Publisher('recog_info', String, queue_size=1)
#cria um publisher que fica publicando mensagens no tópico 'recog_info' do tipo string

model = YOLO("/home/jetson/main_ws/src/work_vision/weights/yolov8n.pt")
names = model.names


def reconhecimento(data):
   #função que vai ser chamada sempre que uma imagem for recebida

   rospy.loginfo(rospy.get_caller_id() + "I received an image")
   #informa que uma imagem foi recebida 

   cv_image = bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
   #converte a mensagem de imagem ROS recebida pela camera em imagem OpenCV

   results = model(cv_image)
   #a imagem é passada para o YOLO para que seja feita a detecção

   detect_list = []
   for r in results:
      for c in r.boxes.cls:
        pub2.publish(names[int(c)])
        #publica cada classe detectada no tópipco recog_info

      im_array = r.plot()
      im = IMG.fromarray(im_array[..., ::-1])

      img_written = bridge.cv2_to_imgmsg(np.array(im), encoding="bgr8")
      #converte a imagem PIL para a mensagem de imagem ROS.

      pub.publish(img_written)
      #a imagem com as bounding box é publicada no tópico "recog_imgs"

def listener():
   rospy.loginfo("listener")
   
   
   rospy.init_node('listener', anonymous=True)

   rospy.loginfo("node initiated")
   

   rospy.Subscriber("/camera/color/image_raw", Image, reconhecimento)
   #cria um subscriber que irá receber as imagens

   rospy.spin()
   #cria um loop que só é encerrado ao matar a aplicação ROS

if __name__ == '__main__':
   listener()