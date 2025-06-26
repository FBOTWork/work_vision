#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
from ultralytics import YOLO
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from PIL import Image as PILImage
import rospkg
import os
from work_vision.msgs import Description, Recognitions

class YoloV8Node:
    def __init__(self):
        rospy.loginfo("Inicializando o nó YOLOv8...")

        # Lê os parâmetros do arquivo YAML
        self.image_topic = rospy.get_param("~image_topic", "/camera/color/image_raw")
        self.recog_img_topic = rospy.get_param("~recog_image_topic", "/recog_imgs")
        self.recog_info_topic = rospy.get_param("~recog_info_topic", "/recog_info")
        self.threshold = rospy.get_param("~threshold", 0.5)
        model_filename = rospy.get_param("~model_file", "yolov8n.pt")

        # Monta o caminho completo para o modelo
        rospack = rospkg.RosPack()
        package_path = rospack.get_path("work_vision")
        self.model_path = os.path.join(package_path, "weights", model_filename)

        # Inicializa YOLOv8
        self.model = YOLO(self.model_path)
        self.model.conf = self.threshold
        self.names = self.model.names

        # CV Bridge
        self.bridge = CvBridge()

        # Publishers
        self.pub_img = rospy.Publisher(self.recog_img_topic, Image, queue_size=1)
        self.pub_info = rospy.Publisher(self.recog_info_topic, String, queue_size=1)

        # Subscriber
        self.sub = rospy.Subscriber(self.image_topic, Image, self.callback)

    def callback(self, data):
        rospy.loginfo("Imagem recebida para detecção.")

        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')
        except Exception as e:
            rospy.logerr(f"Erro ao converter imagem: {e}")
            return

        results = self.model(cv_image)

        for r in results:
            for c in r.boxes.cls:
                label = self.names[int(c)]
                self.pub_info.publish(label)
                rospy.loginfo(f"Objeto detectado: {label}")

            im_array = r.plot()
            im_bgr = im_array[:, :, ::-1]
            img_msg = self.bridge.cv2_to_imgmsg(im_bgr, encoding="bgr8")
            self.pub_img.publish(img_msg)

    def run(self):
        rospy.spin()

if __name__ == "__main__":
    rospy.init_node("object_recognition_node", anonymous=True)
    yolo_node = YoloV8Node()
    yolo_node.run()