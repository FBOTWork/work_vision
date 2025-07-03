#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class DetectorDeFaixasLaranja:
    def __init__(self):
        rospy.init_node('detector_de_faixas_laranja', anonymous=True)

        # Inicializa ponte entre ROS e OpenCV
        self.bridge = CvBridge()

        # Publisher da imagem processada
        self.pub = rospy.Publisher('/fita_zebrada/detected_image', Image, queue_size=10)

        # Subscriber da imagem da câmera
        rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback)

        # Ranges HSV para detectar tons de laranja
        self.lower_orange_normal = np.array([0, 100, 100])
        self.upper_orange_normal = np.array([10, 255, 255])

        self.lower_orange_highlight = np.array([5, 100, 200])
        self.upper_orange_highlight = np.array([15, 200, 255])

        self.min_area = 20
        self.max_area = 60000

        rospy.loginfo("Detector de faixas laranja iniciado.")
        rospy.spin()

    def image_callback(self, msg):
        try:
            # Converte a imagem ROS para OpenCV
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            rospy.logerr("Erro ao converter imagem: {}".format(e))
            return

        output_image = frame.copy()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Máscaras de cor
        mask_normal = cv2.inRange(hsv, self.lower_orange_normal, self.upper_orange_normal)
        mask_highlight = cv2.inRange(hsv, self.lower_orange_highlight, self.upper_orange_highlight)
        mask = cv2.bitwise_or(mask_normal, mask_highlight)

        # Operações morfológicas
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Contornos
        contour_result = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contour_result[1] if len(contour_result) == 3 else contour_result[0]

        coords_centers = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_area < area < self.max_area:
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                box = np.int32(box)
                cv2.drawContours(output_image, [box], 0, (0, 255, 0), 2)

                (cx, cy), _, _ = rect
                cx = int(cx)
                cy = int(cy)
                coords_centers.append((cx, cy))
                cv2.circle(output_image, (cx, cy), 5, (0, 255, 255), -1)

        if len(coords_centers) > 1:
            coords_centers.sort(key=lambda p: (p[0], p[1]))
            pt1 = coords_centers[0]
            for pt2 in coords_centers[1:]:
                cv2.line(output_image, pt1, pt2, (255, 0, 0), 3, lineType=cv2.LINE_AA)
                pt1 = pt2

        # Publica a imagem processada no ROS
        try:
            msg_out = self.bridge.cv2_to_imgmsg(output_image, encoding="bgr8")
            self.pub.publish(msg_out)
        except Exception as e:
            rospy.logerr("Erro ao publicar imagem: {}".format(e))

if __name__ == '__main__':
    try:
        DetectorDeFaixasLaranja()
    except rospy.ROSInterruptException:
        pass
