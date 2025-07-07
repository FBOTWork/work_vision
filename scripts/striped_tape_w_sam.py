#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge

class DetectorDeFaixasLaranja:
    def __init__(self):
        rospy.init_node('detector_de_faixas_laranja', anonymous=True)

        # Inicializa ponte entre ROS e OpenCV
        self.bridge = CvBridge()

        # Publisher da imagem processada
        self.pub_image = rospy.Publisher('/fita_zebrada/detected_image', Image, queue_size=10)

        # Publisher das coordenadas dos centros detectados (x, y, z)
        self.pub_coords = rospy.Publisher('/fita_zebrada/centros', Float32MultiArray, queue_size=10)

        # Subscriber da imagem RGB da câmera
        rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback)

        # Subscriber da imagem de profundidade
        self.depth_image = None
        rospy.Subscriber('/camera/depth/image_rect_raw', Image, self.depth_callback)

        # Ranges HSV para detectar tons de laranja
        self.lower_orange_normal = np.array([0, 100, 100])
        self.upper_orange_normal = np.array([10, 255, 255])
        self.lower_orange_highlight = np.array([5, 100, 200])
        self.upper_orange_highlight = np.array([15, 200, 255])

        self.min_area = 20
        self.max_area = 60000

        rospy.loginfo("Detector de faixas laranja iniciado.")
        rospy.spin()

    def depth_callback(self, msg):
        try:
            # Converte imagem de profundidade ROS para OpenCV (tipo float32 em metros)
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            rospy.logerr("Erro ao converter imagem de profundidade: {}".format(e))

    def image_callback(self, msg):
        try:
            # Converte a imagem ROS para OpenCV
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            rospy.logerr("Erro ao converter imagem: {}".format(e))
            return

        if self.depth_image is None:
            rospy.logwarn("Imagem de profundidade ainda não disponível.")
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

                # Obtém o valor de profundidade (z) no ponto (cx, cy)
                if 0 <= cy < self.depth_image.shape[0] and 0 <= cx < self.depth_image.shape[1]:
                    z = float(self.depth_image[cy, cx])
                else:
                    z = 0.0  # Profundidade inválida, define como 0

                coords_centers.append((cx, cy, z))
                cv2.circle(output_image, (cx, cy), 5, (0, 255, 255), -1)

        # Publica as coordenadas (x, y, z) no tópico ROS
        flat_coords = []
        for (x, y, z) in coords_centers:
            flat_coords.extend([float(x), float(y), z])

        coord_msg = Float32MultiArray()
        coord_msg.data = flat_coords
        self.pub_coords.publish(coord_msg)

        # Desenha linhas entre os centros detectados
        if len(coords_centers) > 1:
            coords_centers.sort(key=lambda p: (p[0], p[1]))
            pt1 = (int(coords_centers[0][0]), int(coords_centers[0][1]))
            for pt in coords_centers[1:]:
                pt2 = (int(pt[0]), int(pt[1]))
                cv2.line(output_image, pt1, pt2, (255, 0, 0), 3, lineType=cv2.LINE_AA)
                pt1 = pt2

        # Publica a imagem processada
        try:
            msg_out = self.bridge.cv2_to_imgmsg(output_image, encoding="bgr8")
            self.pub_image.publish(msg_out)
        except Exception as e:
            rospy.logerr("Erro ao publicar imagem: {}".format(e))

if __name__ == '__main__':
    try:
        DetectorDeFaixasLaranja()
    except rospy.ROSInterruptException:
        pass
