#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge

class DetectorDeFaixasLaranjaEAmarela:
    def __init__(self):
        rospy.init_node('detector_de_faixas_laranja_amarela', anonymous=True)

        self.last_print_time = rospy.Time.now()

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

        # Range HSV para amarelo
        self.lower_yellow = np.array([18, 80, 80])
        self.upper_yellow = np.array([32, 255, 255])


        # Ranges HSV para detectar tons de laranja
        self.lower_orange_normal = np.array([0, 100, 100])
        self.upper_orange_normal = np.array([10, 255, 255])
        self.lower_orange_highlight = np.array([5, 100, 200])
        self.upper_orange_highlight = np.array([15, 200, 255])

        self.min_area = 20
        self.max_area = 60000

        rospy.loginfo("Detector de faixas laranja e amarela iniciado.")
        rospy.spin()

    def depth_callback(self, msg):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            rospy.logerr("Erro ao converter imagem de profundidade: {}".format(e))

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            rospy.logerr("Erro ao converter imagem: {}".format(e))
            return

        if self.depth_image is None:
            rospy.logwarn("Imagem de profundidade ainda não disponível.")
            return

        output_image = frame.copy()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Máscaras de cor para laranja
        mask_orange_normal = cv2.inRange(hsv, self.lower_orange_normal, self.upper_orange_normal)
        mask_orange_highlight = cv2.inRange(hsv, self.lower_orange_highlight, self.upper_orange_highlight)
        mask_orange = cv2.bitwise_or(mask_orange_normal, mask_orange_highlight)

        # Máscara de cor para amarelo
        mask_yellow = cv2.inRange(hsv, self.lower_yellow, self.upper_yellow)

        # Combina máscaras de laranja e amarelo
        mask = cv2.bitwise_or(mask_orange, mask_yellow)

        # Operações morfológicas
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Contornos
        _, contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        coords_centers = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_area < area < self.max_area:
                rect = cv2.minAreaRect(contour)
                (cx, cy), (w, h), angle = rect

                # FILTRO para ignorar objetos muito estreitos ou pequenos
                min_dim = min(w, h)
                max_dim = max(w, h)

                if min_dim < 5:  # muito estreito (em pixels), ignora
                    continue
                if max_dim / min_dim > 15:  # razão de aspecto muito alta, ignora
                    continue

                box = cv2.boxPoints(rect)
                box = np.int32(box)
                cv2.drawContours(output_image, [box], 0, (0, 255, 0), 2)

                cx, cy = int(cx), int(cy)

                if 0 <= cy < self.depth_image.shape[0] and 0 <= cx < self.depth_image.shape[1]:
                    z = float(self.depth_image[cy, cx])
                else:
                    z = 0.0

                coords_centers.append((cx, cy, z))
                cv2.circle(output_image, (cx, cy), 5, (0, 255, 255), -1)


        flat_coords = []
        current_time = rospy.Time.now()

        if len(coords_centers) >= 2:
            max_dist = 0
            pt1 = pt2 = None
            for i in range(len(coords_centers)):
                for j in range(i + 1, len(coords_centers)):
                    x1, y1, _ = coords_centers[i]
                    x2, y2, _ = coords_centers[j]
                    dist = np.hypot(x2 - x1, y2 - y1)
                    if dist > max_dist:
                        max_dist = dist
                        pt1 = coords_centers[i]
                        pt2 = coords_centers[j]
            if pt1 and pt2:
                flat_coords.extend([float(pt1[0]), float(pt1[1]), float(pt1[2]),
                                    float(pt2[0]), float(pt2[1]), float(pt2[2])])
                cv2.line(output_image, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), (255, 0, 0), 3)
                if (current_time - self.last_print_time).to_sec() > 0.5:
                    rospy.loginfo("Extremos (x, y, z): [{:.2f}, {:.2f}, {:.3f}] <-> [{:.2f}, {:.2f}, {:.3f}]".format(
                        pt1[0], pt1[1], pt1[2], pt2[0], pt2[1], pt2[2]))
                    self.last_print_time = current_time
        elif len(coords_centers) == 1:
            x, y, z = coords_centers[0]
            flat_coords.extend([x, y, z])
            if (current_time - self.last_print_time).to_sec() > 0.5:
                rospy.loginfo("Coordenada (x, y, z): [{:.2f}, {:.2f}, {:.3f}]".format(x, y, z))
                self.last_print_time = current_time
        else:
            if (current_time - self.last_print_time).to_sec() > 0.5:
                rospy.logwarn("Nenhuma fita detectada.")
                self.last_print_time = current_time

        coord_msg = Float32MultiArray()
        coord_msg.data = flat_coords
        self.pub_coords.publish(coord_msg)

        try:
            msg_out = self.bridge.cv2_to_imgmsg(output_image, encoding="bgr8")
            self.pub_image.publish(msg_out)
        except Exception as e:
            rospy.logerr("Erro ao publicar imagem: {}".format(e))

if __name__ == '__main__':
    try:
        DetectorDeFaixasLaranjaEAmarela()
    except rospy.ROSInterruptException:
        pass