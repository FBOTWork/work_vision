#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge

class DetectorMesaZeroFitaVerde:
    def __init__(self):
        rospy.init_node('detector_mesa_zero_fita_verde', anonymous=True)

        self.last_print_time = rospy.Time.now()

        self.bridge = CvBridge()

        self.pub_image = rospy.Publisher('/fita_zebrada/detected_image', Image, queue_size=10)
        self.pub_coords = rospy.Publisher('/fita_zebrada/centros', Float32MultiArray, queue_size=10)

        rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback)
        self.depth_image = None
        rospy.Subscriber('/camera/depth/image_rect_raw', Image, self.depth_callback)

        self.lower_green = np.array([55, 60, 95])
        self.upper_green = np.array([85, 255, 255])

        self.min_area = 20
        self.max_area = 60000

        rospy.loginfo("Detector de mesa zero com fita verde iniciado.")
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

        mask = cv2.inRange(hsv, self.lower_green, self.upper_green)

        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

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

                if 0 <= cy < self.depth_image.shape[0] and 0 <= cx < self.depth_image.shape[1]:
                    z = float(self.depth_image[cy, cx])
                else:
                    z = 0.0

                coords_centers.append((cx, cy, z))
                cv2.circle(output_image, (cx, cy), 5, (0, 255, 255), -1)

        current_time = rospy.Time.now()

        if len(coords_centers) >= 4:
            # Usa os 4 maiores contornos para formar retângulo
            coords_centers.sort(key=lambda p: (p[0], p[1]))
            pts = np.array([(int(p[0]), int(p[1])) for p in coords_centers[:4]])
            x, y, w, h = cv2.boundingRect(pts)

            # Desenha o retângulo
            cv2.rectangle(output_image, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Calcula centro geométrico
            center_x = x + w // 2
            center_y = y + h // 2

            # Obtém z
            if 0 <= center_y < self.depth_image.shape[0] and 0 <= center_x < self.depth_image.shape[1]:
                center_z = float(self.depth_image[center_y, center_x])
            else:
                center_z = 0.0

            # Desenha o centro
            cv2.circle(output_image, (center_x, center_y), 7, (255, 0, 255), -1)

            # Publica coordenada
            coord_msg = Float32MultiArray()
            coord_msg.data = [float(center_x), float(center_y), float(center_z)]
            self.pub_coords.publish(coord_msg)

            if (current_time - self.last_print_time).to_sec() > 0.5:
                rospy.loginfo("Centro da mesa zero: [{:.2f}, {:.2f}, {:.3f}]".format(center_x, center_y, center_z))
                self.last_print_time = current_time

        else:
            if (current_time - self.last_print_time).to_sec() > 0.5:
                rospy.logwarn("Fitas insuficientes para formar retângulo (detectadas: {}).".format(len(coords_centers)))
                self.last_print_time = current_time

        try:
            msg_out = self.bridge.cv2_to_imgmsg(output_image, encoding="bgr8")
            self.pub_image.publish(msg_out)
        except Exception as e:
            rospy.logerr("Erro ao publicar imagem: {}".format(e))

if __name__ == '__main__':
    try:
        DetectorMesaZeroFitaVerde()
    except rospy.ROSInterruptException:
        pass
