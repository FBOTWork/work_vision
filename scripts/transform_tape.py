#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
import tf
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import PointStamped
from visualization_msgs.msg import Marker
from cv_bridge import CvBridge

class DetectorDeFaixasLaranja:
    def __init__(self):
        rospy.init_node('detector_de_faixas_laranja', anonymous=True)

        self.last_print_time = rospy.Time.now()

        self.bridge = CvBridge()
        self.tf_listener = tf.TransformListener()

        self.pub_image = rospy.Publisher('/fita_zebrada/detected_image', Image, queue_size=10)
        self.pub_coords = rospy.Publisher('/fita_zebrada/centros', Float32MultiArray, queue_size=10)
        self.marker_pub = rospy.Publisher('/fita_zebrada/obstaculo_virtual', Marker, queue_size=10)

        rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback)
        self.depth_image = None
        rospy.Subscriber('/camera/depth/image_rect_raw', Image, self.depth_callback)

        self.lower_orange_normal = np.array([0, 100, 100])
        self.upper_orange_normal = np.array([10, 255, 255])
        self.lower_orange_highlight = np.array([5, 100, 200])
        self.upper_orange_highlight = np.array([15, 200, 255])

        self.min_area = 20
        self.max_area = 60000

        self.fx = 617.0
        self.fy = 617.0
        self.cx = 320.0
        self.cy = 240.0

        rospy.loginfo("Detector de faixas laranja iniciado.")
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

        mask_normal = cv2.inRange(hsv, self.lower_orange_normal, self.upper_orange_normal)
        mask_highlight = cv2.inRange(hsv, self.lower_orange_highlight, self.upper_orange_highlight)
        mask = cv2.bitwise_or(mask_normal, mask_highlight)

        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contour_result = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contour_result[1] if len(contour_result) == 3 else contour_result[0]

        coords_centers = []
        flat_coords = []
        current_time = rospy.Time.now()

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
                    z = float(self.depth_image[cy, cx]) / 1000.0
                else:
                    z = 0.0

                if z == 0 or np.isnan(z):
                    continue

                x = (cx - self.cx) * z / self.fx
                y = (cy - self.cy) * z / self.fy

                ponto_camera = PointStamped()
                ponto_camera.header.stamp = rospy.Time(0)
                ponto_camera.header.frame_id = "camera_depth_optical_frame"
                ponto_camera.point.x = x
                ponto_camera.point.y = y
                ponto_camera.point.z = z

                try:
                    ponto_mapa = self.tf_listener.transformPoint("map", ponto_camera)
                    rospy.loginfo("Obstáculo virtual em /map: x=%.2f y=%.2f z=%.2f",
                                  ponto_mapa.point.x, ponto_mapa.point.y, ponto_mapa.point.z)
                    self.publicar_marker(ponto_mapa)
                except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
                    rospy.logwarn("Erro ao transformar ponto: {}".format(e))
                    continue

                coords_centers.append((cx, cy, z))
                cv2.circle(output_image, (cx, cy), 5, (0, 255, 255), -1)

        coord_msg = Float32MultiArray()
        coord_msg.data = flat_coords
        self.pub_coords.publish(coord_msg)

        try:
            msg_out = self.bridge.cv2_to_imgmsg(output_image, encoding="bgr8")
            self.pub_image.publish(msg_out)
        except Exception as e:
            rospy.logerr("Erro ao publicar imagem: {}".format(e))

    def publicar_marker(self, ponto_mapa):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "fita"
        marker.id = int(rospy.Time.now().to_sec() * 1000) % 100000
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position = ponto_mapa.point
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        marker.color.r = 1.0
        marker.color.g = 0.5
        marker.color.b = 0.0
        marker.color.a = 1.0
        marker.lifetime = rospy.Duration(30.0)
        self.marker_pub.publish(marker)

if __name__ == '__main__':
    try:
        DetectorDeFaixasLaranja()
    except rospy.ROSInterruptException:
        pass
