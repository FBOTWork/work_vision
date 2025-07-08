#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import Float32MultiArray, Header
from cv_bridge import CvBridge
import sensor_msgs.point_cloud2 as pc2
from geometry_msgs.msg import PointStamped
import tf
from visualization_msgs.msg import Marker

class DetectorDeFaixasLaranja:
    def __init__(self):
        rospy.init_node('detector_de_faixas_laranja', anonymous=True)

        self.last_print_time = rospy.Time.now()
        self.bridge = CvBridge()

        self.pub_image = rospy.Publisher('/fita_zebrada/detected_image', Image, queue_size=10)
        self.pub_coords = rospy.Publisher('/fita_zebrada/centros', Float32MultiArray, queue_size=10)
        self.pc_pub = rospy.Publisher('/virtual_obstacles', PointCloud2, queue_size=1)
        self.marker_pub = rospy.Publisher('/visualization_marker', Marker, queue_size=10)

        rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback)
        rospy.Subscriber('/camera/depth/image_rect_raw', Image, self.depth_callback)

        self.lower_orange_normal = np.array([0, 100, 100])
        self.upper_orange_normal = np.array([10, 255, 255])
        self.lower_orange_highlight = np.array([5, 100, 200])
        self.upper_orange_highlight = np.array([15, 200, 255])

        self.min_area = 20
        self.max_area = 60000

        self.depth_image = None
        self.tf_listener = tf.TransformListener()

        rospy.loginfo("Detector de faixas laranja iniciado.")
        rospy.spin()

    def depth_callback(self, msg):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            rospy.logerr("Erro ao converter imagem de profundidade: {}".format(e))

    def publicar_ponto_como_obstaculo(self, ponto):
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "map"
        points = [(ponto.x, ponto.y, ponto.z)]
        pc2_msg = pc2.create_cloud_xyz32(header, points)
        self.pc_pub.publish(pc2_msg)

    def publicar_marker(self, ponto):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "fita_zebrada"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position = ponto
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        marker.color.r = 1.0
        marker.color.g = 0.5
        marker.color.b = 0.0
        marker.color.a = 1.0
        self.marker_pub.publish(marker)

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            rospy.logerr("Erro ao converter imagem: {}".format(e))
            return

        if self.depth_image is None:
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

        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_area < area < self.max_area:
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                box = np.int32(box)
                cv2.drawContours(output_image, [box], 0, (0, 255, 0), 2)

                (cx, cy), _, _ = rect
                cx, cy = int(cx), int(cy)

                if 0 <= cy < self.depth_image.shape[0] and 0 <= cx < self.depth_image.shape[1]:
                    z = float(self.depth_image[cy, cx])
                else:
                    z = 0.0

                coords_centers.append((cx, cy, z))
                cv2.circle(output_image, (cx, cy), 5, (0, 255, 255), -1)

        if len(coords_centers) >= 1:
            cx, cy, z = coords_centers[0]
            ponto_camera = PointStamped()
            ponto_camera.header.frame_id = "camera_depth_optical_frame"
            ponto_camera.header.stamp = rospy.Time(0)
            ponto_camera.point.x = float(z)
            ponto_camera.point.y = 0.0
            ponto_camera.point.z = 0.0

            try:
                self.tf_listener.waitForTransform("map", ponto_camera.header.frame_id, rospy.Time(0), rospy.Duration(3.0))
                ponto_mapa = self.tf_listener.transformPoint("map", ponto_camera)
                self.publicar_ponto_como_obstaculo(ponto_mapa.point)
                self.publicar_marker(ponto_mapa.point)
                rospy.loginfo("Fita projetada em map: x={:.2f}, y={:.2f}, z={:.2f}".format(
                    ponto_mapa.point.x, ponto_mapa.point.y, ponto_mapa.point.z))
            except Exception as e:
                rospy.logwarn("Erro ao transformar ponto: {}".format(e))

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
