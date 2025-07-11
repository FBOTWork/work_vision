#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Point, PoseStamped
from visualization_msgs.msg import Marker # Nova importacao para Marker
from std_msgs.msg import ColorRGBA # Para definir a cor do Marker

import numpy as np
import os
import sys

import message_filters

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from container_detector_red_py2 import detect_and_segment_red_container_py2

class ContainerDetectorNode:
    def __init__(self):
        rospy.init_node('container_detector_red_node', anonymous=True)

        self.bridge = CvBridge()
        self.intrinsics = None

        self.camera_info_sub = rospy.Subscriber("/camera/color/camera_info", CameraInfo, self.camera_info_callback)
        rospy.loginfo("Subscrito ao topico /camera/color/camera_info para informacoes da camera")

        image_sub = message_filters.Subscriber("/camera/color/image_raw", Image)
        depth_sub = message_filters.Subscriber("/camera/depth/image_rect_raw", Image)

        self.ts = message_filters.ApproximateTimeSynchronizer([image_sub, depth_sub], queue_size=10, slop=0.1)
        self.ts.registerCallback(self.image_depth_callback)
        rospy.loginfo("Subscrito e sincronizando topicos /camera/color/image_raw e /camera/depth/image_rect_raw")

        self.result_image_pub = rospy.Publisher("/container_detector_red/result_image", Image, queue_size=1)
        rospy.loginfo("Publicando resultados em /container_detector_red/result_image")

        self.centroid_2d_pub = rospy.Publisher("/container_detector_red/centroid_2d", Point, queue_size=1)
        rospy.loginfo("Publicando centroide 2D em /container_detector_red/centroid_2d")

        self.pose_3d_pub = rospy.Publisher("/container_detector_red/container_pose_3d", PoseStamped, queue_size=1)
        rospy.loginfo("Publicando pose 3D em /container_detector_red/container_pose_3d")

        # NOVO: Publicador para Marker
        self.marker_pub = rospy.Publisher("/container_detector_red/container_marker", Marker, queue_size=1)
        rospy.loginfo("Publicando Marker 3D em /container_detector_red/container_marker")

        self.marker_id = 0 # ID unico para o marker

    def camera_info_callback(self, data):
        self.intrinsics = np.array(data.K).reshape((3, 3))
        rospy.loginfo("Parametros intrinsecos da camera recebidos.")
        self.camera_info_sub.unregister()

    def image_depth_callback(self, image_data, depth_data):
        if self.intrinsics is None:
            rospy.logwarn("Aguardando parametros intrinsecos da camera...")
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(image_data, "bgr8")
        except CvBridgeError as e:
            #rospy.logerr("CvBridge Error na imagem colorida: %s", e)
            return

        try:
            if depth_data.encoding == "16UC1":
                depth_image = self.bridge.imgmsg_to_cv2(depth_data, "16UC1")
                depth_image = depth_image.astype(np.float32) / 1000.0
            elif depth_data.encoding == "32FC1":
                depth_image = self.bridge.imgmsg_to_cv2(depth_data, "32FC1")
            else:
                #rospy.logerr("Encoding de profundidade desconhecido: %s. Apenas 16UC1 e 32FC1 sao suportados.", depth_data.encoding)
                return
        except CvBridgeError as e:
            #rospy.logerr("CvBridge Error na imagem de profundidade: %s", e)
            return

        result_image, container_mask, centroid_2d_coords = detect_and_segment_red_container_py2(cv_image)

        if result_image is not None:
            try:
                self.result_image_pub.publish(self.bridge.cv2_to_imgmsg(result_image, "bgr8"))
            except CvBridgeError as e:
                rospy.logerr("CvBridge Error ao publicar imagem de resultado: %s", e)

            if centroid_2d_coords is not None:
                centroid_2d_msg = Point()
                centroid_2d_msg.x = float(centroid_2d_coords[0])
                centroid_2d_msg.y = float(centroid_2d_coords[1])
                centroid_2d_msg.z = 0.0
                self.centroid_2d_pub.publish(centroid_2d_msg)

                u = int(centroid_2d_coords[0])
                v = int(centroid_2d_coords[1])

                if 0 <= v < depth_image.shape[0] and 0 <= u < depth_image.shape[1]:
                    depth_value = depth_image[v, u]
                    if np.isfinite(depth_value) and depth_value > 0:
                        fx = self.intrinsics[0, 0]
                        fy = self.intrinsics[1, 1]
                        cx = self.intrinsics[0, 2]
                        cy = self.intrinsics[1, 2]

                        X_camera = (u - cx) * depth_value / fx
                        Y_camera = (v - cy) * depth_value / fy
                        Z_camera = depth_value

                        pose_3d_msg = PoseStamped()
                        pose_3d_msg.header = image_data.header
                        pose_3d_msg.pose.position.x = X_camera
                        pose_3d_msg.pose.position.y = Y_camera
                        pose_3d_msg.pose.position.z = Z_camera
                        pose_3d_msg.pose.orientation.w = 1.0

                        self.pose_3d_pub.publish(pose_3d_msg)
                        #rospy.loginfo("Pose 3D do centroide publicada: X=%.3f, Y=%.3f, Z=%.3f", X_camera, Y_camera, Z_camera)

                        # NOVO: Publicar Marker 3D (Esfera)
                        marker = Marker()
                        marker.header = image_data.header
                        marker.ns = "container_centroid"
                        marker.id = self.marker_id # Use um ID unico, para nao ter multiplos markers no mesmo ns
                        marker.type = Marker.SPHERE # Tipo de marker: Esfera
                        marker.action = Marker.ADD # Adicionar/Modificar o marker

                        # Posicao do marker
                        marker.pose.position.x = X_camera
                        marker.pose.position.y = Y_camera
                        marker.pose.position.z = Z_camera
                        marker.pose.orientation.w = 1.0 # Sem rotacao para uma esfera

                        # Escala do marker (tamanho da esfera)
                        marker.scale.x = 0.1 # Diametro de 10cm
                        marker.scale.y = 0.1
                        marker.scale.z = 0.1

                        # Cor do marker (RGBA: Vermelho, Verde, Azul, Alpha)
                        marker.color.r = 1.0 # Vermelho
                        marker.color.g = 0.0
                        marker.color.b = 0.0
                        marker.color.a = 1.0 # Totalmente opaco

                        marker.lifetime = rospy.Duration(0.5) # Duracao do marker (0.5 segundos)

                        self.marker_pub.publish(marker)
                        self.marker_id += 1 # Incrementar ID para o proximo marker

            #         else:
            #             rospy.logwarn("Profundidade invalida ou zero no centroide (%d, %d).", u, v)
            #     else:
            #         rospy.logwarn("Centroide (%d, %d) esta fora dos limites da imagem de profundidade (%dx%d).", u, v, depth_image.shape[1], depth_image.shape[0])
            # else:
            #     rospy.logwarn("Centroide 2D nao encontrado para calcular a pose 3D.")
        else:
            # Se nenhum container for encontrado, voce pode querer "limpar" o marker
            # Enviando um marker com ACTION_DELETE
            if self.marker_id > 0: # Apenas se ja publicou um marker antes
                marker = Marker()
                marker.header.frame_id = image_data.header.frame_id # Usar o mesmo frame_id
                marker.ns = "container_centroid"
                marker.id = self.marker_id - 1 # Deleta o ultimo ID conhecido
                marker.action = Marker.DELETE
                self.marker_pub.publish(marker)
            self.marker_id = 0 # Resetar o ID ou gerenciar IDs de forma mais robusta
            # rospy.logwarn("Nenhum container encontrado nesta imagem para processar profundidade.")

def main():
    try:
        ContainerDetectorNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("No de deteccao de container encerrado.")

if __name__ == '__main__':
    main()