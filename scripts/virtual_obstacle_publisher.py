#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import tf
from std_msgs.msg import Float32MultiArray, Header
from sensor_msgs.msg import PointCloud
from geometry_msgs.msg import Point32, PointStamped

class VirtualObstaclePublisher:
    def __init__(self):
        rospy.init_node('virtual_obstacle_publisher', anonymous=True)

        # Publica PointCloud no tópico que o costmap escuta
        self.pub = rospy.Publisher('/striped_tape_obstacle', PointCloud, queue_size=10)

        # Assina as coordenadas da fita
        rospy.Subscriber('/fita_zebrada/centros', Float32MultiArray, self.fita_callback)

        # Inicializa o listener do TF
        self.tf_listener = tf.TransformListener()

        rospy.loginfo("VirtualObstaclePublisher iniciado.")
        rospy.spin()

    def fita_callback(self, msg):
        data = msg.data

        pontos_map = []

        # Corrigido: considera o Z real
        if len(data) == 3:
            pontos = [(data[0], data[1], data[2])]
        elif len(data) == 6:
            pontos = [(data[0], data[1], data[2]), (data[3], data[4], data[5])]
        else:
            rospy.logwarn("Nenhum ponto válido recebido.")
            return

        for (x, y, z) in pontos:
            ponto_camera = PointStamped()
            ponto_camera.header.frame_id = "camera_link"  # Ou base_link, conforme seu TF tree
            ponto_camera.header.stamp = rospy.Time(0)

            # 📌 Converte cm -> metros
            ponto_camera.point.x = x / 1000.0
            ponto_camera.point.y = y / 1000.0
            ponto_camera.point.z = z / 1000.0

            try:
                ponto_map = self.tf_listener.transformPoint("map", ponto_camera)
                p_map = Point32()
                p_map.x = ponto_map.point.x
                p_map.y = ponto_map.point.y
                p_map.z = 0.0  # z no costmap é plano, então 0.0

                pontos_map.append(p_map)

            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
                rospy.logwarn("Erro transformando ponto: %s", str(e))

        if not pontos_map:
            rospy.logwarn("Nenhum ponto transformado.")
            return

        # Monta e publica PointCloud
        cloud = PointCloud()
        cloud.header = Header()
        cloud.header.stamp = rospy.Time.now()
        cloud.header.frame_id = "map"
        cloud.points = pontos_map

        self.pub.publish(cloud)
        rospy.loginfo("PointCloud publicado com %d pontos.", len(pontos_map))


if __name__ == '__main__':
    try:
        VirtualObstaclePublisher()
    except rospy.ROSInterruptException:
        pass
