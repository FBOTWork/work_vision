#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image, PointCloud2, CameraInfo
from std_msgs.msg import Float32MultiArray, Header
from cv_bridge import CvBridge
import sensor_msgs.point_cloud2 as pc2
from geometry_msgs.msg import PointStamped
import tf
from visualization_msgs.msg import Marker, MarkerArray
from image_geometry import PinholeCameraModel

class DetectorDeFaixasLaranja:
    def __init__(self):
        rospy.init_node('detector_de_faixas_laranja', anonymous=True)

        self.last_print_time = rospy.Time.now()

        # Inicializa ponte entre ROS e OpenCV
        self.bridge = CvBridge()
        
        # Setup camera model for depth projection
        self.camera_model = PinholeCameraModel()
        self.camera_info_received = False
        rospy.Subscriber('/camera/depth/camera_info', CameraInfo, self.camera_info_callback)

        # Publisher da imagem processada
        self.pub_image = rospy.Publisher('/fita_zebrada/detected_image', Image, queue_size=3)

        # Publisher das coordenadas dos centros detectados (x, y, z)
        self.pub_coords = rospy.Publisher('/fita_zebrada/centros', Float32MultiArray, queue_size=10)
        
        # Publisher para obstáculos virtuais no mapa (PointCloud acumulativo)
        self.pc_pub = rospy.Publisher('/virtual_obstacles', PointCloud2, queue_size=1)
        
        # Publisher para marcadores de visualização
        self.marker_pub = rospy.Publisher('/visualization_marker_array', MarkerArray, queue_size=10)

        # Subscriber da imagem RGB da câmera
        rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback)

        # Subscriber da imagem de profundidade
        self.depth_image = None
        rospy.Subscriber('/camera/depth/image_rect_raw', Image, self.depth_callback)


        # Ranges HSV para detectar tons de amarelo
        self.lower_yellow = np.array([18, 80, 80])
        self.upper_yellow = np.array([15, 255, 255])

        # Ranges HSV para detectar tons de laranja
        self.lower_orange_normal = np.array([0, 80, 120])
        self.upper_orange_normal = np.array([5, 200, 255])
        self.lower_orange_highlight = np.array([5, 100, 120])
        self.upper_orange_highlight = np.array([12, 200, 255])

        self.min_area = 20
        self.max_area = 600
        
        # TF listener para transformações de coordenadas
        self.tf_listener = tf.TransformListener()

        self.notificado = False
        # Inicializa lista de pontos publicados
        self.pontos_publicados = []

        rospy.loginfo("Detector de faixas laranja e amarelo Finiciado.")
        rospy.spin()
    def ponto_ja_existe_proximo(self, ponto, current_time, distancia=0.1, tempo=1.0):
        """
        Verifica se já existe um ponto próximo publicado recentemente.
        distancia: distância máxima (em metros) para considerar pontos próximos
        tempo: tempo máximo (em segundos) para considerar pontos recentes
        """
        for item in self.pontos_publicados:
            p = item['ponto']
            t = item['timestamp']
            if (current_time - t).to_sec() < tempo:
                dist = np.hypot(p.x - ponto.x, p.y - ponto.y)
                if dist < distancia:
                    return True
        return False

    def camera_info_callback(self, msg):
        """Callback para receber informações da câmera (parâmetros intrínsecos)"""
        if not self.camera_info_received:
            self.camera_model.fromCameraInfo(msg)
            self.camera_info_received = True

    def depth_callback(self, msg):
        try:
            # Converte imagem de profundidade ROS para OpenCV (tipo float32)
            depth_raw = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            # Converte de milímetros para metros
            self.depth_image = depth_raw / 1000.0
        except Exception as e:
            rospy.logerr("Erro ao converter imagem de profundidade: {}".format(e))

    def publicar_ponto_como_obstaculo(self, ponto, timestamp):
        """Publica um ponto como obstáculo virtual no mapa com timestamp"""
        # Só publica se for dados frescos
        current_time = rospy.Time.now()
        if (current_time - timestamp).to_sec() > 0.1:  # Dados mais antigos que 100ms
            return
        
        # Verifica se já existe um ponto próximo publicado recentemente
        if self.ponto_ja_existe_proximo(ponto, current_time):
            return
            
        header = Header()
        header.stamp = timestamp  # Usa o timestamp da detecção
        header.frame_id = "map"
        points = [(ponto.x, ponto.y, ponto.z)]
        pc2_msg = pc2.create_cloud_xyz32(header, points)
        self.pc_pub.publish(pc2_msg)
        
        # Adiciona ao cache de pontos publicados
        self.pontos_publicados.append({
            'ponto': ponto,
            'timestamp': current_time
        })

    def publicar_pointcloud_vazio(self):
        """Publica um PointCloud vazio para manter o buffer atualizado"""
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "map"
        points = []  # Lista vazia de pontos
        pc2_msg = pc2.create_cloud_xyz32(header, points)
        self.pc_pub.publish(pc2_msg)

    def publicar_marker(self, ponto):
        """Publica um marcador de visualização no RViz"""
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
        marker_array = MarkerArray()
        marker_array.markers.append(marker)
        self.marker_pub.publish(marker_array)
        rospy.sleep(0.1)  # Aguarda um pouco para garantir que o marcador seja publicado

    def transformar_pixel_para_chao(self, cx, cy, frame_id):
        """Transforma coordenadas de pixel para posição real no chão (z=0 no mapa)"""
        if not self.camera_info_received:
            rospy.logwarn("Parâmetros da câmera ainda não recebidos.")
            return None
            
        try:
            # Projeta pixel para direção 3D normalizada no frame da câmera
            ray_camera = self.camera_model.projectPixelTo3dRay((cx, cy))
            
            # Cria um ponto distante na direção do raio (usando depth se disponível)
            depth = 5.0  # Profundidade arbitrária para criar o ponto 3D
            if self.depth_image is not None:
                if 0 <= cy < self.depth_image.shape[0] and 0 <= cx < self.depth_image.shape[1]:
                    depth_val = self.depth_image[cy, cx]
                    if depth_val > 0 and not np.isnan(depth_val):
                        depth = depth_val
            
            # Cria PointStamped no frame da câmera
            ponto_camera = PointStamped()
            ponto_camera.header.frame_id = frame_id
            ponto_camera.header.stamp = rospy.Time(0)
            ponto_camera.point.x = ray_camera[0] * depth
            ponto_camera.point.y = ray_camera[1] * depth
            ponto_camera.point.z = ray_camera[2] * depth
            
            # Transforma para o frame do mapa usando tf
            ponto_mapa = self.tf_listener.transformPoint("map", ponto_camera)
            
            # Projeta para o chão (z=0) mantendo x e y
            ponto_mapa.point.z = 0.0
            
            return ponto_mapa
            
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logwarn("Erro de transformação TF: {}".format(e))
            return None
        except Exception as e:
            rospy.logwarn("Erro ao transformar pixel para chão: {}".format(e))
            return None

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

        # Timestamp da imagem atual
        image_timestamp = msg.header.stamp
        current_time = rospy.Time.now()
        
        # Verifica se a imagem não é muito antiga (evita processar dados obsoletos)
        if (current_time - image_timestamp) > rospy.Duration(0.2):
            self.publicar_pointcloud_vazio()
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
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Contornos
        contour_result = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contour_result[1] if len(contour_result) == 3 else contour_result[0]

        coords_centers = []
        coords_world = []  

        # Determina o frame da câmera
        camera_frame = self.camera_model.tfFrame() if self.camera_info_received else msg.header.frame_id

        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_area < area < self.max_area:
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                # Ajusta coordenadas para o frame original
                box = box
                box = np.int32(box)
                # cv2.drawContours(output_image, [box], 0, (0, 255, 0), 2)

                (cx, cy), _, _ = rect
                # Ajusta coordenadas para o frame original
                # cx = int(cx)
                # cy = int(cy)

                # Obtém o valor de profundidade (z) no ponto (cx, cy)
                if 0 <= cy < self.depth_image.shape[0] and 0 <= cx < self.depth_image.shape[1]:
                    z = float(self.depth_image[cy, cx]) 
                    
                    # Só processa se a profundidade for válida (não zero e não NaN)
                    # Mas agora vamos projetar para o chão ao invés de usar a profundidade
                    if z > 0 and not np.isnan(z):
                        # Transforma pixel para coordenadas do chão no mapa
                        ponto_mundo = self.transformar_pixel_para_chao(cx, cy, camera_frame)
                        
                        if ponto_mundo is not None:
                            coords_centers.append((cx, cy, 0))
                            coords_world.append(ponto_mundo)
                            # cv2.circle(output_image, (cx, cy), 5, (0, 255, 255), -1)
                            
                            # Publica ponto como obstáculo no mapa (na posição real do chão)
                            self.publicar_ponto_como_obstaculo(ponto_mundo.point, image_timestamp)
                            self.publicar_marker(ponto_mundo.point)
                            
                            # Atualiza timestamp da última detecção
                            self.ultima_deteccao_timestamp = current_time
                else:
                    z = 0.0  # Profundidade inválida, define como 0

        # Publica as coordenadas (x, y, z) no tópico ROS
        flat_coords = []
        current_time = rospy.Time.now()

        # Log das coordenadas transformadas para o mundo
        if coords_world:
            if (current_time - self.last_print_time).to_sec() > 0.5:
                for i, ponto_mundo in enumerate(coords_world):
                    rospy.loginfo("Fita {} no mapa: x={:.2f}, y={:.2f}, z={:.2f}".format(
                        i+1, ponto_mundo.point.x, ponto_mundo.point.y, ponto_mundo.point.z))
                self.last_print_time = current_time

        # Se dois ou mais retângulos, encontra os dois extremos (mais distantes) - usando coordenadas de pixel
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
                    if dist > max_dist:
                        max_dist = dist
                        pt1 = coords_centers[i]
                        pt2 = coords_centers[j]

            if pt1 and pt2:
                flat_coords.extend([float(pt1[0]), float(pt1[1]), float(pt1[2]),
                                    float(pt2[0]), float(pt2[1]), float(pt2[2])])
                cv2.line(output_image, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), (255, 0, 0), 3, lineType=cv2.LINE_AA)
                cv2.line(output_image, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), (255, 0, 0), 3, lineType=cv2.LINE_AA)

        # Se exatamente um retângulo, publica ele mesmo
        elif len(coords_centers) == 1:
            x, y, z = coords_centers[0]
            flat_coords.extend([x, y, z])

        # Se nenhum retângulo, mostra aviso
        elif len(coords_centers) == 0:
            if (current_time - self.last_print_time).to_sec() > 0.3:
                if not self.notificado:
                    rospy.logwarn("Nenhuma fita detectada.")
                    self.notificado = True
                self.last_print_time = current_time
            
            # Publica PointCloud vazio para manter o buffer atualizado
            self.publicar_pointcloud_vazio()

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
            msg_out.header.stamp = image_timestamp  # Mantém timestamp original
            self.pub_image.publish(msg_out)
        except Exception as e:
            rospy.logerr("Erro ao publicar imagem: {}".format(e))

if __name__ == '__main__':
    try:
        DetectorDeFaixasLaranja()
    except rospy.ROSInterruptException:
        pass
