#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
from ultralytics import YOLO
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import rospkg
import os

# Importe as mensagens customizadas que você criou
from work_vision.msg import Description, Recognitions

class YoloV8Node:
    def __init__(self):
        rospy.loginfo("Inicializando o nó YOLOv8...")

        # --- Parâmetros ---
        self.image_topic = rospy.get_param("~image_topic", "/camera/color/image_raw")
        self.recog_img_topic = rospy.get_param("~recog_image_topic", "/work_vision/recog_imgs")
        # Nome do tópico para publicar as informações de reconhecimento (agora usando a msg Recognitions)
        self.recognitions_topic = rospy.get_param("~recognitions_topic", "/work_vision/recognitions")
        self.threshold = rospy.get_param("~threshold", 0.5)
        model_filename = rospy.get_param("~model_file", "yolov8n.pt")

        # --- Inicialização do Modelo YOLO ---
        rospack = rospkg.RosPack()
        package_path = rospack.get_path("work_vision")
        self.model_path = os.path.join(package_path, "weights", model_filename)
        
        try:
            self.model = YOLO(self.model_path)
            self.model.conf = self.threshold
            self.names = self.model.names
            rospy.loginfo(f"Modelo YOLO '{model_filename}' carregado com sucesso.")
        except Exception as e:
            rospy.logfatal(f"Erro ao carregar o modelo YOLO: {e}")
            rospy.signal_shutdown("Falha ao carregar o modelo.")
            return

        # --- CV Bridge ---
        self.bridge = CvBridge()

        # --- Publishers ---
        # Publisher para a imagem com as detecções desenhadas
        self.pub_img = rospy.Publisher(self.recog_img_topic, Image, queue_size=1)
        # Publisher para as informações de reconhecimento (USA A MENSAGEM CUSTOMIZADA)
        self.pub_recognitions = rospy.Publisher(self.recognitions_topic, Recognitions, queue_size=10)

        # --- Subscriber ---
        self.sub = rospy.Subscriber(self.image_topic, Image, self.callback)
        rospy.loginfo("Nó YOLOv8 pronto e aguardando imagens.")

    def callback(self, data):
        # rospy.loginfo("Imagem recebida para detecção.") # Descomente para debug detalhado

        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')
        except Exception as e:
            rospy.logerr(f"Erro ao converter imagem: {e}")
            return

        # Roda a inferência do YOLO
        results = self.model(cv_image, verbose=False) # verbose=False para uma saída mais limpa

        # Cria a mensagem principal que conterá todas as detecções
        recognitions_msg = Recognitions()
        # É uma boa prática usar o mesmo header da imagem original
        recognitions_msg.header = data.header

        # O resultado 'results' é uma lista, geralmente com um elemento por imagem
        for r in results:
            # Itera sobre cada caixa de detecção encontrada
            for box in r.boxes:
                # Cria uma mensagem 'Description' para cada objeto detectado
                desc_msg = Description()

                # Extrai as informações da caixa de detecção
                label_id = int(box.cls)
                desc_msg.label_class = self.names[label_id]
                desc_msg.probability = float(box.conf)
                
                # Pega as coordenadas da bounding box (x1, y1, x2, y2)
                b = box.xyxy[0].cpu().numpy()
                desc_msg.x = int(b[0])
                desc_msg.y = int(b[1])
                desc_msg.width = int(b[2] - b[0])
                desc_msg.height = int(b[3] - b[1])

                # Adiciona a descrição do objeto na lista da mensagem principal
                recognitions_msg.recognitions.append(desc_msg)

            # Publica a imagem com as detecções (bounding boxes) desenhadas
            im_array = r.plot() # r.plot() já desenha as caixas e labels
            img_msg = self.bridge.cv2_to_imgmsg(im_array, encoding="bgr8")
            self.pub_img.publish(img_msg)

        # Publica a mensagem com a lista de todos os objetos reconhecidos DE UMA SÓ VEZ
        if recognitions_msg.recognitions:
            self.pub_recognitions.publish(recognitions_msg)
            # Log apenas se algo for detectado
            rospy.loginfo(f"Detectado {len(recognitions_msg.recognitions)} objeto(s).")


    def run(self):
        rospy.spin()

if __name__ == "__main__":
    rospy.init_node("object_recognition_node", anonymous=True)
    try:
        yolo_node = YoloV8Node()
        yolo_node.run()
    except rospy.ROSInterruptException:
        pass