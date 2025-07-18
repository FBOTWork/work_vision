#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge

class DetectorMesaLUniversal:
    def __init__(self):
        rospy.init_node('detector_mesa_l_universal', anonymous=True)

        self.bridge = CvBridge()
        self.pub_image = rospy.Publisher('/fita_zebrada/detected_image', Image, queue_size=10)
        self.pub_coords = rospy.Publisher('/fita_zebrada/centros', Float32MultiArray, queue_size=10)

        rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback)
        rospy.Subscriber('/camera/depth/image_rect_raw', Image, self.depth_callback)

        self.depth_image = None
        self.last_print_time = rospy.Time.now()

        self.lower_green = np.array([70, 100, 95])
        self.upper_green = np.array([130, 255, 255])
        self.min_area = 100
        self.max_area = 60000

        rospy.loginfo("Detector de mesa L universal iniciado.")
        rospy.spin()

    def depth_callback(self, msg):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            rospy.logerr("Erro ao converter imagem de profundidade: {e}")

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            rospy.logerr("Erro ao converter imagem: {e}")
            return

        if self.depth_image is None:
            rospy.logwarn_throttle(2, "Imagem de profundidade ainda não disponível.")
            return

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_green, self.upper_green)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        _, contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
 
        largest_contour = None
        max_area = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_area < area < self.max_area and area > max_area:
                largest_contour = contour
                max_area = area

        output_image = frame.copy()

        if largest_contour is not None:
            epsilon = 0.02 * cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)
            cv2.drawContours(output_image, [approx], -1, (0, 255, 0), 2)

            if len(approx) < 4:
                rospy.logwarn_throttle(2, "Contorno aproximado não tem vértices suficientes.")
                self.publicar_imagem(output_image)
                return

            # Construir lista de segmentos com seus dados
            segmentos = []
            for i in range(len(approx)):
                p1 = approx[i][0]
                p2 = approx[(i + 1) % len(approx)][0]
                vetor = p2 - p1
                comprimento = np.linalg.norm(vetor)
                segmentos.append({'p1': p1, 'p2': p2, 'vetor': vetor, 'comprimento': comprimento})

            # Procurar pares adjacentes e pontuar
            pares = []
            max_soma = 0
            for i in range(len(segmentos)):
                seg1 = segmentos[i]
                seg2 = segmentos[(i + 1) % len(segmentos)]  # segmento adjacente seguinte
                pts_seg1 = {tuple(seg1['p1']), tuple(seg1['p2'])}
                pts_seg2 = {tuple(seg2['p1']), tuple(seg2['p2'])}
                comum = pts_seg1.intersection(pts_seg2)
                if len(comum) == 1:
                    soma_compr = seg1['comprimento'] + seg2['comprimento']
                    if soma_compr > max_soma:
                        max_soma = soma_compr

                    ponto_intersecao = np.array(list(comum)[0])

                    def vetor_rel(seg, pb):
                        if np.array_equal(seg['p1'], pb):
                            return seg['vetor']
                        else:
                            return -seg['vetor']

                    v1 = vetor_rel(seg1, ponto_intersecao)
                    v2 = vetor_rel(seg2, ponto_intersecao)

                    ang_rad = np.arccos(
                        np.clip(
                            np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6),
                            -1.0, 1.0
                        )
                    )
                    ang_deg = np.degrees(ang_rad)

                    pares.append({
                        'seg1': seg1,
                        'seg2': seg2,
                        'ponto': ponto_intersecao,
                        'angulo': ang_deg,
                        'soma': soma_compr,
                        'v1': v1,
                        'v2': v2,
                    })

            if len(pares) == 0:
                rospy.logwarn_throttle(2, "Nenhum par adjacente encontrado.")
                self.publicar_imagem(output_image)
                return

            # Definir pesos para pontuação
            w_angulo = 0.7
            w_comprimento = 0.3

            # Calcular pontuação para cada par
            for par in pares:
                ang_diff = abs(par['angulo'] - 90)
                score_angulo = 1 - (ang_diff / 90)  # valor entre 0 e 1
                score_comp = par['soma'] / max_soma if max_soma > 0 else 0
                par['score'] = w_angulo * score_angulo + w_comprimento * score_comp

            # Escolher par com maior pontuação
            melhor_par = max(pares, key=lambda x: x['score'])

            ponto_intersecao = melhor_par['ponto']
            v1 = melhor_par['v1']
            v2 = melhor_par['v2']

            # Normalizar vetores
            v1_unit = v1 / (np.linalg.norm(v1) + 1e-6)
            v2_unit = v2 / (np.linalg.norm(v2) + 1e-6)

            rospy.loginfo_throttle(2, "Melhor par escolhido - ângulo: {melhor_par['angulo']:.1f}°, score: {melhor_par['score']:.2f}")

            cv2.circle(output_image, tuple(ponto_intersecao), 8, (0, 0, 255), -1)
            cv2.putText(output_image, "Ponto Zero", tuple(ponto_intersecao + np.array([5, -5])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Ajuste da escala (pixels para cm)
            escala_x = 400  # pixels correspondem a 80cm (lado maior da mesa)
            escala_y = 250  # pixels correspondem a 50cm (lado menor da mesa)

            pt_a = ponto_intersecao
            pt_b = (ponto_intersecao + v1_unit * escala_x).astype(int)
            pt_c = (pt_b + v2_unit * escala_y).astype(int)
            pt_d = (ponto_intersecao + v2_unit * escala_y).astype(int)

            mesa_pts = np.array([pt_a, pt_b, pt_c, pt_d])

            cv2.polylines(output_image, [mesa_pts], isClosed=True, color=(255, 0, 0), thickness=2)

            centro = np.mean(mesa_pts, axis=0).astype(int)
            cx, cy = centro

            if 0 <= cy < self.depth_image.shape[0] and 0 <= cx < self.depth_image.shape[1]:
                cz = float(self.depth_image[cy, cx])
            else:
                cz = 0.0

            cv2.circle(output_image, (cx, cy), 7, (255, 0, 255), -1)

            coord_msg = Float32MultiArray()
            coord_msg.data = [float(cx), float(cy), float(cz)]
            self.pub_coords.publish(coord_msg)

            if (rospy.Time.now() - self.last_print_time).to_sec() > 0.5:
                rospy.loginfo("Centro da mesa: [{:.2f}, {:.2f}, {:.3f}]".format(cx, cy, cz))
                self.last_print_time = rospy.Time.now()
        else:
            rospy.logwarn_throttle(2, "Nenhum contorno válido encontrado.")

        self.publicar_imagem(output_image)

    def publicar_imagem(self, img):
        try:
            msg_out = self.bridge.cv2_to_imgmsg(img, encoding='bgr8')
            self.pub_image.publish(msg_out)
        except Exception as e:
            rospy.logerr("Erro ao publicar imagem: {e}")

if __name__ == '__main__':
    try:
        DetectorMesaLUniversal()
    except rospy.ROSInterruptException:
        pass
