# -*- coding: utf-8 -*-

import cv2
import numpy as np

def detect_and_segment_blue_container_py2(image_np):
    """
    Identifica, segmenta e calcula o centroide de um container azul em uma imagem (np array)
    usando segmentação baseada em cor (HSV). Adaptado para Python 2.7.

    Args:
        image_np (numpy.ndarray): A imagem de entrada como um array NumPy (BGR).

    Returns:
        tuple: Uma tupla contendo:
               - result_image (numpy.ndarray): Imagem com o contorno e centroide marcados.
                                               Retorna None se a imagem for inválida.
               - container_mask (numpy.ndarray): Máscara binária do container. Retorna None se nenhum
                                                 container significativo for encontrado.
               - centroid_coords (tuple): Coordenadas (x, y) do centroide. Retorna None se nenhum
                                          container significativo for encontrado.
    """
    if image_np is None:
        print ("Erro: Imagem de entrada invalida.")
        return None, None, None

    output_image = image_np.copy() # Imagem para desenhar o resultado

    # Conversão para HSV
    hsv_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2HSV)

    # Definir Faixas de Cor para o Azul (mesmos valores ajustados)
    lower_blue = np.array([90, 100, 50])
    upper_blue = np.array([130, 255, 255])

    # Criar uma Máscara de Cor
    mask = cv2.inRange(hsv_image, lower_blue, upper_blue)

    # Refinar a Máscara com Operações Morfológicas
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Encontrar Contornos
    _, contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    container_contour = None
    max_area = 0.0

    # Filtrar Contornos para encontrar o container
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Filtro inicial por área: Ajuste se os containers puderem ser muito pequenos ou muito grandes.
        # Um valor muito pequeno pega ruído, muito grande ignora objetos distantes.
        if area < 1000: # Exemplo: Ignora objetos muito pequenos
            continue

        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h

        # Calcular Solidez (Solidity): Area do contorno / Area do seu casco convexo (convex hull)
        # Objetos "cheios" e convexos (como um retangulo) tem solidez proxima de 1.
        # Objetos irregulares ou com buracos tem solidez menor.
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = 0.0
        if hull_area > 0: # Evitar divisao por zero
            solidity = area / hull_area

        # Ajuste no filtro de relação de aspecto:
        # Um container pode ser quase quadrado quando visto frontalmente.
        # Amplie a faixa para incluir proporções mais próximas de 1.
        # Ex: de 0.5 (mais alto que largo) a 5.0 (muito mais largo que alto).
        # Este é o principal ajuste para containers frontais.
        is_rectangular_like = (0.5 < aspect_ratio < 5.0) 

        # Adicione o filtro de solidez (ajuste o limiar)
        # Um container deve ser relativamente "sólido"
        is_solid_enough = (solidity > 0.8) # Um bom ponto de partida para objetos solidos

        # Combine os filtros
        if is_rectangular_like and is_solid_enough:
            if area > max_area:
                max_area = area
                container_contour = contour

    centroid_coords = None
    if container_contour is not None:
        M = cv2.moments(container_contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centroid_coords = (cX, cY)
        else:
            centroid_coords = (0, 0)

        cv2.drawContours(output_image, [container_contour], -1, (0, 255, 0), 2)
        cv2.circle(output_image, centroid_coords, 7, (255, 0, 0), -1)
        cv2.putText(output_image, "Centroide: (%d, %d)" % centroid_coords, (cX - 50, cY - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    else:
        # print ("Nenhum container azul significativo encontrado.")
        mask = None

    return output_image, mask, centroid_coords