import cv2
import numpy as np
import os
import time
import threading # Mantido para fins de estrutura, mas não essencial sem o SAM thread

class DetectorDeFaixasLaranja:
    def __init__(self):
        # --- Configurações da Câmera ---
        self.cap = cv2.VideoCapture(0) # Tenta abrir a câmera padrão (0)
        if not self.cap.isOpened():
            print("Erro: Não foi possível abrir a câmera.")
            exit() # Sai do programa se a câmera não abrir

        # --- Configurações HSV para Laranja ---
        # Range HSV para laranja forte e vibrante
        self.lower_orange_normal = np.array([4, 120, 130])
        self.upper_orange_normal = np.array([19, 200, 255])
        
        # Range HSV para laranja muito claro / quase branco (sob forte luz/reflexo)
        self.lower_orange_highlight = np.array([5, 20, 200])   
        self.upper_orange_highlight = np.array([15, 80, 255])  
        
        # Área mínima e máxima para filtrar contornos (ajuste conforme o tamanho esperado dos objetos em pixels)
        self.min_area = 600
        self.max_area = 60000 
        
        # --- Inicia o Loop Principal de Exibição de Vídeo ---
        self.run_main_loop()

    def run_main_loop(self):
        """
        Loop principal que captura frames da câmera e gerencia a interface do usuário.
        """
        while True:
            try:
                ret, frame = self.cap.read() # Lê um frame da câmera
                if not ret:
                    print("Fim do stream da câmera ou erro de leitura.")
                    break # Sai do loop se não conseguir ler o frame
                
                self.process_frame(frame) # Chama o método de processamento do frame
                
                k = cv2.waitKey(5) & 0xFF # Espera por uma tecla (5ms)
                if k == 27: # Se a tecla ESC (ASCII 27) for pressionada
                    break # Sai do loop
            except Exception as e:
                print(f"Erro inesperado no loop principal: {e}")
                break # Sai do loop em caso de erro grave

        self._cleanup() # Chama a função de limpeza ao sair do loop

    def process_frame(self, frame):
        """
        Processa cada frame da câmera:
        1. Realiza detecção de cor HSV para identificar múltiplas áreas de laranja.
        2. Calcula e desenha bounding boxes rotacionadas para cada objeto detectado.
        3. Desenha linhas conectando os centros dos objetos detectados em ordem.
        """
        output_image = frame.copy() # Cria uma cópia do frame para desenhar os resultados
        
        # 1. Detecção de cor HSV para laranja (rápido)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Máscara para laranja "normal"
        mask_normal = cv2.inRange(hsv, self.lower_orange_normal, self.upper_orange_normal) 
        
        # Máscara para laranja "quase branco" (reflexo de luz)
        mask_highlight = cv2.inRange(hsv, self.lower_orange_highlight, self.upper_orange_highlight)
        
        # Combina as duas máscaras usando OR lógico
        mask_processed = cv2.bitwise_or(mask_normal, mask_highlight)
        
        # Operações morfológicas para limpar a máscara combinada e unir regiões próximas
        kernel = np.ones((5,5), np.uint8) # Kernel para operações morfológicas
        # 'OPEN' (erosão seguida de dilatação) remove ruídos pequenos e separa objetos
        mask_processed = cv2.morphologyEx(mask_processed, cv2.MORPH_OPEN, kernel) 
        # 'CLOSE' (dilatação seguida de erosão) fecha pequenos buracos e une áreas próximas
        mask_processed = cv2.morphologyEx(mask_processed, cv2.MORPH_CLOSE, kernel)
        
        # Encontra todos os contornos (bordas de objetos) na máscara processada
        contours, _ = cv2.findContours(mask_processed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        coords_centers = [] # Lista para armazenar as coordenadas dos centros dos objetos detectados

        # 2. Iterar sobre todos os contornos encontrados e processar cada objeto
        for contour in contours:
            area = cv2.contourArea(contour)
            # Filtra contornos por área para ignorar ruídos ou objetos muito pequenos/grandes
            if self.min_area < area < self.max_area:
                # Calcula a bounding box rotacionada (retângulo de área mínima que envolve o contorno)
                rect = cv2.minAreaRect(contour)
                box_points = cv2.boxPoints(rect) # Obtém os 4 vértices do retângulo rotacionado
                box_points = np.int32(box_points) # Converte os pontos para inteiros de 32 bits

                # Desenha a bounding box rotacionada no frame de saída (cor verde)
                cv2.drawContours(output_image, [box_points], 0, (0, 255, 0), 2) 
                
                # Pega as coordenadas do centro do retângulo rotacionado
                (center_x, center_y), _, _ = rect
                center_x_int = int(center_x)
                center_y_int = int(center_y)
                
                # Adiciona o centro à lista para posterior conexão por linha
                coords_centers.append((center_x_int, center_y_int))
                
                # Opcional: Desenhar um círculo no centro do retângulo
                cv2.circle(output_image, (center_x_int, center_y_int), 5, (0, 255, 255), -1) # Círculo amarelo preenchido

        # 3. Desenha linhas conectando os centros em ordem crescente (X, Y)
        if len(coords_centers) > 1:
            # Ordena a lista de centros: primeiro por X, depois por Y
            coords_centers.sort(key=lambda ponto: (ponto[0], ponto[1]))
            
            # Começa a desenhar as linhas do primeiro ponto
            pt1 = coords_centers[0]
            for pt2 in coords_centers[1:]:
                cv2.line(output_image, pt1, pt2, (255, 0, 0), 3, lineType=cv2.LINE_AA) # Linha azul, antialiased
                pt1 = pt2 # O fim da linha atual se torna o início da próxima

        # Exibe os resultados na janela do OpenCV
        cv2.imshow("Detector de Faixas Laranja", output_image)
        # Exibe a máscara de cor pré-processada para depuração
        cv2.imshow("Mascara de Cor Laranja", mask_processed)

    def _cleanup(self):
        """
        Finaliza os recursos de forma limpa antes de encerrar a aplicação.
        Libera a câmera e fecha todas as janelas do OpenCV.
        """
        print("Finalizando...")
        self.cap.release() # Libera o objeto da câmera
        cv2.destroyAllWindows() # Fecha todas as janelas do OpenCV
        print("Aplicação encerrada.")


if __name__ == '__main__':
    try:
        # Cria uma instância da classe para iniciar o detector
        detector = DetectorDeFaixasLaranja()
    except KeyboardInterrupt:
        print("Interrupção pelo usuário (Ctrl+C).")
    except Exception as e:
        print(f"Ocorreu um erro inesperado durante a inicialização ou execução: {e}")