import cv2
import numpy as np
import os
import time
import threading # Importa a biblioteca para multithreading
import rospy

from segment_anything import sam_model_registry, SamPredictor

def overlay_mask(image, mask, color, alpha=0.5):
    """Sobrepõe uma máscara colorida e translúcida em uma imagem."""
    colored_mask = np.zeros_like(image, dtype=np.uint8)
    colored_mask[mask] = color
    blended_image = cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)
    return blended_image

class DetectorAndSegmenter:
    def __init__(self):
        # --- Configurações do Detector ---
        rospy.init_node('sam_node', anonymous=True)
        self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback)
        self.image_pub = rospy.Publisher("/container/detected_image", Image, queue_size=1)

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Erro: Não foi possível abrir a câmera.")
            return

        self.lower_blue = np.array([90, 120, 70])
        self.upper_blue = np.array([130, 255, 255])
        self.min_area = 1000
        self.max_area = 50000 
        
        # --- Configurações do SAM (usando o modelo mais leve) ---
        print("Carregando o modelo Segment Anything Model (SAM)...")
        sam_checkpoint = "sam_vit_b.pth"
        model_type = "vit_b"
        device = "cuda"
        
        try:
            sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
            sam.to(device=device)
            self.predictor = SamPredictor(sam)
            print("Modelo SAM carregado com sucesso na GPU.")
        except Exception as e:
            print(f"Erro ao carregar o modelo SAM: {e}")
            return
            
        # --- Variáveis para Multithreading e Comunicação ---
        self.lock = threading.Lock()  # Um "cadeado" para proteger dados compartilhados
        self.frame_to_process = None
        self.box_to_process = None
        self.latest_mask = None
        self.running = True # Flag para controlar a execução da thread
        
        # --- Inicialização e Início da Thread do SAM ---
        print("Iniciando a thread de segmentação em segundo plano...")
        self.sam_thread = threading.Thread(target=self._sam_worker, daemon=True)
        self.sam_thread.start()

        # --- Loop Principal de Exibição de Vídeo ---
        self.run_main_loop()

    def _sam_worker(self):
        """Função que roda em segundo plano, executando o SAM."""
        print("Worker do SAM iniciado.")
        while self.running:
            # Pega um "trabalho" para fazer (um frame e um box)
            with self.lock:
                local_frame = self.frame_to_process
                local_box = self.box_to_process
                # Marca o trabalho como "pego" para não processar de novo
                self.box_to_process = None 
            
            # Se houver um trabalho, executa o SAM (fora do lock!)
            if local_frame is not None and local_box is not None:
                try:
                    print("Worker: Recebeu trabalho. Processando SAM...")
                    rgb_frame = cv2.cvtColor(local_frame, cv2.COLOR_BGR2RGB)
                    self.predictor.set_image(rgb_frame)
                    
                    masks, _, _ = self.predictor.predict(
                        box=local_box,
                        multimask_output=False,
                    )
                    
                    # Após o processamento, atualiza a máscara mais recente
                    with self.lock:
                        self.latest_mask = masks[0]
                    print("Worker: Processamento concluído.")
                except Exception as e:
                    print(f"Worker: Erro durante a predição do SAM: {e}")

            # Pequena pausa para não consumir 100% da CPU quando ocioso
            time.sleep(0.00000001)
        print("Worker do SAM finalizado.")

    def run_main_loop(self):
        """Loop principal que cuida da câmera e da interface."""
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Fim do stream da câmera.")
                break
            
            self.process_frame(frame)
            
            k = cv2.waitKey(5) & 0xFF
            if k == 27: # Tecla ESC
                break
        
        self._cleanup()

    def process_frame(self, frame):
        """Processa cada frame da câmera (tarefas leves apenas)."""
        # Detecção de cor (rápido)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_blue, self.upper_blue)
        kernel = np.ones((7,7), np.uint8)
        mask_processed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask_processed = cv2.morphologyEx(mask_processed, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(mask_processed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        output_image = frame.copy()
        
        # Filtra o maior contorno
        if contours:
            main_contour = max(contours, key=cv2.contourArea)
            if self.min_area < cv2.contourArea(main_contour) < self.max_area:
                x, y, w, h = cv2.boundingRect(main_contour)
                # Desenha o box da detecção de cor (feedback visual imediato)
                cv2.rectangle(output_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
                
                # Envia o trabalho para a thread do SAM
                with self.lock:
                    self.frame_to_process = frame
                    self.box_to_process = np.array([x, y, x + w, y + h])

        # Pega a máscara mais recente calculada pelo worker e a desenha
        with self.lock:
            current_mask = self.latest_mask
        
        if current_mask is not None:
            output_image = overlay_mask(output_image, current_mask, color=(0, 251, 255), alpha=0.6)

        cv2.imshow("Resultado com SAM Fluido", output_image)
        cv2.imshow("Mascara de Cor Azul", mask)

    def _cleanup(self):
        """Finaliza os recursos de forma limpa."""
        print("Finalizando...")
        self.running = False # Sinaliza para a thread parar
        self.sam_thread.join(timeout=1) # Espera a thread terminar
        self.cap.release()
        cv2.destroyAllWindows()
        print("Aplicação encerrada.")


if __name__ == '__main__':
    try:
        detector = DetectorAndSegmenter()
    except KeyboardInterrupt:
        print("Interrupção pelo usuário.")