import cv2
import numpy as np
import os
import time
import threading

from segment_anything import sam_model_registry, SamPredictor

def overlay_mask(image, mask, color, alpha=0.5):
    """Sobrepõe uma máscara colorida e translúcida em uma imagem."""
    colored_mask = np.zeros_like(image, dtype=np.uint8)
    colored_mask[mask] = color
    blended_image = cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)
    return blended_image

class DetectorAndSegmenter:
    def __init__(self):
        # --- Configurações ---
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Erro: Não foi possível abrir a câmera.")
            return

        # --- OTIMIZAÇÃO: Definir uma resolução de processamento menor ---
        self.processing_width = 640
        self.processing_height = 480
        # --- FIM DA OTIMIZAÇÃO ---

        self.lower_blue = np.array([90, 120, 70])
        self.upper_blue = np.array([130, 255, 255])
        self.min_area = 500 # Ajustado para a nova resolução menor

        # --- Configurações do SAM ---
        print("Carregando o modelo SAM...")
        sam_checkpoint = "checkpoints/sam_vit_b.pth"
        model_type = "vit_b"
        device = "cuda"
        
        try:
            sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
            sam.to(device=device)
            self.predictor = SamPredictor(sam)
            print("Modelo SAM carregado com sucesso.")
        except Exception as e:
            print(f"Erro ao carregar o modelo SAM: {e}")
            return
            
        # --- Multithreading ---
        self.lock = threading.Lock()
        self.frame_to_process = None
        self.box_to_process = None
        self.latest_mask = None
        self.running = True
        
        self.sam_thread = threading.Thread(target=self._sam_worker, daemon=True)
        self.sam_thread.start()

        self.run_main_loop()

    def _sam_worker(self):
        """Worker que roda o SAM em segundo plano."""
        while self.running:
            with self.lock:
                local_frame = self.frame_to_process
                local_box = self.box_to_process
                self.box_to_process = None
            
            if local_frame is not None and local_box is not None:
                try:
                    rgb_frame = cv2.cvtColor(local_frame, cv2.COLOR_BGR2RGB)
                    self.predictor.set_image(rgb_frame)
                    
                    masks, _, _ = self.predictor.predict(
                        box=local_box,
                        multimask_output=False,
                    )
                    
                    with self.lock:
                        # A máscara gerada é para a imagem de baixa resolução
                        self.latest_mask = masks[0]
                except Exception as e:
                    print(f"Worker: Erro na predição do SAM: {e}")

            time.sleep(0.01)

    def run_main_loop(self):
        """Loop principal de exibição."""
        while True:
            ret, original_frame = self.cap.read()
            if not ret:
                break
            
            # --- OTIMIZAÇÃO: Redimensiona o frame antes de processar ---
            processed_frame = cv2.resize(original_frame, (self.processing_width, self.processing_height))
            
            # O processamento agora ocorre no frame redimensionado
            self.process_frame(processed_frame, original_frame)
            
            k = cv2.waitKey(5) & 0xFF
            if k == 27:
                break
        
        self._cleanup()

    def process_frame(self, processed_frame, original_frame):
        """Processa o frame de baixa resolução e exibe no de alta."""
        hsv = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_blue, self.upper_blue)
        kernel = np.ones((7,7), np.uint8)
        mask_processed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask_processed = cv2.morphologyEx(mask_processed, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(mask_processed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        output_image = original_frame.copy()
        
        if contours:
            main_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(main_contour) > self.min_area:
                x, y, w, h = cv2.boundingRect(main_contour)
                
                with self.lock:
                    self.frame_to_process = processed_frame # Envia o frame pequeno
                    self.box_to_process = np.array([x, y, x + w, y + h])

        with self.lock:
            small_mask = self.latest_mask
        
        if small_mask is not None:
            # --- OTIMIZAÇÃO: Redimensiona a máscara de volta para o tamanho original ---
            original_h, original_w, _ = original_frame.shape
            full_size_mask = cv2.resize(small_mask.astype(np.uint8), (original_w, original_h), interpolation=cv2.INTER_NEAREST)
            output_image = overlay_mask(output_image, full_size_mask.astype(bool), color=(255, 144, 30), alpha=0.6)

        cv2.imshow("Resultado com SAM Otimizado", output_image)
        # Opcional: mostrar a máscara pequena para debug
        # cv2.imshow("Mascara de Cor (Processamento)", mask_processed)

    def _cleanup(self):
        """Finaliza os recursos."""
        print("Finalizando...")
        self.running = False
        self.sam_thread.join(timeout=1)
        self.cap.release()
        cv2.destroyAllWindows()
        print("Aplicação encerrada.")


if __name__ == '__main__':
    DetectorAndSegmenter()