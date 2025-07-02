import cv2
import numpy as np
import os
import time
import threading
import torch  # Importa a biblioteca para liberar memória da GPU

# Certifique-se de que 'segment_anything' esteja instalado e que o modelo esteja no caminho correto
from segment_anything import sam_model_registry, SamPredictor

def overlay_mask(image, mask, color, alpha=0.6):
    """
    Sobrepõe uma máscara colorida e translúcida em uma imagem,
    pintando apenas a área da máscara sem escurecer o fundo.
    """
    # Cria uma cópia da imagem original para não modificar a referência diretamente
    output_image = image.copy() 
    
    # Cria uma máscara 3D com a cor desejada (BGR) onde a máscara de segmentação é True (255)
    # E zeros (preto) onde a máscara é False (0)
    colored_mask = np.zeros_like(output_image, dtype=np.uint8)
    colored_mask[mask > 0] = color # Pinta apenas onde a máscara é 255 (objeto segmentado)
    
    # Combina a imagem original com a máscara colorida usando o alpha (transparência).
    # Onde colored_mask é preta (0), cv2.addWeighted essencialmente mantém a cor original de output_image
    output_image = cv2.addWeighted(output_image, 1.0, colored_mask, alpha, 0) # Peso 1.0 para imagem original
    
    return output_image

class DetectorAndSegmenter:
    def __init__(self):
        # --- Configurações da Câmera ---
        self.cap = cv2.VideoCapture(0) # Tenta abrir a câmera padrão (0)
        if not self.cap.isOpened():
            print("Erro: Não foi possível abrir a câmera.")
            exit() # Sai do programa se a câmera não abrir

        # --- Configurações HSV para Laranja ---
        # Range HSV para laranja forte e vibrante
        # H (Matiz): 5-15 (laranja puro)
        # S (Saturação): 180-255 (muito saturado)
        # V (Valor/Brilho): 100-255 (bem iluminado)
        self.lower_orange_normal = np.array([4, 120, 130])
        self.upper_orange_normal = np.array([19, 200, 255])
        
        # NOVO: Range HSV para laranja muito claro / quase branco (sob forte luz/reflexo)
        # H (Matiz): Mantém a faixa de laranja.
        # S (Saturação): Reduz bastante para pegar cores "lavadas", que parecem brancas. Ajuste este valor!
        # V (Valor/Brilho): Aumenta para valores muito altos, indicando alta incidência de luz.
        self.lower_orange_highlight = np.array([5, 20, 200])   
        self.upper_orange_highlight = np.array([15, 80, 255])  
        
        # Área mínima e máxima para filtrar contornos (ajuste conforme o tamanho esperado dos objetos em pixels)
        self.min_area = 600
        self.max_area = 60000 
        
        # --- Configurações do SAM (Segment Anything Model) ---
        print("Carregando o modelo Segment Anything Model (SAM)...")
        sam_checkpoint = "checkpoints/sam_vit_b.pth" # Caminho para o arquivo do modelo SAM
        model_type = "vit_b" # Tipo de modelo SAM (vit_b, vit_l, vit_h)
        device = "cuda" # Use 'cuda' para GPU, ou 'cpu' se não tiver GPU compatível

        try:
            # Verifica se o arquivo do checkpoint existe
            if not os.path.exists(sam_checkpoint):
                raise FileNotFoundError(f"Checkpoint SAM não encontrado em: {sam_checkpoint}")
            
            sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
            sam.to(device=device) # Move o modelo para a GPU (ou CPU)
            self.predictor = SamPredictor(sam)
            print(f"Modelo SAM carregado com sucesso na {device.upper()}.")
        except FileNotFoundError as fnfe:
            print(f"Erro: {fnfe}")
            print("Por favor, baixe o checkpoint do SAM e coloque-o na pasta 'checkpoints/'.")
            print("Link para download do 'vit_b': https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth")
            exit()
        except Exception as e:
            print(f"Erro ao carregar o modelo SAM ou configurar o dispositivo: {e}")
            print("Verifique se PyTorch está corretamente instalado com suporte a CUDA (se 'device' for 'cuda').")
            exit() 

        # --- Variáveis para Multithreading e Comunicação entre threads ---
        self.lock = threading.Lock()  # Usado para proteger o acesso a variáveis compartilhadas
        self.frames_to_process = []   # Fila de frames a serem enviados para o SAM worker
        self.boxes_to_process = []    # Fila de listas de boxes (uma lista por frame) para o SAM worker
        self.latest_masks = []        # Lista das máscaras mais recentes geradas pelo SAM worker
        self.running = True           # Flag para controlar o ciclo de vida das threads
        # Timestamp do último envio para o SAM para controlar a frequência de processamento
        self.last_sam_send_time = time.time() 
        
        # --- Inicialização e Início da Thread do SAM Worker ---
        print("Iniciando a thread de segmentação em segundo plano...")
        # 'daemon=True' faz com que a thread se encerre automaticamente quando o programa principal termina
        self.sam_thread = threading.Thread(target=self._sam_worker, daemon=True)
        self.sam_thread.start()

        # --- Inicia o Loop Principal de Exibição de Vídeo ---
        self.run_main_loop()

    def _sam_worker(self):
        """
        Função executada na thread separada para processar frames com o SAM.
        Responsável por pegar um frame e um conjunto de boxes, processá-los com o SAM,
        e armazenar as máscaras resultantes.
        """
        print("Worker do SAM iniciado.")
        while self.running:
            local_frame = None
            local_boxes = []
            
            # Adquire o lock para acessar as filas de forma segura
            with self.lock:
                if self.frames_to_process: # Verifica se há frames na fila
                    local_frame = self.frames_to_process.pop(0) # Pega o frame mais antigo
                    local_boxes = self.boxes_to_process.pop(0) # Pega as boxes correspondentes
                    self.latest_masks = [] # Limpa as máscaras antigas para o novo conjunto de resultados
            
            # Se um trabalho foi obtido, processa com o SAM (fora do lock para não bloquear a thread principal)
            if local_frame is not None and local_boxes:
                try:
                    # O SAM espera imagens em formato RGB
                    rgb_frame = cv2.cvtColor(local_frame, cv2.COLOR_BGR2RGB)
                    self.predictor.set_image(rgb_frame) # Prepara a imagem para o preditor SAM
                    
                    current_frame_masks = []
                    # Processa cada bounding box detectada no frame
                    for box in local_boxes:
                        # Realiza a predição de máscara com o SAM
                        masks, _, _ = self.predictor.predict(
                            box=box,
                            multimask_output=False, # Queremos apenas uma máscara por box
                        )
                        if masks.shape[0] > 0: # Se alguma máscara foi gerada
                            # CONVERSÃO ESSENCIAL: Máscaras do SAM são booleanas. Multiplica por 255 e converte para uint8.
                            uint8_mask = (masks[0] * 255).astype(np.uint8)
                            current_frame_masks.append(uint8_mask)
                    
                    # Adquire o lock para atualizar as máscaras mais recentes de forma segura
                    with self.lock:
                        self.latest_masks = current_frame_masks
                    
                    # Libera a memória não utilizada da GPU após o processamento do SAM
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
                except Exception as e:
                    print(f"Worker: Erro durante a predição do SAM: {e}")
            
            # Pequena pausa para evitar o consumo de 100% da CPU quando ocioso
            time.sleep(0.001) 
        print("Worker do SAM finalizado.")

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
        3. Envia as bounding boxes para a thread do SAM para segmentação em segundo plano.
        4. Exibe as máscaras segmentadas pelo SAM (quando disponíveis), sobrepondo-as.
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
        #mask_processed = cv2.morphologyEx(mask_combined, cv2.MORPH_OPEN, kernel) 
        # 'CLOSE' (dilatação seguida de erosão) fecha pequenos buracos e une áreas próximas
        #mask_processed = cv2.morphologyEx(mask_processed, cv2.MORPH_CLOSE, kernel)
        
        # Encontra todos os contornos (bordas de objetos) na máscara processada
        contours, _ = cv2.findContours(mask_processed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detected_boxes_for_sam = [] # Lista para armazenar as bounding boxes retangulares para o SAM

        # 2. Iterar sobre todos os contornos encontrados e processar cada objeto
        coords_strips = []
        for contour in contours:
            area = cv2.contourArea(contour)
            # Filtra contornos por área para ignorar ruídos ou objetos muito pequenos/grandes
            if self.min_area < area < self.max_area:
                # Calcula a bounding box rotacionada (retângulo de área mínima que envolve o contorno)
                rect = cv2.minAreaRect(contour)
                box_points = cv2.boxPoints(rect) # Obtém os 4 vértices do retângulo rotacionado
                box_points = np.int32(box_points) # Converte os pontos para inteiros de 32 bits (necessário para cv2.drawContours)

                # Desenha a bounding box rotacionada no frame de saída (cor verde)
                cv2.drawContours(output_image, [box_points], 0, (0, 255, 0), 2) 

                # Para o SAM, precisamos da bounding box retangular (x_min, y_min, x_max, y_max)
                x, y, w, h = cv2.boundingRect(contour)
                (center_x, center_y), _ , _ = rect
            
                center_x_int = int(center_x)
                center_y_int = int(center_y)
                coords_strips.append((center_x_int, center_y_int))
                detected_boxes_for_sam.append(np.array([x, y, x + w, y + h]))

        if len(coords_strips) > 1:
            print(coords_strips)
            coords_strips.sort(key=lambda ponto: (ponto[0], ponto[1]))
            pt1 = coords_strips[0]
            for points in coords_strips[1:]:
                cv2.line(output_image, pt1, points, (255, 0, 0), 3, lineType=cv2.LINE_8, shift=0)
                pt1 = points


        # 3. Envia o trabalho (frame e boxes) para a thread do SAM worker
        # Condições para envio:
        # - Há boxes detectadas.
        # - A fila de processamento do SAM não está cheia (limite de 1 frame na fila).
        # - Um tempo mínimo (0.5 segundos) passou desde o último envio para o SAM.
        if detected_boxes_for_sam and \
           (len(self.frames_to_process) < 1) and \
           ((time.time() - self.last_sam_send_time) > 0.05):
            
            with self.lock: # Adquire o lock para acessar as filas de forma segura
                self.frames_to_process.append(frame.copy()) # Envia uma cópia do frame para evitar problemas de concorrência
                self.boxes_to_process.append(detected_boxes_for_sam) # Envia a lista de boxes detectadas
                self.last_sam_send_time = time.time() # Atualiza o timestamp do último envio
                # Opcional: print(f"Main: Enviou {len(detected_boxes_for_sam)} boxes para o worker. Fila: {len(self.frames_to_process)}")

        # 4. Pega as máscaras mais recentes geradas pelo SAM worker e as desenha
        # As máscaras já vêm convertidas para np.uint8 da thread do SAM
        with self.lock: # Adquire o lock para acessar as máscaras de forma segura
            current_sam_masks = self.latest_masks
        
        if current_sam_masks: # Se houver máscaras disponíveis do SAM
            for mask_sam in current_sam_masks:
                # Sobrepõe a máscara segmentada (amarela) no frame de saída
                output_image = overlay_mask(output_image, mask_sam, color=(0, 255, 255), alpha=0.6) # Amarelo (BGR)

        # Exibe os resultados na janela do OpenCV
        cv2.imshow("Resultado com SAM Fluido (Laranja)", output_image)
        # Exibe a máscara de cor pré-processada para depuração
        cv2.imshow("Mascara de Cor Laranja (Pré-processamento)", mask_processed)

    def _cleanup(self):
        """
        Finaliza os recursos de forma limpa antes de encerrar a aplicação.
        Libera a câmera, encerra a thread do SAM e fecha todas as janelas do OpenCV.
        """
        print("Finalizando...")
        self.running = False # Sinaliza para a thread do SAM parar
        # Espera a thread do SAM terminar (com timeout para não travar indefinidamente)
        self.sam_thread.join(timeout=2) 
        self.cap.release() # Libera o objeto da câmera
        cv2.destroyAllWindows() # Fecha todas as janelas do OpenCV
        print("Aplicação encerrada.")


if __name__ == '__main__':
    try:
        # Cria uma instância da classe para iniciar o detector e segmentador
        detector = DetectorAndSegmenter()
    except KeyboardInterrupt:
        print("Interrupção pelo usuário (Ctrl+C).")
    except Exception as e:
        print(f"Ocorreu um erro inesperado durante a inicialização ou execução principal: {e}")