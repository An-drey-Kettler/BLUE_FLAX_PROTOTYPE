import time
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Настройки
THRESHOLD = 0.45  # Порог скачка (подобран эмпирически)
MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'

def main():
    print("[*] Loading model...")
    model = SentenceTransformer(MODEL_NAME)
    
    print("[*] Live Monitor v4 initialized. Waiting for stream...")
    print("[*] Simulation: reading from 'test_text.txt' line by line.\n")
    
    # Для симуляции мы будем читать файл построчно с задержкой
    # В реальности здесь был бы сокет или чтение из лога
    previous_embedding = None
    
    try:
        with open('test_text.txt', 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # Кодируем текущую строку
                current_embedding = model.encode([line])[0]
                
                if previous_embedding is not None:
                    # Считаем расстояние (энтропию) между предыдущей и текущей строкой
                    sim = cosine_similarity([previous_embedding], [current_embedding])[0][0]
                    distance = 1.0 - sim
                    
                    status = "OK"
                    if distance > THRESHOLD:
                        status = "!! ANOMALY DETECTED !!"
                    
                    print(f"[{status}] Distance: {distance:.4f} | Text: {line[:50]}...")
                
                previous_embedding = current_embedding
                time.sleep(0.5) # Задержка для имитации реального времени
                
    except FileNotFoundError:
        print("[ERROR] test_text.txt not found.")
    except KeyboardInterrupt:
        print("\n[*] Monitor stopped.")

if __name__ == "__main__":
    main()

