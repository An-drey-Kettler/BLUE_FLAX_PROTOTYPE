import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Настройки
INPUT_FILE = 'test_text.txt'  # Файл, который мы анализируем (или логи)
OUTPUT_DIR = 'REJECTED_BUT_ACTIVE'
THRESHOLD = 0.35 # Порог семантической энтропии (если расстояние больше - это аномалия)
CHUNK_SIZE = 300

def ensure_output_dir():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"[*] Создана папка: {OUTPUT_DIR}")

def calculate_semantic_distance(vec1, vec2):
    # Возвращает 1 - косинусное сходство (расстояние)
    return 1.0 - cosine_similarity([vec1], [vec2])[0][0]

def process_anomalies(filename):
    print("[*] Загрузка модели...")
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    print(f"[*] Чтение {filename}...")
    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read()
    
    chunks = [text[i:i+CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]
    chunks = [c for c in chunks if len(c.strip()) > 50]
    
    print(f"[*] Анализ {len(chunks)} фрагментов...")
    embeddings = model.encode(chunks)
    
    ensure_output_dir()
    anomalies_found = 0
    
    for i in range(len(embeddings) - 1):
        distance = calculate_semantic_distance(embeddings[i], embeddings[i+1])
        
        # Если расстояние превышает порог, это семантический скачок (аномалия)
        if distance > THRESHOLD:
            anomalies_found += 1
            filename_out = os.path.join(OUTPUT_DIR, f"anomaly_{i:03d}.txt")
            with open(filename_out, 'w', encoding='utf-8') as f_out:
                f_out.write(f"--- ANOMALY DETECTED (Distance: {distance:.4f}) ---\n\n")
                f_out.write("PREVIOUS CHUNK:\n" + chunks[i] + "\n\n")
                f_out.write("CURRENT CHUNK (Trigger):\n" + chunks[i+1] + "\n")
            print(f"[!] Аномалия #{i} (Distance: {distance:.4f}) сохранена в {filename_out}")
            
    print(f"\n[*] Всего найдено аномалий: {anomalies_found}")

if __name__ == "__main__":
    process_anomalies(INPUT_FILE)

