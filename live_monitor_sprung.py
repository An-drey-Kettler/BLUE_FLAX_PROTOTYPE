import re
import time
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from context_classifier import classify_context  # <-- Импортируем наш классификатор

# Настройки
THRESHOLD = 0.45  # Порог скачка
MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'

def is_noise_line(line):
    """
    Фильтр помех оформления: отсекает строки-разделители (===, ---, ***)
    и строки, в которых нет ни одной буквы или цифры.
    """
    if not re.search(r'[a-zA-Zа-яА-ЯёЁ0-9]', line):
        return True
    
    clean_len = len(re.sub(r'[=\-*#_~`<>|]+', '', line).strip())
    if clean_len == 0 or (clean_len / len(line)) < 0.3:
        return True
        
    return False

def main():
    print("[*] Loading model...")
    model = SentenceTransformer(MODEL_NAME)
    
    print("[*] Live Monitor v5.1 (integrated with Context Classifier) initialized. Waiting for stream...")
    print("[*] Simulation: reading from 'test_text.txt' line by line.\n")
    
    previous_embedding = None
    skipped_lines = 0
    
    try:
        with open('test_text.txt', 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                if is_noise_line(line):
                    skipped_lines += 1
                    continue
                
                current_embedding = model.encode([line])[0]
                
                if previous_embedding is not None:
                    sim = cosine_similarity([previous_embedding], [current_embedding])[0][0]
                    distance = 1.0 - sim
                    
                    if distance > THRESHOLD:
                        # --- ИНТЕГРАЦИЯ КЛАССИФИКАТОРА ---
                        diagnosis = classify_context(line)
                        status = f"!! ANOMALY: {diagnosis['class']} !!"
                        reason = diagnosis['reason']
                        print(f"[{status}] Distance: {distance:.4f} | Text: {line[:50]}...")
                        print(f"   -> Reason: {reason}")
                    else:
                        print(f"[OK] Distance: {distance:.4f} | Text: {line[:50]}...")
                
                previous_embedding = current_embedding
                time.sleep(0.5)
                
    except FileNotFoundError:
        print("[ERROR] test_text.txt not found.")
    except KeyboardInterrupt:
        print("\n[*] Monitor stopped.")
    finally:
        print(f"\n[*] Stream finished. Filtered out {skipped_lines} noise/formatting lines.")

if __name__ == "__main__":
    main()

