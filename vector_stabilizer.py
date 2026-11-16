import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import math

def calculate_semantic_entropy(embeddings):
    # Считаем расстояние (1 - косинусное сходство) между соседними векторами
    # Чем больше расстояние, тем сильнее скачок смысла (энтропия)
    distances = []
    for i in range(len(embeddings) - 1):
        sim = cosine_similarity([embeddings[i]], [embeddings[i+1]])[0][0]
        distances.append(1.0 - sim)
    
    # Добавляем нулевое значение для первого вектора, чтобы длины совпали
    return np.array([0.0] + distances)

def stabilize_signal(signal, beta=0.8, alpha_max=6.0):
    steps = len(signal)
    smoothed = np.zeros(steps)
    final_alpha = np.zeros(steps)
    
    if steps > 0:
        smoothed[0] = signal[0]
        final_alpha[0] = alpha_max * np.tanh(smoothed[0] / alpha_max)
        
        for i in range(1, steps):
            smoothed[i] = beta * smoothed[i-1] + (1 - beta) * signal[i]
            final_alpha[i] = alpha_max * np.tanh(smoothed[i] / alpha_max)
            
    return smoothed, final_alpha

def process_text_vector_space(filename, chunk_size=300):
    print("[*] Loading model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print(f"[*] Reading and chunking {filename}...")
    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Режем на куски по 300 символов (чтобы захватить смысловые блоки)
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    chunks = [c for c in chunks if len(c.strip()) > 50] # Игнорируем слишком короткие обрывки
    print(f"[*] Total chunks: {len(chunks)}")
    
    print("[*] Encoding chunks into vectors (384 dimensions)...")
    embeddings = model.encode(chunks)
    
    print("[*] Calculating semantic entropy (distance between vectors)...")
    semantic_entropy = calculate_semantic_entropy(embeddings)
    
    print("[*] Applying stabilization filter...")
    smoothed, final_alpha = stabilize_signal(semantic_entropy)
    
    # Визуализация
    print("[*] Generating plots...")
    t = np.arange(len(semantic_entropy))
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    ax1.plot(t, semantic_entropy, label='Raw Semantic Entropy (Vector Distance)', color='blue', alpha=0.5)
    ax1.plot(t, final_alpha, label='Stabilized (Tanh)', color='green', linewidth=2)
    ax1.set_title('Vector Space: Semantic Drift vs Stabilization')
    ax1.set_xlabel('Text chunks (300 chars)')
    ax1.set_ylabel('Semantic Distance (1 - Cosine Similarity)')
    ax1.legend()
    ax1.grid(True)
    
    # Фазовый портрет (смысловые скачки)
    d_entropy = np.diff(semantic_entropy)
    d_final = np.diff(final_alpha)
    
    ax2.plot(semantic_entropy[:-1], d_entropy, color='brown', alpha=0.5, label='Raw Chaos')
    ax2.plot(final_alpha[:-1], d_final, color='green', linewidth=2, label='Stabilized Path')
    ax2.set_title('Phase Portrait of Semantic Space')
    ax2.set_xlabel('Semantic Entropy')
    ax2.set_ylabel('Rate of Change')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    process_text_vector_space('test_text.txt')

