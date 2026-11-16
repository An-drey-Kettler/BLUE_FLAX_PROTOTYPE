import numpy as np
import matplotlib.pyplot as plt
import math
from collections import Counter

def calculate_entropy(text):
    if not text: return 0.0
    chars = list(text.lower())
    counts = Counter(chars)
    total = len(chars)
    return -sum((count / total) * math.log2(count / total) for count in counts.values())

def calculate_phase_locking_index(input_signal, output_signal, tau=1):
    """Считает нормированную корреляцию со сдвигом (R_in/out)"""
    if len(input_signal) <= tau:
        return 0.0
    # Берем срезы со сдвигом
    x = input_signal[:-tau]
    y = output_signal[tau:]
    
    # Центрируем
    x = x - np.mean(x)
    y = y - np.mean(y)
    
    # Считаем корреляцию
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    
    correlation = np.corrcoef(x, y)[0, 1]
    return correlation

def process_text_file(filename, chunk_size=200):
    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read()
    
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    entropies = np.array([calculate_entropy(chunk) for chunk in chunks])
    
    steps = len(entropies)
    smoothed = np.zeros(steps)
    final_alpha = np.zeros(steps)
    
    beta = 0.8
    alpha_max = 6.0 
    
    if steps > 0:
        smoothed[0] = entropies[0]
        final_alpha[0] = alpha_max * np.tanh(smoothed[0] / alpha_max)
        
        for i in range(1, steps):
            smoothed[i] = beta * smoothed[i-1] + (1 - beta) * entropies[i]
            final_alpha[i] = alpha_max * np.tanh(smoothed[i] / alpha_max)

    # Расчет индекса фазовой синхронизации (R_in/out)
    # Здесь входной сигнал - это энтропия, а выходной - наш стабилизированный alpha
    # Это покажет, насколько мы "отклонились" или "синхронизировались" с потоком
    r_index = calculate_phase_locking_index(entropies, final_alpha, tau=1)
    
    # Делаем скользящее окно для R, чтобы увидеть его изменение во времени
    window_size = 10
    r_moving = []
    for i in range(len(entropies) - window_size):
        r = calculate_phase_locking_index(
            entropies[i:i+window_size], 
            final_alpha[i:i+window_size], 
            tau=1
        )
        r_moving.append(r)

    # Построение графиков
    t = np.arange(steps)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # График 1: Энтропия и фильтр (как раньше)
    ax1.plot(t, entropies, label='Raw Text Entropy', linestyle='--', alpha=0.5, color='blue')
    ax1.plot(t, smoothed, label='Smoothed (EMA)', linewidth=2, color='orange')
    ax1.plot(t, final_alpha, label='Final (Tanh Saturation)', linewidth=2, color='green')
    ax1.set_title('Real Text: Entropy vs Stabilization')
    ax1.set_xlabel('Text chunks')
    ax1.set_ylabel('Entropy (bits)')
    ax1.legend()
    ax1.grid(True)

    # График 2: Индекс фазовой синхронизации (R_in/out)
    # Ось X сдвигаем на размер окна, так как R_moving короче
    t_r = np.arange(len(r_moving)) + window_size
    
    # Порог идеальной синхронизации из твоих PDF - 0.95. 
    # Но поскольку текст - это шум, мы нарисуем и порог стабильности (например, 0.5)
    ax2.plot(t_r, r_moving, label='R_in/out (Moving Window)', color='purple', linewidth=2)
    ax2.axhline(y=0.95, color='red', linestyle='--', alpha=0.5, label='Ideal Lock (0.95)')
    ax2.axhline(y=0.0, color='gray', linestyle=':', alpha=0.5, label='Chaos (0.0)')
    
    ax2.set_title('Phase Locking Index (R_in/out)')
    ax2.set_xlabel('Text chunks')
    ax2.set_ylabel('Correlation')
    ax2.set_ylim(-1.1, 1.1)
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    process_text_file('test_text.txt')

