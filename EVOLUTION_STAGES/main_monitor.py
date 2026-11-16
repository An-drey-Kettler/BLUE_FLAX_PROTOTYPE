import numpy as np
import matplotlib.pyplot as plt
import math
from collections import Counter

# --- Модуль 1: Энтропия ---
def calculate_entropy(text):
    if not text: return 0.0
    chars = list(text.lower())
    counts = Counter(chars)
    total = len(chars)
    return -sum((count / total) * math.log2(count / total) for count in counts.values())

# --- Модуль 2: Стабилизация (EMA + Tanh) ---
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

# --- Модуль 3: Фазовая синхронизация (R_in/out) ---
def calculate_r_index(signal_in, signal_out, window_size=10):
    r_moving = []
    for i in range(len(signal_in) - window_size):
        x = signal_in[i:i+window_size]
        y = signal_out[i:i+window_size]
        
        x_c = x - np.mean(x)
        y_c = y - np.mean(y)
        
        if np.std(x_c) == 0 or np.std(y_c) == 0:
            r_moving.append(0.0)
        else:
            r = np.corrcoef(x_c, y_c)[0, 1]
            r_moving.append(r)
    return np.array(r_moving)

# --- ГЛАВНЫЙ КОНВЕЙЕР ---
def run_full_diagnostics(filename, chunk_size=200):
    print(f"[*] Запуск диагностики для файла: {filename}")
    
    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read()
    
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    entropies = np.array([calculate_entropy(chunk) for chunk in chunks])
    print(f"[*] Текст разбит на {len(chunks)} фрагментов.")
    
    # Стабилизация
    smoothed, final_alpha = stabilize_signal(entropies)
    
    # Расчет синхронизации
    r_index = calculate_r_index(entropies, final_alpha)
    
    # Статистика
    avg_r = np.mean(r_index)
    min_r = np.min(r_index)
    
    print("\n=== РЕЗУЛЬТАТЫ ДИАГНОСТИКИ ===")
    print(f"Средний индекс синхронизации (R): {avg_r:.3f}")
    print(f"Минимальный индекс (самый сильный сбой): {min_r:.3f}")
    
    if min_r < 0.0:
        print("[!] ВНИМАНИЕ: Обнаружен разрыв контекста (хаос).")
    elif avg_r > 0.7:
        print("[+] СИСТЕМА СТАБИЛЬНА: Высокий уровень когерентности.")
    else:
        print("[-] УМЕРЕННЫЙ ДРЕЙФ: Текст имеет нестабильную структуру.")
    
        # --- Отрисовка общего графика ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    ax1.plot(entropies, label='Raw Entropy', color='blue', alpha=0.5)
    ax1.plot(final_alpha, label='Stabilized', color='green')
    ax1.set_title('Entropy Diagnostics')  # <-- БЫЛО: 'Диагностика энтропии'
    ax1.set_ylabel('Entropy (bits)')      # <-- ДОБАВИТЬ (если нет)
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(r_index, label='R_in/out', color='purple')
    ax2.axhline(y=0, color='red', linestyle='--', label='Chaos Line')
    ax2.set_title('Phase Locking Index (R)') # <-- БЫЛО: 'Индекс фазовой синхронизации (R)'
    ax2.set_ylabel('Correlation')            # <-- ДОБАВИТЬ (если нет)
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout()
    plt.show()
if __name__ == "__main__":
    run_full_diagnostics('test_text.txt')

