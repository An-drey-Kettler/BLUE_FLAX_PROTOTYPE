import numpy as np
import matplotlib.pyplot as plt

def simulate_phase_lock(steps=100, alpha_base=0.5, beta=0.8, alpha_max=6.0):
    # 1. Генерируем "грязный" входной сигнал (имитация резких скачков контекста)
    # В реальности это была бы энтропия текста, здесь мы просто делаем резкий скачок
    t = np.arange(steps)
    raw_signal = np.zeros(steps)
    raw_signal[20:50] = 1.0  # Резкий скачок (шок)
    raw_signal[50:80] = -0.5 # Обратный скачок
    
    # Добавляем шум (Hardware noise)
    noise = np.random.normal(0, 0.1, steps)
    input_signal = raw_signal + noise

    # 2. Применяем фильтрацию (EMA) и Tanh-насыщение (твоя идея)
    smoothed = np.zeros(steps)
    final_alpha = np.zeros(steps)
    
    # Инициализация
    smoothed[0] = input_signal[0]
    
    for i in range(1, steps):
        # ФНЧ-маховик (EMA)
        smoothed[i] = beta * smoothed[i-1] + (1 - beta) * input_signal[i]
        
        # Tanh-насыщение (жесткое плато)
        # Мы применяем tanh к сырому или сглаженному? В твоих графиках - к сглаженному.
        # Используем формулу: alpha_max * tanh(smoothed / alpha_max)
        final_alpha[i] = alpha_max * np.tanh(smoothed[i] / alpha_max)

    # 3. Построение графиков
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # График 1: Сглаживание и насыщение
    ax1.plot(t, input_signal, label='Input Signal (with noise)', linestyle='--', alpha=0.6)
    ax1.plot(t, smoothed, label='Smoothed (EMA)', linewidth=2)
    ax1.plot(t, final_alpha, label='Final (Tanh Saturation)', linewidth=2)
    ax1.set_title('Phase Lock: EMA Smoothing & Tanh Saturation')
    ax1.set_xlabel('Time steps')
    ax1.set_ylabel('Amplitude')
    ax1.legend()
    ax1.grid(True)

    # График 2: Фазовый портрет (Chaos vs Soliton)
    # Берем производную (скорость изменения) для фазового портрета
    d_input = np.diff(input_signal)
    d_final = np.diff(final_alpha)

    # Рисуем хаос (сырой сигнал)
    ax2.plot(input_signal[:-1], d_input, color='brown', alpha=0.5, label='Dissipative Chaos (Raw)')
    # Рисуем солитон (стабилизированный сигнал)
    ax2.plot(final_alpha[:-1], d_final, color='green', linewidth=2, label='NESS Soliton (Filtered)')
    
    ax2.set_title('Phase Portrait: Attractor vs Chaos')
    ax2.set_xlabel('State Amplitude (X)')
    ax2.set_ylabel('Rate of Change (dX/dt)')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    simulate_phase_lock()

