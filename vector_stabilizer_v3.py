import os
import re
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def smart_chunk_text(text, max_chars=350, min_chars=60):
    """
    Нарезает текст по границам предложений/абзацев, 
    не разрывая слова и сохраняя целостность мысли.
    """
    sentences = re.split(r'(?<=[.!?…])\s+|\n+', text.strip())
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if len(current_chunk) + len(sentence) + 1 <= max_chars:
            current_chunk = f"{current_chunk} {sentence}".strip()
        else:
            if len(current_chunk) >= min_chars:
                chunks.append(current_chunk)
            current_chunk = sentence

    if len(current_chunk) >= min_chars:
        chunks.append(current_chunk)

    return chunks


def calculate_semantic_metrics(embeddings, attractor_beta=0.85):
    """
    Считает две метрики:
    1. step_entropy: мгновенный скачок смысла между соседними репликами.
    2. attractor_drift: отклонение от скользящего «центра тяжести» (аттрактора) диалога.
    """
    n = len(embeddings)
    step_entropy = np.zeros(n)
    attractor_drift = np.zeros(n)

    if n == 0:
        return step_entropy, attractor_drift

    attractor = embeddings[0].copy()

    for i in range(1, n):
        # 1. Мгновенный скачок относительно предыдущего шага
        sim_step = cosine_similarity([embeddings[i - 1]], [embeddings[i]])[0][0]
        step_entropy[i] = max(0.0, 1.0 - sim_step)

        # 2. Отклонение от общего русла беседы (аттрактора)
        sim_attractor = cosine_similarity([attractor], [embeddings[i]])[0][0]
        attractor_drift[i] = max(0.0, 1.0 - sim_attractor)

        # Плавно обновляем вектор-аттрактор и нормализуем его
        attractor = attractor_beta * attractor + (1.0 - attractor_beta) * embeddings[i]
        norm = np.linalg.norm(attractor)
        if norm > 0:
            attractor = attractor / norm

    return step_entropy, attractor_drift


def stabilize_signal(signal, beta=0.8, alpha_max=0.45):
    """
    ФНЧ-маховик (EMA) + нелинейное Tanh-насыщение, 
    откалиброванное под реальный диапазон косинусных расстояний.
    """
    steps = len(signal)
    smoothed = np.zeros(steps)
    stabilized = np.zeros(steps)

    if steps > 0:
        smoothed[0] = signal[0]
        stabilized[0] = alpha_max * np.tanh(smoothed[0] / alpha_max)

        for i in range(1, steps):
            smoothed[i] = beta * smoothed[i - 1] + (1.0 - beta) * signal[i]
            stabilized[i] = alpha_max * np.tanh(smoothed[i] / alpha_max)

    return smoothed, stabilized


def export_phase_events(chunks, step_entropy, attractor_drift, 
                        step_threshold=0.38, axis_threshold=0.42, 
                        output_dir="PHASE_EVENTS"):
    """
    Селективный фильтр событий: разделяет аномалии на
    1. RESONANT_LEAP (глубокий поворот мысли при сохранении общей оси)
    2. AXIS_DRIFT (срыв с частоты / уход от цели диалога)
    """
    os.makedirs(output_dir, exist_ok=True)
    events_found = 0

    for i in range(1, len(chunks)):
        s_jump = step_entropy[i]
        a_drift = attractor_drift[i]

        event_type = None
        if a_drift >= axis_threshold:
            event_type = "AXIS_DRIFT"      # Срыв с оси (потеря резонанса / шум)
        elif s_jump >= step_threshold:
            event_type = "RESONANT_LEAP"   # Глубокий инсайт (скачок внутри темы)

        if event_type:
            events_found += 1
            filename = os.path.join(output_dir, f"event_{i:03d}_{event_type}.txt")
            with open(filename, "w", encoding="utf-8") as f:
                f.write(f"=== PHASE EVENT #{i:03d} [{event_type}] ===\n")
                f.write(f"Step Jump (Instant Impulse): {s_jump:.4f} (threshold: {step_threshold})\n")
                f.write(f"Attractor Drift (Axis Dev):  {a_drift:.4f} (threshold: {axis_threshold})\n")
                f.write("-" * 60 + "\n\n")
                f.write("PREVIOUS CHUNK (Context):\n")
                f.write(chunks[i - 1] + "\n\n")
                f.write("TRIGGER CHUNK (Phase Shift):\n")
                f.write(chunks[i] + "\n")

    print(f"[*] Exported {events_found} phase events into './{output_dir}/'")


def process_text_vector_space(filename, max_chars=350, alpha_max=0.45):
    print("[*] Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    print(f"[*] Reading and smart-chunking {filename}...")
    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read()

    chunks = smart_chunk_text(text, max_chars=max_chars)
    print(f"[*] Total semantic chunks: {len(chunks)}")

    if len(chunks) < 3:
        print("[!] Not enough chunks for phase analysis.")
        return

    print("[*] Encoding chunks into 384D vector space...")
    embeddings = model.encode(chunks)

    print("[*] Calculating step entropy & attractor drift...")
    step_entropy, attractor_drift = calculate_semantic_metrics(embeddings)

    print("[*] Applying calibrated stabilization filter...")
    _, stab_step = stabilize_signal(step_entropy, beta=0.8, alpha_max=alpha_max)
    _, stab_attractor = stabilize_signal(attractor_drift, beta=0.85, alpha_max=alpha_max)

    # Экспорт событий фазового перехода
    export_phase_events(chunks, step_entropy, attractor_drift)

    # Визуализация
    t = np.arange(len(step_entropy))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9))

    ax1.plot(t, step_entropy, label='Raw Step Jump (Noise/Impulse)', color='royalblue', alpha=0.35)
    ax1.plot(t, stab_step, label='Stabilized Step Rhythm (Tanh)', color='seagreen', linewidth=2)
    ax1.plot(t, stab_attractor, label='Attractor Drift (Axis Deviation)', color='darkorange', linewidth=2, linestyle='--')
    ax1.axhline(y=alpha_max, color='red', linestyle=':', alpha=0.6, label=f'Saturation Plateau ({alpha_max})')
    ax1.set_title('Vector Space v3: Semantic Rhythm & Selective Filtering')
    ax1.set_xlabel('Semantic Chunks (Sentence-aligned)')
    ax1.set_ylabel('Cosine Distance')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.4)

    d_raw = np.diff(step_entropy)
    d_stab = np.diff(stab_step)

    ax2.plot(step_entropy[:-1], d_raw, color='brown', alpha=0.35, label='Raw Phase Chaos')
    ax2.plot(stab_step[:-1], d_stab, color='seagreen', linewidth=2, label='Stabilized Soliton Orbit')
    ax2.set_title('Phase Portrait: Trajectory in Semantic Space')
    ax2.set_xlabel('Stabilized Amplitude (X)')
    ax2.set_ylabel('Velocity of Shift (dX/dt)')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.4)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    process_text_vector_space('test_text.txt')

