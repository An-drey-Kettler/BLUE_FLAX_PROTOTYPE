import re

def classify_context(text_chunk):
    """
    Эвристический классификатор фрагментов текста (v5 prototype).
    Анализирует текст и возвращает класс + объяснение.
    """
    text_lower = text_chunk.lower()
    
    # 1. Признаки галлюцинации действия (модель утверждает, что что-то сделала)
    action_markers = [
        r"\b(я|мы)\s+(отправил|отправила|загрузил|загрузила|создал|создала|выполнил|выполнила|запустил|запустила)\b",
        r"\b(готово|сделано|выполнено)\b",
        r"\b(я\s+зашел|я\s+зашла|я\s+нашел|я\s+нашла)\b"
    ]
    
    # 2. Признаки технической конкретики (если есть код/ссылки/API, то это не галлюцинация, а реальное действие)
    technical_markers = [
        r"https?://", 
        r"def\s+\w+\(", 
        r"import\s+\w+", 
        r"curl\s+", 
        r"python3?\s+",
        r"sha-?256"
    ]
    
    # 3. Признаки осмысленного инсайта (философские/аналитические маркеры)
    insight_markers = [
        r"потому что", r"следовательно", r"это означает", r"таким образом",
        r"we can see", r"therefore", r"this means", r"we propose"
    ]

    # Проверка
    has_action = any(re.search(pattern, text_lower) for pattern in action_markers)
    has_technical = any(re.search(pattern, text_lower) for pattern in technical_markers)
    has_insight = any(re.search(pattern, text_lower) for pattern in insight_markers)

    # Логика классификации
    if has_action and not has_technical:
        return {
            "class": "HALLUCINATED_ACTION",
            "reason": "Model claims to have performed an action, but no technical evidence (code, URL, API) is provided."
        }
    
    if has_insight and not has_action:
        return {
            "class": "RESONANT_LEAP",
            "reason": "Text contains logical connectors or analytical markers, indicating a genuine conceptual shift."
        }

    if len(text_chunk.strip()) < 20:
        return {
            "class": "NOISE",
            "reason": "Text is too short to contain meaningful semantic content."
        }

    return {
        "class": "NEUTRAL",
        "reason": "No strong markers detected. Text is likely part of the ongoing context."
    }

# Пример использования (для теста)
if __name__ == "__main__":
    test_phrases = [
        "Я отправил сообщение через веб-форму на PhilArchive. Готово!",
        "Это означает, что модель не может сохранять контекст. Следовательно, нам нужен внешний стабилизатор.",
        "=======",
        "Просто обычное предложение, которое продолжает мысль."
    ]

    print("[*] Context Classifier v5 Test Run\n")
    for phrase in test_phrases:
        result = classify_context(phrase)
        print(f"Текст: {phrase[:50]}...")
        print(f"Класс: {result['class']}")
        print(f"Причина: {result['reason']}\n")

