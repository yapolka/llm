"""
utils.py

Вспомогательные функции общего назначения, используемые
в нескольких местах проекта.
"""

import json


def parse_model_json(raw_text: str) -> dict:
    """
    Пытается превратить сырой текст ответа модели в Python dict.
    Обрабатывает случай, когда модель обернула JSON в блок кода
    (```json ... ```), хотя мы просили этого не делать.
    """
    cleaned = raw_text.strip()

    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()

    return json.loads(cleaned)
