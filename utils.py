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

    Исключения:
    - json.JSONDecodeError - текст вообще не JSON;
    - ValueError - валидный JSON, но не объект (например, [], "text", 42).
    JSONDecodeError наследуется от ValueError, поэтому вызывающий код
    может ловить оба случая одним `except ValueError`.
    """
    cleaned = raw_text.strip()

    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()

    data = json.loads(cleaned)

    if not isinstance(data, dict):
        raise ValueError(
            f"Ожидался JSON-объект ({{...}}), а получен {type(data).__name__}"
        )

    return data

