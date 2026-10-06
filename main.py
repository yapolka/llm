"""
main.py

Точка входа. С Day 4 вся логика пайплайна вынесена в pipeline.py -
main.py теперь только запускает и печатает/сохраняет результат.
"""

import json
import logging
from pathlib import Path

from llm_client import LLMClient
from pipeline import run_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


TEST_TEXTS = [
    # --- support (техническая проблема) ---
    "Приложение вылетает каждый раз, когда я пытаюсь открыть раздел 'Профиль'. Помогите разобраться.",
    "Как включить двухфакторную аутентификацию в настройках аккаунта?",

    # --- feedback (отзыв, не жалоба) ---
    "В целом сервисом доволен, но было бы здорово добавить тёмную тему в приложении.",
    "Обновление интерфейса вышло действительно удобным, стало гораздо приятнее пользоваться.",

    # --- complaint (жалоба) ---
    "Уже третий день не могу зайти в личный кабинет приложения. Пишет 'ошибка сервера'. Это возмутительно!",
    "С меня списали деньги дважды за один и тот же заказ, требую немедленного возврата!",

    # --- sales (вопрос о покупке) ---
    "Сколько стоит годовая подписка на профессиональный тариф и какие в неё входят функции?",
    "Планирую подключить сервис для команды из 20 человек - есть ли скидки на групповые тарифы?",

    # --- general_question (общий вопрос) ---
    "Подскажите, пожалуйста, в чём разница между процессорами Intel и AMD с точки зрения энергопотребления?",
    "Фотосинтез — это процесс, при котором растения преобразуют энергию света в химическую энергию. Как он устроен?",
]


def print_result(index: int, text: str, result: dict) -> None:
    print(f"\n{'=' * 60}")
    print(f"ТЕКСТ #{index}: {text[:70]}...")
    print(f"{'=' * 60}")

    if "error" in result:
        print(f"⚠️  Ошибка на этапе '{result.get('stage', '?')}': {result['error']}")
        if "details" in result:
            print(f"Причина: {result['details']}")
        return

    print(f"Категория: {result.get('category')} | Intent: {result.get('intent')}")
    print(f"Sentiment: {result.get('sentiment')} | Urgency: {result.get('urgency')}")
    print(f"Summary: {result.get('summary')}")
    print("Key points:")
    for point in result.get("key_points", []):
        print(f"  • {point}")
    print(f"Final answer: {result.get('final_answer')}")

    check = result.get("self_check", {})
    print(f"\nSelf-check: consistent={check.get('consistent')}, verdict={check.get('verdict')}")
    if check.get("issues"):
        print("Найденные проблемы:")
        for issue in check["issues"]:
            print(f"  ⚠ {issue}")


def main():
    logger.info("Запуск пайплайна Day 4 — классификация и routing")
    client = LLMClient()

    all_results = []

    for i, text in enumerate(TEST_TEXTS, start=1):
        logger.info("Обработка текста #%d из %d", i, len(TEST_TEXTS))
        result = run_pipeline(text, client)
        print_result(i, text, result)
        all_results.append({"input_text": text, "result": result})

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "day5_results.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    logger.info("Готово! Результаты сохранены в %s", output_path)


if __name__ == "__main__":
    main()