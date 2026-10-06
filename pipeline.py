"""
pipeline.py

Пайплайн из 5 последовательных шагов, каждый использует результат
предыдущего:
1. extract_meaning         - извлечение сути (summary + key_points)
2. classify_request         - определение категории и intent
3. build_structured_fields  - sentiment + urgency
4. generate_final_answer    - routing + генерация ответа
5. self_check                - проверка результата на противоречия
"""

import json
import logging
import time

from pydantic import ValidationError

from llm_client import LLMClient
from prompts import (
    MEANING_SYSTEM_PROMPT, build_meaning_prompt,
    CLASSIFY_SYSTEM_PROMPT, build_classify_prompt,
    STRUCTURED_FIELDS_SYSTEM_PROMPT, build_structured_fields_prompt,
    FINAL_ANSWER_SYSTEM_PROMPT_TEMPLATE, build_final_answer_prompt,
    SELF_CHECK_SYSTEM_PROMPT, build_self_check_prompt,
    ROUTING_INSTRUCTIONS,
)
from schemas import (
    MeaningExtraction, ClassificationResult, StructuredFields,
    FinalAnswerResult, SelfCheckResult,
)
from utils import parse_model_json

logger = logging.getLogger(__name__)


MAX_RETRIES = 2  # доп. попытки после первой (итого до 3 попыток на шаг)
RETRY_DELAY_SECONDS = 15


def _call_step(step_name, user_prompt, system_prompt, schema_cls, client):
    """
    Обёртка для одного шага с retry и fallback (Day 6):
    - при empty_response - просто повторяем попытку
    - при невалидном JSON - повторяем, добавив явное напоминание
    - при ошибке валидации схемы - повторяем, ВКЛЮЧАЯ текст ошибки
      от Pydantic прямо в промпт, чтобы модель поняла, что исправить
    """
    attempt = 0
    current_prompt = user_prompt

    while attempt <= MAX_RETRIES:
        attempt += 1
        logger.info("  [шаг: %s] попытка %d/%d", step_name, attempt, MAX_RETRIES + 1)

        raw_response = client.generate(current_prompt, system_prompt=system_prompt)

        if not raw_response:
            logger.warning("  [шаг: %s] пустой ответ, повторяем...", step_name)
            time.sleep(RETRY_DELAY_SECONDS)
            continue

        try:
            raw_dict = parse_model_json(raw_response)
        except json.JSONDecodeError:
            logger.warning("  [шаг: %s] невалидный JSON, повторяем с напоминанием...", step_name)
            current_prompt = (
                user_prompt
                + "\n\nВАЖНО: предыдущий ответ не был валидным JSON. "
                "Верни ТОЛЬКО валидный JSON, без какого-либо текста вокруг."
            )
            time.sleep(RETRY_DELAY_SECONDS)
            continue

        try:
            validated = schema_cls(**raw_dict)
        except ValidationError as e:
            logger.warning("  [шаг: %s] ошибка валидации, повторяем с деталями ошибки...", step_name)
            current_prompt = (
                user_prompt
                + f"\n\nВАЖНО: предыдущая попытка не прошла проверку:\n{e}\n"
                "Исправь именно эту проблему и верни корректный JSON."
            )
            time.sleep(RETRY_DELAY_SECONDS)
            continue

        logger.info("  [шаг: %s] успех с попытки %d/%d", step_name, attempt, MAX_RETRIES + 1)
        return validated, None

    logger.error("  [шаг: %s] все %d попыток исчерпаны, шаг провален", step_name, MAX_RETRIES + 1)
    return None, {"error": "step_failed_after_retries", "stage": step_name, "attempts": attempt}


def run_pipeline(text: str, client: LLMClient) -> dict:
    """Полный 5-шаговый пайплайн Day 5."""

    # Шаг 1: extract meaning
    meaning, err = _call_step(
        "extract_meaning", build_meaning_prompt(text),
        MEANING_SYSTEM_PROMPT, MeaningExtraction, client,
    )
    if err:
        return err

    # Шаг 2: classify request
    classification, err = _call_step(
        "classify_request", build_classify_prompt(text),
        CLASSIFY_SYSTEM_PROMPT, ClassificationResult, client,
    )
    if err:
        return err

    # Шаг 3: build structured fields (использует summary из шага 1)
    fields, err = _call_step(
        "build_structured_fields",
        build_structured_fields_prompt(text, meaning.summary),
        STRUCTURED_FIELDS_SYSTEM_PROMPT, StructuredFields, client,
    )
    if err:
        return err

    # Шаг 4: generate final answer (использует ВСЁ из шагов 1-3)
    routing_instruction = ROUTING_INSTRUCTIONS.get(
        classification.category, "Дай нейтральный, полезный ответ по существу."
    )
    final_system_prompt = FINAL_ANSWER_SYSTEM_PROMPT_TEMPLATE.format(
        routing_instruction=routing_instruction
    )
    final, err = _call_step(
        "generate_final_answer",
        build_final_answer_prompt(
            text, meaning.summary, meaning.key_points,
            fields.sentiment, fields.urgency,
        ),
        final_system_prompt, FinalAnswerResult, client,
    )
    if err:
        return err

    # Шаг 5: self-check (использует key_points из шага 1 и ответ из шага 4)
    check, err = _call_step(
        "self_check",
        build_self_check_prompt(text, meaning.key_points, final.final_answer),
        SELF_CHECK_SYSTEM_PROMPT, SelfCheckResult, client,
    )
    if err:
        return err

    return {
        "category": classification.category,
        "intent": classification.intent,
        "summary": meaning.summary,
        "key_points": meaning.key_points,
        "sentiment": fields.sentiment,
        "urgency": fields.urgency,
        "final_answer": final.final_answer,
        "self_check": check.model_dump(),
    }