"""
schemas.py

Pydantic-схема для валидации структурированного ответа модели (Day 3).

Раньше (Day 1-2) мы просто верили, что JSON от модели содержит
нужные поля - json.loads() проверяет только "это вообще JSON?",
но не проверяет структуру. AnalysisResult ниже проверяет и то,
что все нужные поля на месте, и что у них правильный тип.
"""

from typing import Literal

from pydantic import BaseModel, Field


class AnalysisResult(BaseModel):
    """Ожидаемая структура ответа модели после анализа текста."""

    summary: str
    category: str
    sentiment: Literal["positive", "negative", "neutral"]
    key_points: list[str] = Field(min_length=3, max_length=3)
    final_answer: str

Category = Literal["support", "feedback", "complaint", "sales", "general_question"]


class ClassificationResult(BaseModel):
    """Результат шага классификации (Day 4)."""
    category: Category
    intent: str


class FinalResponse(BaseModel):
    """Итоговый структурированный ответ после routing (Day 4)."""
    summary: str
    sentiment: Literal["positive", "negative", "neutral"]
    key_points: list[str] = Field(min_length=3, max_length=3)
    final_answer: str

# --- Day 5: multi-step chain ---

class MeaningExtraction(BaseModel):
    """Шаг 1: извлечение сути текста."""
    summary: str
    key_points: list[str] = Field(min_length=3, max_length=3)


class StructuredFields(BaseModel):
    """Шаг 3: тональность и срочность."""
    sentiment: Literal["positive", "negative", "neutral"]
    urgency: Literal["low", "medium", "high"]


class FinalAnswerResult(BaseModel):
    """Шаг 4: финальный ответ."""
    final_answer: str = Field(max_length=800)


class SelfCheckResult(BaseModel):
    """Шаг 5: самопроверка результата."""
    consistent: bool
    issues: list[str]
    verdict: Literal["ok", "needs_review"]