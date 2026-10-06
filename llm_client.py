"""
llm_client.py

Единственное место в проекте, которое знает, КАК именно
мы обращаемся к LLM API. Остальной код просто вызывает
client.generate(prompt) и получает текст — ему не важно,
какой именно провайдер под капотом.
"""

import os
import time
import logging
import openai

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

logger = logging.getLogger(__name__)


class LLMClient:
    """Тонкая обёртка над LLM API."""

    def __init__(self, model: str | None = None):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY не найден. "
                "Скопируй .env.example в .env и вставь туда свой ключ."
            )

        base_url = os.getenv("OPENAI_BASE_URL")

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

        # Максимальная длина ответа в токенах. На бесплатном Groq лимит
        # на выходные токены в минуту ~1000, поэтому держим значение ниже.
        self.max_tokens = int(os.getenv("MAX_TOKENS", "700"))

        # Минимальная пауза между запросами (секунды), чтобы не упираться
        # в лимиты запросов/токенов в минуту.
        self.min_interval = float(os.getenv("MIN_REQUEST_INTERVAL", "3"))
        self._last_request_time = 0.0

    def generate(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
    ) -> str:
        """
        Отправляет промпт в модель и возвращает "сырой" текст ответа.

        temperature необязательный (по умолчанию None) - некоторые
        модели (например, reasoning-модели) не поддерживают кастомную
        температуру. Если temperature не передан явно - просто не
        отправляем этот параметр, пусть провайдер сам решает.
        """
        logger.info("-> Отправка запроса к модели '%s'", self.model)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        request_kwargs = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
        }
        if temperature is not None:
            request_kwargs["temperature"] = temperature

        # Пауза: не чаще одного запроса в min_interval секунд
        elapsed = time.time() - self._last_request_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)

        try:
            response = self.client.chat.completions.create(**request_kwargs)
        except openai.APIError as e:
            logger.error("Ошибка API при обращении к модели: %s", e)
            return ""
        finally:
            self._last_request_time = time.time()

        if not response.choices:
            provider_error = getattr(response, "error", None)
            logger.error("Провайдер не вернул ответ. Детали: %s", provider_error)
            return ""

        content = response.choices[0].message.content or ""
        logger.info("<- Ответ получен (%d символов)", len(content))

        return content
