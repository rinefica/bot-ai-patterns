from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

import openai

from bot_ai_patterns.config import MAX_TOKENS, MODEL_URI
from bot_ai_patterns.prompts import EXPERT_PROMPTS, META_PROMPT_GENERATOR, META_PROMPT_HTML_SUFFIX, NO_PROMPT, STEP_BY_STEP_PROMPT


@dataclass
class StrategyResult:
    name: str
    sections: dict[str, str] = field(default_factory=dict)
    error: str | None = None


class Strategy(ABC):
    name: str

    def run(self, client: openai.OpenAI, user_message: str) -> StrategyResult:
        try:
            return self._execute(client, user_message)
        except Exception as e:
            return StrategyResult(name=self.name, error=str(e))

    @abstractmethod
    def _execute(self, client: openai.OpenAI, user_message: str) -> StrategyResult:
        ...

    def _request(self, client: openai.OpenAI, messages: list[dict]) -> str:
        response = client.chat.completions.create(
            model=MODEL_URI,
            messages=messages,
            max_tokens=MAX_TOKENS,
        )
        content = response.choices[0].message.content or ""
        content = content.strip()
        if content.startswith("```html"):
            content = content[len("```html"):].lstrip("\n")
        if content.endswith("```"):
            content = content[:-3].rstrip("\n")
        return content


class NoPromptStrategy(Strategy):
    name = "Без промпта"

    def _execute(self, client: openai.OpenAI, user_message: str) -> StrategyResult:
        result = self._request(client, [
            {"role": "system", "content": NO_PROMPT},
            {"role": "user", "content": user_message},
        ])
        return StrategyResult(name=self.name, sections={"Ответ": result})


class StepByStepStrategy(Strategy):
    name = "Пошаговое решение"

    def _execute(self, client: openai.OpenAI, user_message: str) -> StrategyResult:
        result = self._request(client, [
            {"role": "system", "content": STEP_BY_STEP_PROMPT},
            {"role": "user", "content": user_message},
        ])
        return StrategyResult(name=self.name, sections={"Ответ": result})


class MetaPromptStrategy(Strategy):
    name = "Мета-промпт"

    def _execute(self, client: openai.OpenAI, user_message: str) -> StrategyResult:
        print("  [Мета-промпт] Генерирую промпт...", flush=True)
        generated_prompt = self._request(client, [
            {"role": "system", "content": META_PROMPT_GENERATOR},
            {"role": "user", "content": user_message},
        ])
        print("  [Мета-промпт] Решаю задачу по промпту...", flush=True)
        result = self._request(client, [
            {"role": "system", "content": generated_prompt + META_PROMPT_HTML_SUFFIX},
            {"role": "user", "content": user_message},
        ])
        return StrategyResult(
            name=self.name,
            sections={"Сгенерированный промпт": generated_prompt, "Ответ": result},
        )


class ExpertsStrategy(Strategy):
    name = "Группа экспертов"

    def _execute(self, client: openai.OpenAI, user_message: str) -> StrategyResult:
        sections: dict[str, str] = {}
        with ThreadPoolExecutor(max_workers=len(EXPERT_PROMPTS)) as executor:
            futures = {
                executor.submit(self._request, client, [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user_message},
                ]): expert_name
                for expert_name, prompt in EXPERT_PROMPTS.items()
            }
            for future in as_completed(futures):
                expert_name = futures[future]
                try:
                    sections[expert_name] = future.result()
                except Exception as e:
                    sections[expert_name] = f"Ошибка: {e}"
        return StrategyResult(name=self.name, sections=sections)


STRATEGIES: list[Strategy] = [
    NoPromptStrategy(),
    StepByStepStrategy(),
    MetaPromptStrategy(),
    ExpertsStrategy(),
]
