import openai

from bot_ai_patterns.config import MODEL_URI, MAX_TOKENS


class Agent:
    """Агент, инкапсулирующий логику общения с LLM.

    Хранит историю диалога и отправляет запросы к API через синхронный HTTP-клиент.
    """

    def __init__(self, client: openai.OpenAI, system_prompt: str = "Ты полезный ассистент."):
        self._client = client
        self._history: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]

    def chat(self, user_input: str) -> str:
        self._history.append({"role": "user", "content": user_input})

        response = self._client.chat.completions.create(
            model=MODEL_URI,
            messages=self._history,
            max_tokens=MAX_TOKENS,
        )

        reply = response.choices[0].message.content or ""
        self._history.append({"role": "assistant", "content": reply})
        return reply

    def reset(self) -> None:
        system = self._history[0]
        self._history = [system]
