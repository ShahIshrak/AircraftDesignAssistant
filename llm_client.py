import requests
import time
from dotenv import load_dotenv
import os

FEATHERLESS_URL = "https://api.featherless.ai/v1/chat/completions"
FEATHERLESS_MODEL = "Cannae-AI/Gemini-3.1-Pro-Qwen3-14B"


class OllamaClient:
    def __init__(self, model=FEATHERLESS_MODEL, temperature=0.7, num_ctx=None, max_tokens=2048):
        self.model = model
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.max_tokens = max_tokens

        self.api_key = os.getenv("FEATHERLESS_API_KEY")

        self.system_prompt = """
You are an Aircraft Design Assistant. You will help take airfoil related decisions and decide other design parameters from your knowledge
"""

    # -----------------------------
    # CORE GENERATION (Featherless)
    # -----------------------------
    def _call_api(self, messages, max_tokens=None):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }

        if max_tokens:
            payload["max_tokens"] = max_tokens
        elif self.max_tokens:
            payload["max_tokens"] = self.max_tokens

        response = requests.post(FEATHERLESS_URL, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        return response.json()

    # -----------------------------
    # SIMPLE GENERATE
    # -----------------------------
    def generate(self, prompt: str, system_prompt: str = "", on_token=None, max_tokens: int = None):
        import threading

        print(
            f"[FEATHERLESS GENERATE] thread: {threading.current_thread().name} | "
            f"is_main: {threading.current_thread() is threading.main_thread()}"
        )

        effective_system = self.system_prompt.strip()
        if system_prompt:
            effective_system += "\n\n" + system_prompt.strip()

        messages = []
        if effective_system:
            messages.append({"role": "system", "content": effective_system})
        messages.append({"role": "user", "content": prompt})

        data = self._call_api(messages, max_tokens=max_tokens)

        text = data["choices"][0]["message"]["content"]

        # Simulated streaming (Featherless usually returns full response)
        if on_token:
            for ch in text:
                on_token(ch)
                time.sleep(0.001)
            on_token(None)

        return text

    # -----------------------------
    # THINKING VERSION (SIMULATED)
    # -----------------------------
    def generate_with_thinking(
        self,
        prompt: str,
        system_prompt: str = "",
        on_token=None,
        on_thinking_token=None,
        max_tokens: int = None
    ):
        effective_system = self.system_prompt.strip()
        if system_prompt:
            effective_system += "\n\n" + system_prompt.strip()

        messages = []
        if effective_system:
            messages.append({"role": "system", "content": effective_system})
        messages.append({"role": "user", "content": prompt})

        # NOTE:
        # Featherless API does NOT reliably expose "thinking" tokens like Ollama Qwen3.
        # So we simulate single-pass reasoning + response.

        data = self._call_api(messages, max_tokens=max_tokens)
        text = data["choices"][0]["message"]["content"]

        # Optional pseudo "thinking phase"
        if on_thinking_token:
            on_thinking_token("[internal reasoning completed]\n")

        if on_token:
            for ch in text:
                on_token(ch)
                time.sleep(0.001)
            on_token(None)

        return text


# -----------------------------
# SINGLETON INSTANCE
# -----------------------------
llm_client = OllamaClient()