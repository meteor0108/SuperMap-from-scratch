"""Pluggable VLM backend (REQ-057, p.7 Sec. V-F; D-003).

Interface: VLMBackend.complete(prompt: str) -> str.
Default model per the paper: Gemini 2.0 Flash ("gemini-2.0-flash", configs/default.yaml vlm.model).
The Gemini SDK and API key are only needed when GeminiBackend is constructed, not at import time.
Key lookup: environment variables first, then a simple KEY=VALUE `.env` file when a path is given
(the CLI passes the project-root `.env`, D-098). Key values are never logged or put in messages.
"""
from __future__ import annotations

import abc
import os
from pathlib import Path
from typing import Sequence

from supermap.config import load_config

DEFAULT_KEY_ENV = ("GEMINI_API_KEY", "GOOGLE_API_KEY")
PROJECT_DOTENV = Path(__file__).resolve().parents[3] / ".env"


# REQ-057: prompt text -> response text
class VLMBackend(abc.ABC):
    """Abstract VLM: one call per grounding query."""

    # REQ-057: send one prompt, return the raw response text
    @abc.abstractmethod
    def complete(self, prompt: str) -> str:
        ...


# REQ-057: simple KEY=VALUE parser for a .env file (no shell expansion)
def read_dotenv(path: str | Path) -> dict[str, str]:
    p = Path(path)
    if not p.is_file():
        return {}
    out: dict[str, str] = {}
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        if k.startswith("export "):
            k = k[len("export "):].strip()
        v = v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
            v = v[1:-1]
        out[k] = v
    return out


# REQ-057: API key lookup, environment variables first, then the optional .env file (D-098)
def find_api_key(env_names: Sequence[str] = DEFAULT_KEY_ENV,
                 dotenv_path: str | Path | None = None) -> str | None:
    for name in env_names:
        v = os.environ.get(name)
        if v:
            return v
    if dotenv_path is not None:
        vals = read_dotenv(dotenv_path)
        for name in env_names:
            if vals.get(name):
                return vals[name]
    return None


# REQ-057: Gemini 2.0 Flash through google-genai (or google-generativeai if that is installed)
class GeminiBackend(VLMBackend):
    """Raises RuntimeError at construction if no SDK or no API key is available."""

    # REQ-057: resolve SDK and key lazily
    def __init__(self, model: str = "gemini-2.0-flash", temperature: float = 0.0,
                 api_key: str | None = None, api_key_env: Sequence[str] = DEFAULT_KEY_ENV,
                 fallback_models: Sequence[str] = ()) -> None:
        self.model = model
        self.temperature = float(temperature)
        # REQ-057: flash-family models tried in order when `model` is not served by the API (D-099)
        self.fallback_models = [m for m in fallback_models if m != model]
        key = api_key or find_api_key(api_key_env)
        if not key:
            raise RuntimeError(f"no Gemini API key in environment ({', '.join(api_key_env)})")
        try:
            from google import genai  # type: ignore
            self._sdk = "genai"
            self._client = genai.Client(api_key=key)
            self._types = genai.types
        except ImportError:
            try:
                import google.generativeai as legacy  # type: ignore
            except ImportError as e:
                raise RuntimeError("neither google-genai nor google-generativeai is installed") from e
            legacy.configure(api_key=key)
            self._sdk = "legacy"
            self._client = legacy.GenerativeModel(model)

    # REQ-057: one generate call with the current model
    def _generate(self, prompt: str) -> str:
        if self._sdk == "genai":
            cfg = self._types.GenerateContentConfig(temperature=self.temperature)
            resp = self._client.models.generate_content(model=self.model, contents=prompt, config=cfg)
        else:
            resp = self._client.generate_content(prompt, generation_config={"temperature": self.temperature})
        return resp.text or ""

    # REQ-057: one generate call, returns response text; a model-not-found reply moves to the
    # next fallback model (D-099); other API failures become RuntimeError with the reason only
    def complete(self, prompt: str) -> str:
        while True:
            try:
                return self._generate(prompt)
            except Exception as e:
                if isinstance(e, RuntimeError):
                    raise
                if _is_model_not_found(e) and self.fallback_models:
                    self.model = self.fallback_models.pop(0)
                    if self._sdk == "legacy":
                        import google.generativeai as legacy  # type: ignore
                        self._client = legacy.GenerativeModel(self.model)
                    continue
                code, status = getattr(e, "code", None), getattr(e, "status", None)
                msg = str(getattr(e, "message", "") or "")[:160]
                raise RuntimeError(f"Gemini call failed for model {self.model!r} "
                                   f"({type(e).__name__}, code={code}, status={status}: {msg})") from None


# REQ-057: API reply meaning the requested model is not served (HTTP 404 / NOT_FOUND)
def _is_model_not_found(e: Exception) -> bool:
    code = getattr(e, "code", None)
    status = str(getattr(e, "status", "") or "")
    return code == 404 or status == "NOT_FOUND" or "NOT_FOUND" in type(e).__name__.upper()


# REQ-057: backend from the `vlm` config section
def make_backend(sec: dict | None = None, api_key: str | None = None) -> VLMBackend:
    sec = sec if sec is not None else load_config()["vlm"]
    name = sec.get("backend", "gemini")
    if name != "gemini":
        raise ValueError(f"unknown VLM backend {name!r}")
    return GeminiBackend(model=sec["model"], temperature=sec.get("temperature", 0.0), api_key=api_key,
                         api_key_env=tuple(sec.get("api_key_env", DEFAULT_KEY_ENV)),
                         fallback_models=tuple(sec.get("fallback_models", ())))
