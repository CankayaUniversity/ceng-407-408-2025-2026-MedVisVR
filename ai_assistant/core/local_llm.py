import json
import os
import urllib.error
import urllib.request
from typing import Any


def _provider() -> str:
    return os.getenv("AI_LLM_PROVIDER", "embedded_llama").strip().lower()


def _llamacpp_base_url() -> str:
    return os.getenv("AI_LLAMACPP_URL", "http://127.0.0.1:8080").rstrip("/")


def _timeout_s() -> int:
    return int(os.getenv("AI_LLM_TIMEOUT_S", "60") or 60)


def _post_json(url: str, payload: dict[str, Any], timeout_s: int) -> dict[str, Any]:
    req = urllib.request.Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        body = resp.read().decode("utf-8")
        return json.loads(body) if body else {}


def _get_json(url: str, timeout_s: int) -> dict[str, Any]:
    req = urllib.request.Request(url=url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        body = resp.read().decode("utf-8")
        return json.loads(body) if body else {}


def available_models(timeout_s: int | None = None) -> list[str]:
    _ = timeout_s
    return preferred_models(use_high_model=False)


def health_check(timeout_s: int | None = None) -> bool:
    timeout = timeout_s or _timeout_s()
    for path in ("/health", "/v1/models"):
        try:
            _get_json(f"{_llamacpp_base_url()}{path}", timeout)
            return True
        except Exception:
            continue
    return False


def preferred_models(use_high_model: bool) -> list[str]:
    default_model = os.getenv("AI_MODEL_DEFAULT", "deepseek-r1:14b")
    high_model = os.getenv("AI_MODEL_HIGH", "deepseek-r1:14b")
    low_fallback = os.getenv("AI_MODEL_FALLBACK", "deepseek-r1:14b")
    primary = high_model if use_high_model else default_model
    ordered = [primary, default_model, low_fallback]
    out: list[str] = []
    for m in ordered:
        if m and m not in out:
            out.append(m)
    return out


def generate_text(
    system_prompt: str,
    user_prompt: str,
    use_high_model: bool = False,
) -> tuple[str, str | None]:
    provider = _provider()
    if provider not in {"embedded_llama", "llama_cpp", "llama.cpp"}:
        return "", None
    return _generate_text_llamacpp(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        use_high_model=use_high_model,
    )


def _generate_text_llamacpp(
    system_prompt: str,
    user_prompt: str,
    use_high_model: bool = False,
) -> tuple[str, str | None]:
    timeout = _timeout_s()
    retries = int(os.getenv("AI_LLM_RETRIES", "1") or 1)
    models = preferred_models(use_high_model)
    temperature = float(os.getenv("AI_TEMPERATURE", "0.2") or 0.2)
    n_ctx = int(os.getenv("AI_N_CTX", "4096") or 4096)
    base = _llamacpp_base_url()

    for model in models:
        for _ in range(max(1, retries)):
            try:
                data = _post_json(
                    url=f"{base}/v1/chat/completions",
                    payload={
                        "model": model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        "temperature": temperature,
                        "stream": False,
                    },
                    timeout_s=timeout,
                )
                choices = data.get("choices", []) if isinstance(data, dict) else []
                if choices and isinstance(choices[0], dict):
                    message = choices[0].get("message", {})
                    text = message.get("content", "") if isinstance(message, dict) else ""
                    if isinstance(text, str) and text.strip():
                        return text.strip(), model
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError):
                pass
            except Exception:
                pass

            try:
                prompt = f"[SYSTEM]\n{system_prompt}\n\n[USER]\n{user_prompt}\n\n[ASSISTANT]\n"
                data = _post_json(
                    url=f"{base}/completion",
                    payload={
                        "prompt": prompt,
                        "temperature": temperature,
                        "n_predict": int(os.getenv("AI_N_PREDICT", "512") or 512),
                        "n_ctx": n_ctx,
                        "stop": ["[USER]", "[SYSTEM]"],
                        "stream": False,
                    },
                    timeout_s=timeout,
                )
                text = data.get("content", "") if isinstance(data, dict) else ""
                if isinstance(text, str) and text.strip():
                    return text.strip(), model
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError):
                continue
            except Exception:
                continue

    return "", None
