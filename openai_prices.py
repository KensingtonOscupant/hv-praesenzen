"""openai_prices.py

Static price sheet for **text‑generation** models in the OpenAI API.  All figures
are **US‑dollar cost per 1 million tokens** (standard processing tier) and are
current as of **2025‑07‑30**.

Included models (≥ GPT‑4o mini, released 2024‑07‑18):
    • `gpt‑4o`              — flagship multimodal model
    • `gpt‑4o‑mini`        — cost‑efficient sibling
    • `gpt‑4.1`            — 2025 upgrade to 4o
    • `gpt‑4.1‑mini`
    • `gpt‑4.1‑nano`
    • `o3`                 — reasoning model
    • `o4‑mini`            — faster, cheaper reasoning model

> **Why no “‑high” variants?**  The *‑high* suffixes you sometimes see in the
> ChatGPT UI (e.g. *o4‑mini‑high*) are throttled, higher‑compute versions of the
> same underlying API model.  They do **not** have a separate model name or
> price in the API, so you can use the same entry (e.g. `o4‑mini`).
>
> Vision/audio/TTS variants have distinct prices and are therefore left out of
> this **text‑token** table.

Example
-------
>>> from openai_prices import price
>>> price("gpt-4o-mini")
{'input': 0.15, 'output': 0.6}
>>> price("gpt-4.1", "input")
2.0
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
#  Main price table  (USD per 1 M tokens)
# ---------------------------------------------------------------------------
PRICES: dict[str, dict[str, float]] = {
    "gpt-4o": {"input": 2.50, "output": 10.00},           # ([platform.openai.com](https://platform.openai.com/docs/models/gpt-4o?utm_source=chatgpt.com))
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},      # ([platform.openai.com](https://platform.openai.com/docs/models/gpt-4o-mini?utm_source=chatgpt.com))
    "gpt-4.1": {"input": 2.00, "output": 8.00},          # ([platform.openai.com](https://platform.openai.com/docs/models/compare?model=gpt-4.1&utm_source=chatgpt.com))
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},      # ([platform.openai.com](https://platform.openai.com/docs/models/compare?model=gpt-4.1-mini&utm_source=chatgpt.com))
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},      # ([platform.openai.com](https://platform.openai.com/docs/models/gpt-4.1-nano?utm_source=chatgpt.com))
    "o3": {"input": 2.00, "output": 8.00},                # ([platform.openai.com](https://platform.openai.com/docs/models/o3?utm_source=chatgpt.com))
    "o4-mini": {"input": 1.10, "output": 4.40},           # ([platform.openai.com](https://platform.openai.com/docs/models/o4-mini?utm_source=chatgpt.com))
}

__all__ = ["price", "PRICES"]

# ---------------------------------------------------------------------------
#  Helper function
# ---------------------------------------------------------------------------

def price(model: str, kind: str | None = None) -> float | dict[str, float]:
    """Return per‑million‑token price(s) for *model*.

    Parameters
    ----------
    model: str
        Official OpenAI model name (e.g. ``"gpt-4o-mini"``).
    kind: {"input", "output"} | None, optional
        ``"input"`` or ``"output"`` to get a single numeric price,
        or *None* (default) to return both rates as a dict.

    Raises
    ------
    KeyError
        If *model* is not in :data:`PRICES`.
    ValueError
        If *kind* is provided but invalid.
    """
    info = PRICES[model]
    if kind is None:
        return info
    kind = kind.lower()
    if kind not in info:
        raise ValueError("kind must be 'input' or 'output'")
    return info[kind]


def refresh() -> None:
    """Placeholder for future automatic refresh logic.

    For now, edit :data:`PRICES` manually when OpenAI updates prices.
    """
    raise NotImplementedError(
        "Automatic price refresh not implemented; update PRICES manually."
    )
