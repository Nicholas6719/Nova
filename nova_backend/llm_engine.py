"""
LLM Engine — MLX Llama streaming inference.

Uses mlx-lm for Apple Silicon-optimized inference.
All inference is local — no network calls.

Streaming API: on_token callback receives each token as it generates.
Nova's pipeline uses this for sentence-by-sentence TTS overlap.
"""

from __future__ import annotations

import logging
from typing import Callable

log = logging.getLogger("nova.llm")


class LLMEngine:
    def __init__(self, config: dict) -> None:
        self.config = config
        log.info(f"Loading MLX model: {config['model']}")
        from mlx_lm import load
        self.model, self.tokenizer = load(config["model"])
        log.info("LLM ready.")

    # ── Streaming generation ──────────────────────────────────────────────────────
    def stream(
        self,
        system_prompt: str,
        history: list[dict],
        user_message: str,
        on_token: Callable[[str], None],
    ) -> str:
        """
        Stream response tokens via on_token callback.

        history: list of {"role": "user"|"assistant", "content": "..."}
        Returns the full response string after streaming completes.
        """
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        messages = self._build_messages(system_prompt, history, user_message)
        prompt   = self._format_prompt(messages)

        sampler = make_sampler(temp=self.config.get("temperature", 0.7))
        full = ""
        for response in stream_generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=self.config.get("max_tokens", 512),
            sampler=sampler,
        ):
            # mlx-lm yields GenerationResponse objects; the text is on `.text`.
            token = response.text
            full += token
            on_token(token)

        return full.strip()

    # ── Non-streaming generation (internal / tool use) ────────────────────────────
    def generate(
        self,
        system_prompt: str,
        history: list[dict],
        user_message: str,
    ) -> str:
        """Blocking generation. Used internally for tool-assisted queries."""
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        messages = self._build_messages(system_prompt, history, user_message)
        prompt   = self._format_prompt(messages)

        sampler = make_sampler(temp=self.config.get("temperature", 0.7))
        return generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=self.config.get("max_tokens", 512),
            sampler=sampler,
        ).strip()

    # ── Helpers ───────────────────────────────────────────────────────────────────
    def _build_messages(
        self,
        system_prompt: str,
        history: list[dict],
        user_message: str,
    ) -> list[dict]:
        messages = [{"role": "system", "content": system_prompt}]
        # Cap history to avoid overflowing context window
        messages.extend(history[-20:])
        messages.append({"role": "user", "content": user_message})
        return messages

    def _format_prompt(self, messages: list[dict]) -> str:
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
