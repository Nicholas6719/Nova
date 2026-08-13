"""
LLM Engine — MLX Llama streaming inference.

Uses mlx-lm for Apple Silicon-optimized inference.
All inference is local — no network calls.

Streaming API: on_token callback receives each token as it generates.
Nova's pipeline uses this for sentence-by-sentence TTS overlap.

PROMPT CACHE: most of every prompt is identical to the last one — Nova's
identity block alone is ~930 tokens and never changes. Reprocessing it each
turn was measured at 1.67s of a 1.77s wait before Nova started speaking. The
KV cache for the longest shared prefix is now carried between turns, so only
the genuinely new tokens are processed. See `_reuse_cache`.
"""

from __future__ import annotations

import logging
import re
import threading
from typing import Callable, Optional

log = logging.getLogger("nova.llm")

# Terminal punctuation at the very end of what has been generated so far.
# Deliberately anchored: a full stop mid-string (an abbreviation, a decimal)
# is not the end of the reply.
_ENDS_SENTENCE = re.compile(r"[.!?][\"')\]]?\s*$")
# ...but the marker of a numbered list is not a sentence, even though it ends
# in a full stop. Stopping there left Nova saying "...bachelor's degree. 2."
# out loud. Checked separately rather than folded into the pattern above,
# because a real sentence may legitimately end in a digit ("in 1896.").
_TRAILING_LIST_MARKER = re.compile(r"(?:^|\n)\s*\d+[.)]\s*$")


class LLMEngine:
    def __init__(self, config: dict) -> None:
        self.config = config
        log.info(f"Loading MLX model: {config['model']}")
        from mlx_lm import load
        self.model, self.tokenizer = load(config["model"])

        # ── Prompt cache state ────────────────────────────────────────────────
        # `_cache_ids` is the exact token sequence `_cache` holds, so the two can
        # never silently disagree about what the model has already seen. Any
        # failure drops both (see _drop_cache) — a stale cache would not raise,
        # it would quietly answer from the wrong context.
        self._cache_enabled = bool(config.get("prompt_cache", True))
        self._cache = None
        self._cache_ids: list[int] = []
        self._cache_thread: Optional[int] = None
        log.info(f"LLM ready. (prompt cache: {'on' if self._cache_enabled else 'off'})")

    # ── Prompt cache ──────────────────────────────────────────────────────────────
    def _encode(self, prompt: str) -> list[int]:
        """Tokenize exactly the way mlx-lm tokenizes a string prompt.

        mlx-lm skips the special tokens when the text already starts with BOS.
        A plain `encode()` here would add a SECOND BOS and feed the model a
        different prompt than it gets today — measured, so this mirrors it
        rather than assuming."""
        bos = getattr(self.tokenizer, "bos_token", None)
        add_special = bos is None or not prompt.startswith(bos)
        return list(self.tokenizer.encode(prompt, add_special_tokens=add_special))

    @staticmethod
    def _common_prefix(a: list[int], b: list[int]) -> int:
        n = 0
        for x, y in zip(a, b):
            if x != y:
                break
            n += 1
        return n

    def _drop_cache(self) -> None:
        self._cache = None
        self._cache_ids = []

    def _reuse_cache(self, ids: list[int]):
        """Return (cache, start) — a cache holding exactly ids[:start].

        Always recomputes the shared prefix against the real token sequence, so
        it stays correct no matter how the system prompt changes: a reworded
        prompt or a ticked-over clock simply shortens the reuse instead of
        feeding the model a context it never saw."""
        from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache, can_trim_prompt_cache

        # MLX arrays belong to the thread that made them.
        if self._cache is not None and self._cache_thread != threading.get_ident():
            log.debug("prompt cache built on another thread; rebuilding")
            self._drop_cache()

        n = 0
        if self._cache is not None:
            n = self._common_prefix(self._cache_ids, ids)
            # Never feed an empty prompt: leave at least one token to process.
            n = min(n, len(ids) - 1)
            if n > 0 and can_trim_prompt_cache(self._cache):
                extra = len(self._cache_ids) - n
                if extra > 0:
                    trim_prompt_cache(self._cache, extra)
                return self._cache, n
            self._drop_cache()

        self._cache = make_prompt_cache(self.model)
        self._cache_thread = threading.get_ident()
        return self._cache, 0

    def warm(self, system_prompt: str) -> int:
        """Pre-process the unchanging part of the prompt so the FIRST turn is
        fast too. Must run on the thread that owns MLX. Returns tokens cached."""
        if not self._cache_enabled:
            return 0
        import mlx.core as mx

        from mlx_lm.models.cache import make_prompt_cache

        prompt = self._format_prompt(
            self._build_messages(system_prompt, [], "hello")
        )
        ids = self._encode(prompt)
        try:
            cache = make_prompt_cache(self.model)
            # A direct model call leaves the cache holding EXACTLY these tokens.
            # generate_step would additionally feed its own sampled token in,
            # leaving a phantom token in the cache — measured.
            self.model(mx.array(ids)[None], cache=cache)
            mx.eval([c.state for c in cache])
        except Exception as exc:
            log.warning(f"Prompt cache warm-up failed ({exc}); continuing uncached.")
            self._drop_cache()
            return 0

        self._cache = cache
        self._cache_ids = ids
        self._cache_thread = threading.get_ident()
        log.info(f"Prompt cache warmed: {len(ids)} tokens")
        return len(ids)

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
        max_tokens = self.config.get("max_tokens", 512)

        # Spoken length budget. Nova reaches the first word in about a second
        # now, and then talks for 20-46s, which is the remaining complaint.
        # Measured, a SENTENCE cap does not fix it: "why is the sky blue" is two
        # sentences but 57 words and 19.9s. What matters is words, so generation
        # stops at the first sentence end past this budget — always on a
        # boundary, so Nova is never cut off mid-sentence. 0 disables.
        soft_words = self.config.get("soft_max_words", 0)

        # ONE generation loop, whether or not the cache is on. There used to be
        # a second copy for the uncached path, and the length budget was added
        # to only one of them — so turning the prompt cache off silently turned
        # the budget off too. Two unrelated switches must not be entangled.
        ids: list[int] = []
        cache = None
        start = 0
        if self._cache_enabled:
            ids = self._encode(prompt)
            cache, start = self._reuse_cache(ids)
            gen_prompt = ids[start:]
        else:
            gen_prompt = prompt

        full = ""
        generated: list[int] = []
        try:
            for response in stream_generate(
                self.model,
                self.tokenizer,
                prompt=gen_prompt,
                max_tokens=max_tokens,
                sampler=sampler,
                prompt_cache=cache,
            ):
                # mlx-lm yields GenerationResponse objects; the text is on `.text`.
                token = response.text
                full += token
                generated.append(response.token)
                on_token(token)

                if soft_words and len(full.split()) >= soft_words \
                        and _ENDS_SENTENCE.search(full) \
                        and not _TRAILING_LIST_MARKER.search(full):
                    log.debug(f"soft word budget reached ({len(full.split())} words)")
                    break
        except Exception:
            # The cache no longer matches _cache_ids; anything else would answer
            # the next turn from a context the model never actually saw.
            self._drop_cache()
            raise

        if self._cache_enabled:
            # The cache now holds the prompt plus what was just generated, which
            # is exactly the prefix of next turn's prompt — so a follow-up
            # reprocesses almost nothing.
            self._cache_ids = ids + generated
            self._bound_cache()
        return full.strip()

    def _bound_cache(self) -> None:
        """Keep the cache inside the context window. Nova caps history, so this
        is a backstop rather than the normal path."""
        limit = self.config.get("context_window", 4096)
        if len(self._cache_ids) > limit:
            log.debug(f"prompt cache exceeded {limit} tokens; resetting")
            self._drop_cache()

    # ── Non-streaming generation (internal / tool use) ────────────────────────────
    def generate(
        self,
        system_prompt: str,
        history: list[dict],
        user_message: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Blocking generation. Used internally for tool-assisted and structured
        (JSON) tasks. Pass temperature=0 for deterministic structured output."""
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        messages = self._build_messages(system_prompt, history, user_message)
        prompt   = self._format_prompt(messages)

        temp = temperature if temperature is not None else self.config.get("temperature", 0.7)
        sampler = make_sampler(temp=temp)
        return generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens if max_tokens is not None else self.config.get("max_tokens", 512),
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
