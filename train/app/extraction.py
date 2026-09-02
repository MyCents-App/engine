"""Model loading and generation.

Lifted out of demo_server.py so the HTTP layer and any offline script share
one code path — the discipline demo_server already applied between its photo
and paste routes, extended to the service.

The model is loaded once per process and generation is serialized: one CUDA
context, one set of weights, and `generate` is not safe to call concurrently
on the same instance.
"""

from __future__ import annotations

import asyncio
import logging
import time

import postprocess
from app.config import settings
from prompts import build_messages

logger = logging.getLogger(__name__)

_model = None
_tokenizer = None
# Bounds how many callers may queue for the GPU. Without it a burst of
# uploads sits in an unbounded await and every client times out with no
# explanation; with it, the overflow gets an immediate, honest 503.
_gpu = asyncio.Semaphore(1)
_waiting = 0


class ModelNotLoaded(RuntimeError):
    pass


class Overloaded(RuntimeError):
    pass


def load_model(checkpoint: str | None = None) -> None:
    """Load the fine-tuned checkpoint. Called once, at startup."""
    global _model, _tokenizer
    if _model is not None:
        return

    # Imported here, not at module import: `unsloth` must be imported before
    # transformers, and it pulls in CUDA. Keeping it inside the function lets
    # the date/postprocess tests run on a machine with no GPU.
    from unsloth import FastLanguageModel  # noqa: I001
    import torch

    path = checkpoint or settings.checkpoint
    logger.info("loading %s (4bit=%s)", path, settings.load_in_4bit)
    t0 = time.time()
    _model, _tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(path).replace("\\", "/"),
        max_seq_length=settings.max_seq_length,
        dtype=None,
        load_in_4bit=settings.load_in_4bit,
    )
    FastLanguageModel.for_inference(_model)
    vram = (
        torch.cuda.max_memory_allocated() / 1024**3
        if torch.cuda.is_available() else 0.0
    )
    logger.info("model ready in %.1fs (%.2f GiB VRAM)", time.time() - t0, vram)


def is_loaded() -> bool:
    return _model is not None


def _text_tokenizer():
    """Unsloth returns a multimodal PROCESSOR for Qwen3.5 (it is a VL model).
    Calling it positionally routes a string into image preprocessing and dies
    with a confusing base64 error. Tokenize through the inner text tokenizer;
    use the processor only for chat templates."""
    return getattr(_tokenizer, "tokenizer", _tokenizer)


def build_prompt(ocr_text: str) -> str:
    messages = build_messages(ocr_text)   # identical to training and eval
    try:
        return _tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        return _tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


def count_prompt_tokens(ocr_text: str) -> int:
    """How many tokens the prompt will occupy.

    Used to reject an over-long request BEFORE generating. Silent truncation
    is the failure mode that matters here: the tail of a long receipt is
    simply dropped, the model returns confident JSON missing its last items,
    and nothing anywhere reports a problem. A multi-page receipt is exactly
    the case that hits this.
    """
    if _tokenizer is None:
        raise ModelNotLoaded("model is not loaded")
    inner = _text_tokenizer()
    return len(inner(text=build_prompt(ocr_text))["input_ids"])


def token_budget() -> int:
    """Prompt tokens available once room is reserved for the answer."""
    return settings.max_seq_length - settings.max_new_tokens


def _generate(ocr_text: str) -> tuple[dict | None, str, float]:
    import torch

    inner = _text_tokenizer()
    inputs = inner(text=build_prompt(ocr_text), return_tensors="pt").to(_model.device)
    input_len = inputs["input_ids"].shape[1]

    t0 = time.time()
    with torch.no_grad():
        out = _model.generate(
            **inputs,
            max_new_tokens=settings.max_new_tokens,
            do_sample=False,          # greedy — matches every published eval number
            pad_token_id=inner.pad_token_id or inner.eos_token_id,
        )
    elapsed = time.time() - t0

    raw = inner.decode(out[0][input_len:], skip_special_tokens=True)
    pred, _err = postprocess.extract_json(raw)
    if pred is not None:
        # task.md §4 code-layer job: reformat prices to NN.DD. Never changes
        # a value, only its representation.
        pred = postprocess.normalize_prediction(pred)
    return pred, raw, elapsed


async def extract(ocr_text: str) -> tuple[dict | None, str, float]:
    """Run the model, one caller at a time."""
    global _waiting
    if _model is None:
        raise ModelNotLoaded("model is not loaded")
    if _waiting >= settings.max_queue:
        raise Overloaded(
            f"{_waiting} requests already waiting for the GPU; try again shortly"
        )
    _waiting += 1
    try:
        async with _gpu:
            # Generation is blocking and CPU/GPU bound — off the event loop,
            # or health checks stall behind every receipt.
            return await asyncio.wait_for(
                asyncio.to_thread(_generate, ocr_text),
                timeout=settings.queue_timeout_s,
            )
    finally:
        _waiting -= 1


def queue_depth() -> int:
    return _waiting
