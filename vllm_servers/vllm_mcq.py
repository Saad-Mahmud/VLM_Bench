#!/usr/bin/env python3

from __future__ import annotations

import argparse
import base64
import io
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel
from vllm import LLM, SamplingParams

try:
    from .model_registry import (
        ModelConfig,
        ModelRuntime,
        build_scoring_prompts,
        create_model_runtime,
        get_model_config,
        list_model_aliases,
    )
except ImportError:
    from model_registry import (  # type: ignore[no-redef]
        ModelConfig,
        ModelRuntime,
        build_scoring_prompts,
        create_model_runtime,
        get_model_config,
        list_model_aliases,
    )

LOGGER = logging.getLogger(__name__)

SEQ_OPT_MODE = "seq_opt"
PARALLEL_OPT_MODE = "parallel_opt"
PARALLELE_OPT_MODE = "parallele_opt"  # Common misspelling; treat as parallel.
OPT_MODES = (SEQ_OPT_MODE, PARALLEL_OPT_MODE, PARALLELE_OPT_MODE)

DEFAULT_DTYPE = "bfloat16"
DTYPE_ALIASES: dict[str, str] = {
    "bf16": "bfloat16",
    "fp16": "float16",
    "half": "float16",
}
DTYPES = ("auto", "float16", "bfloat16", "float32", "bf16", "fp16", "half")

DEFAULT_GPU_MEMORY_UTILIZATION = 0.80
# Keep this small so long prompts (esp. multimodal) prefill in chunks, reducing
# prompt_logprobs peak memory (log_softmax alloc scales with chunk_len * vocab).
DEFAULT_MAX_NUM_BATCHED_TOKENS = 512


class RankRequest(BaseModel):
    text: str
    options: list[str]
    image_path: str | None = None
    image_base64: str | None = None
    option_labels: list[str] | None = None
    opt_mode: str | None = None


class ScoreItem(BaseModel):
    index: int
    label: str
    option: str
    logprob: float


class RankResponse(BaseModel):
    model: str
    alias: str
    scores: list[ScoreItem]
    ranked: list[ScoreItem]
    predicted: ScoreItem


@dataclass
class ServerState:
    alias: str
    config: ModelConfig
    runtime: ModelRuntime
    llm: LLM
    opt_mode: str = SEQ_OPT_MODE


APP_STATE: ServerState | None = None
app = FastAPI(title="vLLM MCQ LogLikelihood Server")


def _ensure_xformers_import_safe_for_pixtral() -> None:
    try:
        import xformers.ops  # noqa: F401
    except ImportError:
        return
    except Exception as exc:
        raise RuntimeError(
            "Detected an incompatible xformers installation. "
            "This breaks Pixtral/Ministral model inspection in vLLM. "
            "Uninstall xformers in this env (`pip uninstall -y xformers`) "
            "or reinstall an xformers build compatible with your torch/CUDA."
        ) from exc


def _get_state() -> ServerState:
    if APP_STATE is None:
        raise HTTPException(status_code=503, detail="Server is not initialized.")
    return APP_STATE


def _decode_base64_image(data: str) -> Image.Image:
    payload = data.strip()
    if payload.startswith("data:") and "," in payload:
        payload = payload.split(",", 1)[1]
    raw = base64.b64decode(payload, validate=True)
    image = Image.open(io.BytesIO(raw))
    return image.convert("RGB")


def _load_image(image_path: str | None, image_base64: str | None) -> Image.Image:
    if bool(image_path) == bool(image_base64):
        raise HTTPException(
            status_code=400,
            detail="Provide exactly one of 'image_path' or 'image_base64'.",
        )

    if image_path:
        path = Path(image_path)
        if not path.is_file():
            raise HTTPException(status_code=400, detail=f"image_path not found: {image_path}")
        try:
            return Image.open(path).convert("RGB")
        except Exception as exc:
            raise HTTPException(
                status_code=400, detail=f"Failed to open image_path '{image_path}': {exc}"
            ) from exc

    try:
        return _decode_base64_image(image_base64 or "")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to decode image_base64: {exc}") from exc


def _index_to_label(index: int) -> str:
    # Excel-style labels: A..Z, AA..AZ, BA...
    value = index + 1
    chars: list[str] = []
    while value > 0:
        value, rem = divmod(value - 1, 26)
        chars.append(chr(ord("A") + rem))
    return "".join(reversed(chars))


def _derive_labels(num_options: int) -> list[str]:
    return [_index_to_label(i) for i in range(num_options)]


def _resolve_labels(option_labels: list[str] | None, num_options: int) -> list[str]:
    if option_labels is None:
        return _derive_labels(num_options)
    if len(option_labels) != num_options:
        raise HTTPException(
            status_code=400,
            detail=f"option_labels length ({len(option_labels)}) does not match options length ({num_options}).",
        )
    normalized = [str(label).strip() for label in option_labels]
    if any(not label for label in normalized):
        raise HTTPException(status_code=400, detail="option_labels must not contain empty values.")
    return normalized


def _lcp_length(a: list[int], b: list[int]) -> int:
    max_len = min(len(a), len(b))
    idx = 0
    while idx < max_len and a[idx] == b[idx]:
        idx += 1
    return idx


def _coerce_logprob_value(value: Any) -> float:
    if isinstance(value, (float, int)):
        return float(value)

    if hasattr(value, "logprob"):
        return float(value.logprob)

    if isinstance(value, dict):
        if "logprob" in value:
            return float(value["logprob"])
        if "log_prob" in value:
            return float(value["log_prob"])

    raise ValueError(f"Unsupported logprob value type: {type(value)}")


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _extract_token_logprob(entry: Any, token_id: int, position: int) -> float:
    if entry is None:
        raise ValueError(f"prompt_logprobs[{position}] is None.")
    target_int = _as_int(token_id)

    if isinstance(entry, dict):
        if token_id in entry:
            return _coerce_logprob_value(entry[token_id])

        token_id_str = str(token_id)
        if token_id_str in entry:
            return _coerce_logprob_value(entry[token_id_str])

        for key, value in entry.items():
            if target_int is not None and _as_int(key) == target_int:
                return _coerce_logprob_value(value)

            key_token_id = _as_int(getattr(key, "token_id", None))
            if target_int is not None and key_token_id == target_int:
                return _coerce_logprob_value(value)

            value_token_id = _as_int(getattr(value, "token_id", None))
            if target_int is not None and value_token_id == target_int:
                return _coerce_logprob_value(value)

        raise ValueError(
            f"Actual token id {token_id} missing in prompt_logprobs[{position}] candidates."
        )

    # Some backends return a single Logprob-like object.
    value_token_id = _as_int(getattr(entry, "token_id", None))
    if target_int is not None and value_token_id == target_int:
        return _coerce_logprob_value(entry)

    raise ValueError(f"Unsupported prompt_logprobs[{position}] type: {type(entry)}")


def _sum_candidate_logprob(
    base_ids: list[int],
    full_ids: list[int],
    prompt_logprobs: list[Any] | None,
) -> float:
    if prompt_logprobs is None:
        raise ValueError("prompt_logprobs is None; ensure SamplingParams(prompt_logprobs=1).")

    prefix_len = _lcp_length(base_ids, full_ids)
    total = 0.0

    for pos in range(prefix_len, len(full_ids)):
        # vLLM prompt_logprobs usually has None at first position.
        if pos == 0:
            continue
        if pos >= len(prompt_logprobs):
            raise ValueError(
                f"prompt_logprobs shorter than prompt token ids: pos={pos}, len={len(prompt_logprobs)}"
            )
        total += _extract_token_logprob(prompt_logprobs[pos], full_ids[pos], pos)

    return total


def _build_vllm_input(
    prompt: str | list[int],
    image: Image.Image,
    is_multimodal: bool,
) -> dict[str, Any]:
    prompt_key = "prompt_token_ids" if isinstance(prompt, list) else "prompt"
    payload: dict[str, Any] = {prompt_key: prompt}
    if is_multimodal:
        payload["multi_modal_data"] = {"image": image}
    return payload


def _normalize_opt_mode(value: str) -> str:
    normalized = value.strip().lower()
    if normalized == PARALLELE_OPT_MODE:
        return PARALLEL_OPT_MODE
    return normalized


def _normalize_dtype(value: str) -> str:
    normalized = value.strip().lower()
    return DTYPE_ALIASES.get(normalized, normalized)


def _get_prompt_token_ids(
    state: ServerState,
    prompt: str | list[int],
    image: Image.Image,
) -> list[int]:
    if isinstance(prompt, list):
        return list(prompt)

    # Tokenize via vLLM itself to ensure exact parity with how prompt_token_ids are produced.
    payload = _build_vllm_input(prompt, image=image, is_multimodal=state.config.is_multimodal)
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        prompt_logprobs=None,
        logprobs=None,
    )

    outputs = state.llm.generate([payload], sampling_params=sampling_params, use_tqdm=False)
    if not outputs:
        raise RuntimeError("vLLM returned no outputs when tokenizing base prompt.")
    return list(outputs[0].prompt_token_ids or [])


@app.get("/health")
def health() -> dict[str, str]:
    state = _get_state()
    return {"status": "ok", "model": state.config.hf_id, "alias": state.alias}


@app.post("/rank", response_model=RankResponse)
def rank(request: RankRequest) -> RankResponse:
    state = _get_state()

    if not request.text.strip():
        raise HTTPException(status_code=400, detail="'text' must be a non-empty string.")
    if not request.options:
        raise HTTPException(status_code=400, detail="'options' must contain at least one option.")

    labels = _resolve_labels(request.option_labels, num_options=len(request.options))
    image = _load_image(request.image_path, request.image_base64)

    opt_mode = _normalize_opt_mode(request.opt_mode or state.opt_mode or SEQ_OPT_MODE)
    if opt_mode not in (SEQ_OPT_MODE, PARALLEL_OPT_MODE):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported opt_mode '{request.opt_mode}'. Supported: {', '.join(OPT_MODES)}",
        )

    candidates = [state.config.candidate_format(label) for label in labels]

    try:
        prompt_inputs = build_scoring_prompts(
            state.runtime,
            text=request.text,
            candidates=[str(candidate) for candidate in candidates],
            image=image,
        )
    except Exception as exc:
        LOGGER.exception("Failed to build prompt for alias=%s", state.alias)
        raise HTTPException(status_code=500, detail=f"Failed to build chat prompt: {exc}") from exc

    if len(prompt_inputs) != len(request.options) + 1:
        raise HTTPException(
            status_code=500,
            detail=(
                "Prompt builder returned unexpected prompt count: "
                f"got {len(prompt_inputs)}, expected {len(request.options) + 1}"
            ),
        )

    base_prompt = prompt_inputs[0]
    option_prompts = prompt_inputs[1:]

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        prompt_logprobs=1,
        logprobs=None,
    )

    try:
        base_ids = _get_prompt_token_ids(state, base_prompt, image=image)
    except Exception as exc:
        LOGGER.exception("Failed to tokenize base prompt for alias=%s", state.alias)
        raise HTTPException(status_code=500, detail=f"Failed to tokenize base prompt: {exc}") from exc

    scores: list[ScoreItem] = []

    if opt_mode == PARALLEL_OPT_MODE:
        option_inputs = [
            _build_vllm_input(prompt, image=image, is_multimodal=state.config.is_multimodal)
            for prompt in option_prompts
        ]
        try:
            outputs = state.llm.generate(option_inputs, sampling_params=sampling_params, use_tqdm=False)
        except Exception as exc:
            LOGGER.exception(
                "vLLM generate failed for alias=%s options=%d opt_mode=%s",
                state.alias,
                len(request.options),
                opt_mode,
            )
            raise HTTPException(status_code=500, detail=f"vLLM generate failed: {exc}") from exc

        if len(outputs) != len(option_inputs):
            raise HTTPException(
                status_code=500,
                detail=(
                    "Unexpected number of outputs from vLLM: "
                    f"got {len(outputs)}, expected {len(option_inputs)}"
                ),
            )

        scored_outputs = list(outputs)
    else:
        scored_outputs = []
        for idx, prompt in enumerate(option_prompts):
            option_input = _build_vllm_input(prompt, image=image, is_multimodal=state.config.is_multimodal)
            try:
                outputs = state.llm.generate([option_input], sampling_params=sampling_params, use_tqdm=False)
            except Exception as exc:
                LOGGER.exception(
                    "vLLM generate failed for alias=%s option=%d/%d opt_mode=%s",
                    state.alias,
                    idx + 1,
                    len(option_prompts),
                    opt_mode,
                )
                raise HTTPException(status_code=500, detail=f"vLLM generate failed: {exc}") from exc
            if not outputs:
                raise HTTPException(
                    status_code=500,
                    detail=f"Unexpected number of outputs from vLLM: got 0, expected 1",
                )
            scored_outputs.append(outputs[0])

    for idx, (label, option_text, out) in enumerate(zip(labels, request.options, scored_outputs)):
        full_ids = list(out.prompt_token_ids or [])
        prompt_logprobs = list(out.prompt_logprobs) if out.prompt_logprobs is not None else None

        try:
            score = _sum_candidate_logprob(base_ids, full_ids, prompt_logprobs)
        except Exception as exc:
            LOGGER.exception(
                "Candidate logprob extraction failed for alias=%s option=%s index=%d",
                state.alias,
                label,
                idx,
            )
            raise HTTPException(
                status_code=500,
                detail=f"Failed to compute candidate logprob for option {idx} ({label}): {exc}",
            ) from exc

        scores.append(
            ScoreItem(
                index=idx,
                label=label,
                option=str(option_text),
                logprob=score,
            )
        )

    ranked = sorted(scores, key=lambda item: item.logprob, reverse=True)
    predicted = ranked[0]

    return RankResponse(
        model=state.config.hf_id,
        alias=state.alias,
        scores=scores,
        ranked=ranked,
        predicted=predicted,
    )


def initialize_server(
    model_alias: str,
    max_model_len: int | None = None,
    opt_mode: str = SEQ_OPT_MODE,
    dtype: str = DEFAULT_DTYPE,
    gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION,
    max_num_batched_tokens: int = DEFAULT_MAX_NUM_BATCHED_TOKENS,
) -> ServerState:
    config = get_model_config(model_alias)

    if model_alias == "ministral_3_3b":
        _ensure_xformers_import_safe_for_pixtral()

    runtime = create_model_runtime(config)

    engine_kwargs = dict(config.vllm_engine_kwargs)

    # vLLM V1: prompt_logprobs is incompatible with prefix caching.
    engine_kwargs["enable_prefix_caching"] = False
    engine_kwargs["dtype"] = _normalize_dtype(dtype)
    engine_kwargs.setdefault("gpu_memory_utilization", float(gpu_memory_utilization))
    engine_kwargs.setdefault("max_num_batched_tokens", int(max_num_batched_tokens))

    if config.is_multimodal:
        limit_mm = dict(engine_kwargs.get("limit_mm_per_prompt") or {})
        limit_mm.setdefault("image", 1)
        engine_kwargs["limit_mm_per_prompt"] = limit_mm

    if max_model_len is not None:
        engine_kwargs["max_model_len"] = max_model_len

    llm = LLM(model=config.hf_id, trust_remote_code=False, **engine_kwargs)
    return ServerState(
        alias=model_alias,
        config=config,
        runtime=runtime,
        llm=llm,
        opt_mode=_normalize_opt_mode(opt_mode),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve MCQ option ranking via vLLM prompt log-likelihood."
    )
    parser.add_argument(
        "--model-alias",
        required=True,
        choices=list_model_aliases(),
        help="Model alias from the registry.",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9009)
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Optional override for vLLM max_model_len.",
    )
    parser.add_argument(
        "--dtype",
        default=DEFAULT_DTYPE,
        choices=list(DTYPES),
        help="vLLM dtype (default: bfloat16).",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=DEFAULT_GPU_MEMORY_UTILIZATION,
        help=(
            "Fraction of total GPU memory to allocate for vLLM (KV cache, graphs, etc). "
            f"Default: {DEFAULT_GPU_MEMORY_UTILIZATION}. Lower this if prompt_logprobs OOMs."
        ),
    )
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=DEFAULT_MAX_NUM_BATCHED_TOKENS,
        help=(
            "Max tokens per iteration for chunked prefill. "
            f"Default: {DEFAULT_MAX_NUM_BATCHED_TOKENS}. Lower reduces peak memory."
        ),
    )
    parser.add_argument(
        "--opt-mode",
        default=SEQ_OPT_MODE,
        choices=list(OPT_MODES),
        help=(
            "How to score options. "
            f"{SEQ_OPT_MODE} scores options one-by-one (default, lower memory). "
            f"{PARALLEL_OPT_MODE}/{PARALLELE_OPT_MODE} scores all options in one batched call (faster, higher memory)."
        ),
    )
    parser.add_argument("--log-level", default="info")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    global APP_STATE
    APP_STATE = initialize_server(
        args.model_alias,
        max_model_len=args.max_model_len,
        opt_mode=args.opt_mode,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_batched_tokens=args.max_num_batched_tokens,
    )

    LOGGER.info(
        "Loaded model alias=%s hf_id=%s",
        APP_STATE.alias,
        APP_STATE.config.hf_id,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
