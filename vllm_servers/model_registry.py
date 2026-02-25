#!/usr/bin/env python3

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

LOGGER = logging.getLogger(__name__)

CandidateFormatter = Callable[[str], str]
PromptInput = str | list[int]


def default_candidate_formatter(label: str) -> str:
    return label


class ChatTemplateKind(str, Enum):
    HF_MULTIMODAL_PROCESSOR = "hf_multimodal_processor"
    MISTRAL_TOKENIZER = "mistral_tokenizer"


@dataclass(frozen=True)
class ModelConfig:
    alias: str
    hf_id: str
    is_multimodal: bool
    vllm_engine_kwargs: dict[str, Any] = field(default_factory=dict)
    system_prompt: str | None = None
    system_prompt_hf_file: str | None = None
    chat_template: ChatTemplateKind = ChatTemplateKind.HF_MULTIMODAL_PROCESSOR
    candidate_format: CandidateFormatter = default_candidate_formatter
    image_placeholder_text: str = "<image>"


@dataclass
class ModelRuntime:
    config: ModelConfig
    processor: Any | None = None
    tokenizer: Any | None = None
    mistral_tokenizer: Any | None = None
    resolved_system_prompt: str | None = None


@lru_cache(maxsize=16)
def _load_hf_processor(model_id: str) -> Any:
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(model_id, trust_remote_code=False)


@lru_cache(maxsize=16)
def _load_hf_tokenizer(model_id: str) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)


@lru_cache(maxsize=16)
def _load_mistral_common_tokenizer(model_id: str) -> Any:
    from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

    try:
        return MistralTokenizer.from_hf_hub(repo_id=model_id, local_files_only=True)
    except Exception:
        return MistralTokenizer.from_hf_hub(repo_id=model_id, local_files_only=False)


@lru_cache(maxsize=16)
def _load_text_file_from_hf(model_id: str, filename: str) -> str:
    from huggingface_hub import hf_hub_download

    local_path = hf_hub_download(repo_id=model_id, filename=filename)
    return Path(local_path).read_text(encoding="utf-8")


def _resolve_system_prompt(config: ModelConfig) -> str | None:
    if config.system_prompt:
        value = config.system_prompt.strip()
        return value if value else None

    if config.system_prompt_hf_file:
        try:
            text = _load_text_file_from_hf(config.hf_id, config.system_prompt_hf_file).strip()
        except Exception as exc:
            LOGGER.warning(
                "Failed to load system prompt from %s/%s: %s",
                config.hf_id,
                config.system_prompt_hf_file,
                exc,
            )
            return None
        return text if text else None

    return None


def create_model_runtime(config: ModelConfig) -> ModelRuntime:
    runtime = ModelRuntime(config=config, resolved_system_prompt=_resolve_system_prompt(config))

    if config.chat_template == ChatTemplateKind.HF_MULTIMODAL_PROCESSOR:
        runtime.processor = _load_hf_processor(config.hf_id)
    elif config.chat_template == ChatTemplateKind.MISTRAL_TOKENIZER:
        try:
            runtime.tokenizer = _load_hf_tokenizer(config.hf_id)
        except Exception as exc:
            LOGGER.warning(
                "Failed to load HF tokenizer for %s (%s). Falling back to manual Mistral prompt.",
                config.hf_id,
                exc,
            )
            runtime.tokenizer = None

        try:
            runtime.mistral_tokenizer = _load_mistral_common_tokenizer(config.hf_id)
            LOGGER.info(
                "Loaded mistral_common tokenizer for %s.",
                config.hf_id,
            )
        except Exception as mistral_exc:
            LOGGER.warning(
                "Failed to load mistral_common tokenizer for %s (%s).",
                config.hf_id,
                mistral_exc,
            )
            runtime.mistral_tokenizer = None
    else:
        raise ValueError(f"Unsupported chat template kind: {config.chat_template}")

    return runtime


def _build_hf_multimodal_base_prompt(runtime: ModelRuntime, text: str) -> str:
    if runtime.processor is None:
        raise RuntimeError("HF multimodal processor is not initialized for this runtime.")

    messages: list[dict[str, Any]] = []
    if runtime.resolved_system_prompt:
        messages.append(
            {
                "role": "system",
                "content": [{"type": "text", "text": runtime.resolved_system_prompt}],
            }
        )
    messages.append(
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": text}],
        }
    )

    prompt = runtime.processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    return str(prompt)


def _build_mistral_base_prompt(runtime: ModelRuntime, text: str) -> str:
    # Mistral tokenizer templates are text-based, so keep image as explicit placeholder text.
    user_content = f"{runtime.config.image_placeholder_text}\n{text}".strip()

    if runtime.tokenizer is not None:
        messages: list[dict[str, Any]] = []
        if runtime.resolved_system_prompt:
            messages.append({"role": "system", "content": runtime.resolved_system_prompt})
        messages.append({"role": "user", "content": user_content})

        prompt = runtime.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        return str(prompt)

    if runtime.mistral_tokenizer is not None:
        from mistral_common.protocol.instruct.request import ChatCompletionRequest

        messages: list[dict[str, Any]] = []
        if runtime.resolved_system_prompt:
            messages.append({"role": "system", "content": runtime.resolved_system_prompt})
        messages.append({"role": "user", "content": user_content})

        request = ChatCompletionRequest.from_openai(messages=messages)
        tokenized = runtime.mistral_tokenizer.encode_chat_completion(request)
        if getattr(tokenized, "text", None):
            return str(tokenized.text)
        return str(runtime.mistral_tokenizer.decode(tokenized.tokens))

    # Fallback path for tokenizers that are unavailable in current transformers build.
    if runtime.resolved_system_prompt:
        return (
            "<s>SYSTEM:\n"
            f"{runtime.resolved_system_prompt}\n"
            f"USER: {user_content}\n"
            "ASSISTANT:"
        )
    return f"<s>USER: {user_content}\nASSISTANT:"


def _get_mistral_encoder(runtime: ModelRuntime) -> Any | None:
    if runtime.mistral_tokenizer is None:
        return None

    # mistral_common tokenizer (preferred path)
    if hasattr(runtime.mistral_tokenizer, "encode_chat_completion"):
        return runtime.mistral_tokenizer

    # vLLM tokenizer wrapper exposes `.mistral`
    mistral_obj = getattr(runtime.mistral_tokenizer, "mistral", None)
    if mistral_obj is not None and hasattr(mistral_obj, "encode_chat_completion"):
        return mistral_obj

    return None


def _build_mistral_chat_tokens_with_image(
    runtime: ModelRuntime,
    text: str,
    image: Any,
) -> list[int] | None:
    encoder = _get_mistral_encoder(runtime)
    if encoder is None:
        return None

    from mistral_common.protocol.instruct.chunk import ImageChunk, TextChunk
    from mistral_common.protocol.instruct.messages import SystemMessage, UserMessage
    from mistral_common.protocol.instruct.request import ChatCompletionRequest

    request = ChatCompletionRequest(
        messages=[
            *(
                [SystemMessage(content=runtime.resolved_system_prompt)]
                if runtime.resolved_system_prompt
                else []
            ),
            UserMessage(content=[TextChunk(text=text), ImageChunk(image=image)]),
        ],
    )
    tokenized = encoder.encode_chat_completion(request)
    return [int(tok) for tok in list(tokenized.tokens)]


def _tokenize_mistral_text_fragment(runtime: ModelRuntime, text: str) -> list[int]:
    if runtime.mistral_tokenizer is None:
        raise RuntimeError("mistral_common tokenizer is not initialized.")

    # mistral_common uses a Tekkenizer underneath (requires explicit BOS/EOS flags).
    instruct_tokenizer = getattr(runtime.mistral_tokenizer, "instruct_tokenizer", None)
    tekkenizer = getattr(instruct_tokenizer, "tokenizer", None) if instruct_tokenizer else None
    if tekkenizer is not None and hasattr(tekkenizer, "encode"):
        ids = tekkenizer.encode(text, bos=False, eos=False)
        return [int(tok) for tok in list(ids)]

    # Last resort: attempt common HF-style encoding if available.
    encode_fn = getattr(runtime.mistral_tokenizer, "encode", None)
    if callable(encode_fn):
        ids = encode_fn(text)
        return [int(tok) for tok in list(ids)]

    raise RuntimeError("Could not tokenize mistral text fragment via mistral_common.")


def build_scoring_prompts(
    runtime: ModelRuntime,
    text: str,
    candidates: list[str],
    image: Any,
) -> list[PromptInput]:
    if runtime.config.chat_template == ChatTemplateKind.HF_MULTIMODAL_PROCESSOR:
        base_prompt = _build_hf_multimodal_base_prompt(runtime, text=text)
        return [base_prompt, *[base_prompt + str(candidate) for candidate in candidates]]

    if runtime.config.chat_template == ChatTemplateKind.MISTRAL_TOKENIZER:
        base_prompt_tokens = _build_mistral_chat_tokens_with_image(runtime, text=text, image=image)
        if base_prompt_tokens is None:
            base_prompt = _build_mistral_base_prompt(runtime, text=text)
            return [base_prompt, *[base_prompt + str(candidate) for candidate in candidates]]

        empty_ids = _tokenize_mistral_text_fragment(runtime, "")
        prompts: list[PromptInput] = [base_prompt_tokens]
        for candidate in candidates:
            candidate_ids = _tokenize_mistral_text_fragment(runtime, str(candidate))
            if empty_ids and candidate_ids[: len(empty_ids)] == empty_ids:
                candidate_ids = candidate_ids[len(empty_ids) :]
            prompts.append(base_prompt_tokens + candidate_ids)
        return prompts

    raise ValueError(f"Unsupported chat template kind: {runtime.config.chat_template}")


MODEL_REGISTRY: dict[str, ModelConfig] = {
    "gemma3_4b_it": ModelConfig(
        alias="gemma3_4b_it",
        hf_id="google/gemma-3-4b-it",
        is_multimodal=True,
        vllm_engine_kwargs={"limit_mm_per_prompt": {"image": 1}},
        chat_template=ChatTemplateKind.HF_MULTIMODAL_PROCESSOR,
    ),
    "smolvlm2_2_2b": ModelConfig(
        alias="smolvlm2_2_2b",
        hf_id="HuggingFaceTB/SmolVLM2-2.2B-Instruct",
        is_multimodal=True,
        vllm_engine_kwargs={"limit_mm_per_prompt": {"image": 1}},
        chat_template=ChatTemplateKind.HF_MULTIMODAL_PROCESSOR,
    ),
    "smolvlm_256m_instruct": ModelConfig(
        alias="smolvlm_256m_instruct",
        hf_id="HuggingFaceTB/SmolVLM-256M-Instruct",
        is_multimodal=True,
        vllm_engine_kwargs={"limit_mm_per_prompt": {"image": 1}},
        chat_template=ChatTemplateKind.HF_MULTIMODAL_PROCESSOR,
    ),
    "ministral_3_3b": ModelConfig(
        alias="ministral_3_3b",
        hf_id="mistralai/Ministral-3-3B-Instruct-2512",
        is_multimodal=True,
        vllm_engine_kwargs={
            "tokenizer_mode": "mistral",
            "config_format": "mistral",
            "load_format": "mistral",
            "limit_mm_per_prompt": {"image": 1},
        },
        system_prompt_hf_file="SYSTEM_PROMPT.txt",
        chat_template=ChatTemplateKind.MISTRAL_TOKENIZER,
        image_placeholder_text="[IMG]",
    ),
    "qwen3_vl_4b": ModelConfig(
        alias="qwen3_vl_4b",
        hf_id="Qwen/Qwen3-VL-4B-Instruct",
        is_multimodal=True,
        vllm_engine_kwargs={"limit_mm_per_prompt": {"image": 1}},
        chat_template=ChatTemplateKind.HF_MULTIMODAL_PROCESSOR,
    ),
    "qwen3_vl_8b": ModelConfig(
        alias="qwen3_vl_8b",
        hf_id="Qwen/Qwen3-VL-8B-Instruct",
        is_multimodal=True,
        vllm_engine_kwargs={"limit_mm_per_prompt": {"image": 1}},
        chat_template=ChatTemplateKind.HF_MULTIMODAL_PROCESSOR,
    ),
    "qwen3_5_35b_a3b": ModelConfig(
        alias="qwen3_5_35b_a3b",
        hf_id="Qwen/Qwen3.5-35B-A3B",
        is_multimodal=False,
        vllm_engine_kwargs={},
        chat_template=ChatTemplateKind.MISTRAL_TOKENIZER,
        image_placeholder_text="<image>",
    ),
    "gpt_oss_20b": ModelConfig(
        alias="gpt_oss_20b",
        hf_id="openai/gpt-oss-20b",
        is_multimodal=False,
        vllm_engine_kwargs={},
        chat_template=ChatTemplateKind.MISTRAL_TOKENIZER,
        image_placeholder_text="<image>",
    ),
    "qwen3_vl_2b": ModelConfig(
        alias="qwen3_vl_2b",
        hf_id="Qwen/Qwen3-VL-2B-Instruct",
        is_multimodal=True,
        vllm_engine_kwargs={"limit_mm_per_prompt": {"image": 1}},
        chat_template=ChatTemplateKind.HF_MULTIMODAL_PROCESSOR,
    ),
}


def list_model_aliases() -> list[str]:
    return sorted(MODEL_REGISTRY.keys())


def get_model_config(alias: str) -> ModelConfig:
    config = MODEL_REGISTRY.get(alias)
    if config is None:
        raise KeyError(
            f"Unknown model alias: {alias}. Supported aliases: {', '.join(list_model_aliases())}"
        )
    return config
