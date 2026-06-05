"""Shared CLI helpers for configuring CogAlpha LLM providers."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-pro"
DEFAULT_DEEPSEEK_REASONING_EFFORT = "high"
DEFAULT_DEEPSEEK_THINKING = "enabled"
DEFAULT_OPENAI_CHAT_MODEL = "gpt-4o-mini-2024-07-18"


def add_llm_provider_args(
    parser: argparse.ArgumentParser,
    *,
    default_max_tokens: int,
) -> None:
    """Add the standard OpenAI-compatible provider CLI flags."""

    parser.add_argument("--key-file", default="KEY.md")
    parser.add_argument("--provider", choices=["deepseek", "openai", "custom"], default="deepseek")
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--reasoning-effort", default=None)
    parser.add_argument("--thinking", choices=["enabled", "disabled"], default=None)
    parser.add_argument("--max-tokens", type=int, default=default_max_tokens)


def load_key_file(path: str) -> None:
    """Load common LLM environment variables from a simple key file."""

    key_path = Path(path)
    if not key_path.exists():
        return
    for raw_line in key_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.match(r"(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*[:=]\s*(.+)", line)
        if not match:
            continue
        name, value = match.groups()
        canonical_name = canonical_llm_env_name(name)
        clean_value = value.strip().strip('"').strip("'")
        if canonical_name:
            os.environ.setdefault(canonical_name, clean_value)
        elif name.isupper():
            os.environ.setdefault(name, clean_value)


def configure_llm_provider(args: argparse.Namespace) -> None:
    """Apply provider defaults and explicit overrides to environment variables."""

    if args.provider == "deepseek":
        os.environ.setdefault("COGALPHA_LLM_BASE_URL", DEFAULT_DEEPSEEK_BASE_URL)
        os.environ.setdefault("COGALPHA_LLM_MODEL", DEFAULT_DEEPSEEK_MODEL)
        os.environ.setdefault(
            "COGALPHA_LLM_REASONING_EFFORT",
            DEFAULT_DEEPSEEK_REASONING_EFFORT,
        )
        os.environ.setdefault("COGALPHA_LLM_THINKING", DEFAULT_DEEPSEEK_THINKING)
    elif args.provider == "openai":
        os.environ.setdefault("COGALPHA_LLM_MODEL", DEFAULT_OPENAI_CHAT_MODEL)

    if args.model:
        os.environ["COGALPHA_LLM_MODEL"] = args.model
    if args.base_url:
        os.environ["COGALPHA_LLM_BASE_URL"] = args.base_url
    if args.reasoning_effort:
        os.environ["COGALPHA_LLM_REASONING_EFFORT"] = args.reasoning_effort
    if args.thinking:
        os.environ["COGALPHA_LLM_THINKING"] = args.thinking
    if args.max_tokens is not None:
        os.environ["COGALPHA_LLM_MAX_TOKENS"] = str(args.max_tokens)


def canonical_llm_env_name(name: str) -> str | None:
    """Map common key-file aliases to CogAlpha LLM environment variables."""

    normalized = name.lower().replace("-", "_")
    if normalized in {
        "key",
        "api_key",
        "llm_api_key",
        "deepseek_api_key",
        "openai_api_key",
    }:
        return "COGALPHA_LLM_API_KEY"
    if normalized in {"model", "llm_model", "chat_model", "deepseek_model", "openai_model"}:
        return "COGALPHA_LLM_MODEL"
    if normalized in {
        "base_url",
        "api_base",
        "llm_base_url",
        "deepseek_base_url",
        "openai_base_url",
    }:
        return "COGALPHA_LLM_BASE_URL"
    if normalized in {"reasoning_effort", "deepseek_reasoning_effort"}:
        return "COGALPHA_LLM_REASONING_EFFORT"
    if normalized in {"thinking", "deepseek_thinking"}:
        return "COGALPHA_LLM_THINKING"
    if normalized in {"max_tokens", "llm_max_tokens", "deepseek_max_tokens"}:
        return "COGALPHA_LLM_MAX_TOKENS"
    return None
