"""Incrementally update compressed factor memory from the shared factor pool."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cogalpha.factor_memory import build_factor_memory_summarizer, update_factor_memory
from cogalpha.llm import OpenAICompatibleClient
from cogalpha.llm.provider_config import (
    add_llm_provider_args,
    configure_llm_provider,
    load_key_file,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Update factor_memory artifacts from outputs/factor_pool."
    )
    parser.add_argument(
        "--factor-pool",
        default="outputs/factor_pool",
        help="Shared factor_pool root containing index.json.",
    )
    parser.add_argument(
        "--memory-root",
        default="outputs/factor_memory",
        help="Output root for compressed factor memory.",
    )
    parser.add_argument(
        "--max-patterns-per-kind",
        type=int,
        default=20,
        help="Maximum success/failure/avoid patterns retained per domain.",
    )
    parser.add_argument("--use-llm-summarizer", action="store_true")
    parser.add_argument("--inline-references", action="store_true")
    add_llm_provider_args(parser, default_max_tokens=4096)
    args = parser.parse_args()
    summarizer = None
    if args.use_llm_summarizer:
        load_key_file(args.key_file)
        configure_llm_provider(args)
        summarizer = build_factor_memory_summarizer(
            OpenAICompatibleClient.from_env(),
            inline_references=args.inline_references,
        )

    result = update_factor_memory(
        factor_pool_root=Path(args.factor_pool),
        memory_root=Path(args.memory_root),
        summarizer=summarizer,
        max_patterns_per_kind=args.max_patterns_per_kind,
    )
    print(
        json.dumps(
            {
                "factor_pool_root": str(result.factor_pool_root),
                "memory_root": str(result.memory_root),
                "processed_factor_ids": result.processed_factor_ids,
                "domain_updates": result.domain_updates,
                "state_path": str(result.state_path),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
