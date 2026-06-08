"""Validate qualified formal MVP factors and update factor memory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cogalpha.factor_memory import (
    build_factor_memory_summarizer,
    update_factor_memory_from_validation_audit,
)
from cogalpha.llm import OpenAICompatibleClient
from cogalpha.llm.provider_config import (
    add_llm_provider_args,
    configure_llm_provider,
    load_key_file,
)
from cogalpha.validation_audit import validate_qualified_factors


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit qualified formal MVP factors on a validation split."
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Formal MVP run directory containing final_state.json.",
    )
    parser.add_argument("--data-dir", default="data/processed/company_all_a")
    parser.add_argument("--split", choices=["train", "valid", "test"], default="valid")
    parser.add_argument("--factor-pool", default="outputs/factor_pool")
    parser.add_argument("--memory-root", default="outputs/factor_memory")
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

    result = validate_qualified_factors(
        run_dir=args.run_dir,
        data_dir=args.data_dir,
        split=args.split,
        factor_pool_root=args.factor_pool,
    )
    memory_result = None
    if args.split != "test":
        memory_result = update_factor_memory_from_validation_audit(
            audit_path=result.report_path,
            memory_root=args.memory_root,
            summarizer=summarizer,
        )

    print(
        json.dumps(
            {
                "report_path": str(result.report_path),
                "cache_path": str(result.cache_path),
                "counts": result.counts,
                "memory_state_path": (
                    str(memory_result.state_path) if memory_result is not None else None
                ),
                "memory_domain_updates": (
                    memory_result.domain_updates if memory_result is not None else {}
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
