"""Ingest completed independent backtest audits into factor_memory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cogalpha.factor_memory import (  # noqa: E402
    build_factor_memory_summarizer,
    update_factor_memory_from_backtest_audit,
)
from cogalpha.llm import OpenAICompatibleClient  # noqa: E402
from cogalpha.llm.provider_config import (  # noqa: E402
    add_llm_provider_args,
    configure_llm_provider,
    load_key_file,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Incrementally ingest completed backtest_audit.json files."
    )
    parser.add_argument("--backtest-root", default="outputs/backtests")
    parser.add_argument("--memory-root", default="outputs/factor_memory")
    parser.add_argument("--allow-test-memory", action="store_true")
    parser.add_argument("--use-llm-summarizer", action="store_true")
    parser.add_argument("--inline-references", action="store_true")
    parser.add_argument("--max-patterns-per-kind", type=int, default=20)
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

    audit_paths = sorted(Path(args.backtest_root).glob("**/backtest_audit.json"))
    results = []
    skipped = []
    for audit_path in audit_paths:
        try:
            result = update_factor_memory_from_backtest_audit(
                audit_path=audit_path,
                memory_root=Path(args.memory_root),
                summarizer=summarizer,
                max_patterns_per_kind=args.max_patterns_per_kind,
                allow_test_memory=args.allow_test_memory,
            )
        except ValueError as exc:
            skipped.append({"audit_path": str(audit_path), "reason": str(exc)})
            continue
        if result.processed_audit_keys:
            results.append(
                {
                    "audit_path": str(result.audit_path),
                    "processed_audit_keys": result.processed_audit_keys,
                    "domain_updates": result.domain_updates,
                }
            )

    print(
        json.dumps(
            {
                "backtest_root": str(Path(args.backtest_root)),
                "memory_root": str(Path(args.memory_root)),
                "audits_seen": len(audit_paths),
                "audits_ingested": len(results),
                "results": results,
                "skipped": skipped,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
