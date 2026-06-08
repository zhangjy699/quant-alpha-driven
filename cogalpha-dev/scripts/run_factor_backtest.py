"""Run an independent cross-sectional backtest for one factor_pool factor."""

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
from factor_backtest import load_factor_from_pool, run_factor_backtest  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backtest one factor from outputs/factor_pool by factor_id."
    )
    parser.add_argument("--factor-id", type=int, required=True)
    parser.add_argument("--data-dir", default="data/processed/company_all_a")
    parser.add_argument("--factor-pool", default="outputs/factor_pool")
    parser.add_argument("--output-dir", default="outputs/backtests")
    parser.add_argument("--memory-root", default="outputs/factor_memory")
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--quantiles", type=int, default=5)
    parser.add_argument("--cost-bps", type=float, default=10.0)
    parser.add_argument("--neutralization-data")
    parser.add_argument("--no-memory-update", action="store_true")
    parser.add_argument("--allow-test-memory", action="store_true")
    parser.add_argument("--use-llm-summarizer", action="store_true")
    parser.add_argument("--inline-references", action="store_true")
    parser.add_argument(
        "--max-patterns-per-kind",
        type=int,
        default=20,
        help="Maximum memory patterns retained per domain when writing feedback.",
    )
    add_llm_provider_args(parser, default_max_tokens=4096)
    args = parser.parse_args()

    factor_input = load_factor_from_pool(
        factor_id=args.factor_id,
        factor_pool_root=Path(args.factor_pool),
    )
    result = run_factor_backtest(
        factor_input=factor_input,
        data_dir=Path(args.data_dir),
        output_root=Path(args.output_dir),
        start_date=args.start_date,
        end_date=args.end_date,
        quantiles=args.quantiles,
        cost_bps=args.cost_bps,
        neutralization_data=args.neutralization_data,
    )

    memory_update = None
    if not args.no_memory_update:
        summarizer = None
        if args.use_llm_summarizer:
            load_key_file(args.key_file)
            configure_llm_provider(args)
            summarizer = build_factor_memory_summarizer(
                OpenAICompatibleClient.from_env(),
                inline_references=args.inline_references,
            )
        memory_result = update_factor_memory_from_backtest_audit(
            audit_path=result.audit_path,
            memory_root=Path(args.memory_root),
            summarizer=summarizer,
            max_patterns_per_kind=args.max_patterns_per_kind,
            allow_test_memory=args.allow_test_memory,
        )
        memory_update = {
            "processed_audit_keys": memory_result.processed_audit_keys,
            "domain_updates": memory_result.domain_updates,
            "state_path": str(memory_result.state_path),
        }

    print(
        json.dumps(
            {
                "factor_id": args.factor_id,
                "output_dir": str(result.output_dir),
                "report_path": str(result.report_path),
                "audit_path": str(result.audit_path),
                "counts": result.counts,
                "memory_update": memory_update,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
