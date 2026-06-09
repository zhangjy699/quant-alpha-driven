"""Run independent Alphalens analysis for one factor_pool factor."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from factor_backtest import load_factor_from_pool, run_factor_backtest  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze one factor from outputs/factor_pool by factor_id."
    )
    parser.add_argument("--factor-id", type=int, required=True)
    parser.add_argument("--data-dir", default="data/processed/company_all_a")
    parser.add_argument("--factor-pool", default="outputs/factor_pool")
    parser.add_argument("--output-dir", default="outputs/backtests")
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--quantiles", type=int, default=5)
    parser.add_argument("--neutralization-data")
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
        neutralization_data=args.neutralization_data,
    )

    print(
        json.dumps(
            {
                "factor_id": args.factor_id,
                "output_dir": str(result.output_dir),
                "report_path": str(result.report_path),
                "counts": result.counts,
                "memory_update": "disabled",
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
