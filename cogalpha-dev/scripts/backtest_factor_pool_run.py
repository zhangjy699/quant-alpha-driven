"""Backtest factor_pool entries exported by one formal MVP run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from factor_backtest import load_factor_from_pool, run_factor_backtest  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backtest factor_pool entries belonging to one formal run."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--data-dir", default="data/processed/company_all_a")
    parser.add_argument("--factor-pool", default="outputs/factor_pool")
    parser.add_argument("--output-dir", default="outputs/backtests")
    parser.add_argument("--pools", nargs="+", default=["elite", "qualified"])
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--quantiles", type=int, default=5)
    parser.add_argument("--cost-bps", type=float, default=10.0)
    parser.add_argument("--neutralization-data")
    args = parser.parse_args()

    factor_ids = _factor_ids_for_run(
        factor_pool_root=Path(args.factor_pool),
        run_id=args.run_id,
        pools=set(args.pools),
    )
    results = []
    for factor_id in factor_ids:
        factor_input = load_factor_from_pool(
            factor_id=factor_id,
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
        results.append(
            {
                "factor_id": factor_id,
                "output_dir": str(result.output_dir),
                "report_path": str(result.report_path),
                "audit_path": str(result.audit_path),
            }
        )

    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "pools": args.pools,
                "factor_ids": factor_ids,
                "count": len(factor_ids),
                "results": results,
            },
            indent=2,
            sort_keys=True,
        )
    )


def _factor_ids_for_run(
    *,
    factor_pool_root: Path,
    run_id: str,
    pools: set[str],
) -> list[int]:
    index_path = factor_pool_root / "index.json"
    if not index_path.exists():
        return []
    index = json.loads(index_path.read_text(encoding="utf-8"))
    factor_ids = [
        int(entry["factor_id"])
        for entry in index.get("factors", [])
        if str(entry.get("run_id")) == run_id and str(entry.get("pool")) in pools
    ]
    return sorted(set(factor_ids))


if __name__ == "__main__":
    main()
