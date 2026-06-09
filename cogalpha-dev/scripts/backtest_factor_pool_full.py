"""Run independent full-window Alphalens analyses for factor_pool candidates."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from factor_backtest import load_factor_from_pool, run_factor_backtest  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze factor_pool candidates independently from mining. "
            "No factor_memory update is performed."
        )
    )
    parser.add_argument("--data-dir", default="data/processed/company_all_a")
    parser.add_argument("--factor-pool", default="outputs/factor_pool")
    parser.add_argument("--output-dir", default="outputs/backtests/full-factor-pool")
    parser.add_argument("--pools", nargs="+", default=["elite", "qualified"])
    parser.add_argument(
        "--factor-id",
        type=int,
        help="Run a smoke analysis for one global factor_id. Default runs all selected factors.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run selected factors even when matching completed reports already exist.",
    )
    parser.add_argument("--start-date", default="2018-01-01")
    parser.add_argument("--end-date", default="2026-12-31")
    parser.add_argument("--quantiles", type=int, default=10)
    parser.add_argument("--neutralization-data")
    args = parser.parse_args()

    output_root = Path(args.output_dir)
    run_id = datetime.now(UTC).strftime("full-factor-pool-%Y%m%d-%H%M%S")
    batch_dir = output_root / run_id
    factor_output_root = batch_dir / "factors"
    batch_dir.mkdir(parents=True, exist_ok=False)

    selected = _selected_factor_entries(
        factor_pool_root=Path(args.factor_pool),
        pools=set(args.pools),
        factor_id=args.factor_id,
    )
    completed = _completed_factor_reports(
        output_root,
        start_date=args.start_date,
        end_date=args.end_date,
        quantiles=args.quantiles,
    )
    results: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for entry in selected:
        factor_id = int(entry["factor_id"])
        if factor_id in completed and not args.force:
            skipped.append(
                {
                    "factor_id": factor_id,
                    "candidate_id": str(entry["candidate_id"]),
                    "factor_name": str(entry["factor_name"]),
                    "pool": str(entry["pool"]),
                    "domain_agent": str(entry["domain_agent"]),
                    "existing_report_path": str(completed[factor_id]),
                    "reason": "already_completed",
                }
            )
            continue
        factor_input = load_factor_from_pool(
            factor_id=factor_id,
            factor_pool_root=Path(args.factor_pool),
        )
        result = run_factor_backtest(
            factor_input=factor_input,
            data_dir=Path(args.data_dir),
            output_root=factor_output_root,
            start_date=args.start_date,
            end_date=args.end_date,
            quantiles=args.quantiles,
            neutralization_data=args.neutralization_data,
        )
        report = json.loads(result.report_path.read_text(encoding="utf-8"))
        results.append(
            {
                "factor_id": factor_id,
                "candidate_id": str(entry["candidate_id"]),
                "factor_name": str(entry["factor_name"]),
                "pool": str(entry["pool"]),
                "domain_agent": str(entry["domain_agent"]),
                "run_id": str(entry["run_id"]),
                "output_dir": str(result.output_dir.relative_to(batch_dir)),
                "report_path": str(result.report_path.relative_to(batch_dir)),
                "tear_sheets": [
                    str((result.output_dir / path).relative_to(batch_dir))
                    for path in report["artifacts"].get("tear_sheets", [])
                ],
                "factor_direction": int(report["factor_direction"]),
                "summary": report["summary"],
                "start_date": report["start_date"],
                "end_date": report["end_date"],
            }
        )

    summary = {
        "batch_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "data_dir": args.data_dir,
        "factor_pool": args.factor_pool,
        "pools": args.pools,
        "start_date_requested": args.start_date,
        "end_date_requested": args.end_date,
        "quantiles": args.quantiles,
        "memory_update": "disabled",
        "factor_id_filter": args.factor_id,
        "selection": {
            "unique_candidates": len(selected),
            "factor_ids": [int(entry["factor_id"]) for entry in selected],
            "completed_factor_ids": sorted(completed),
            "skipped_factor_ids": [item["factor_id"] for item in skipped],
            "force": bool(args.force),
        },
        "results": results,
        "skipped": skipped,
    }
    _write_json(batch_dir / "batch_report.json", summary)
    _write_summary_csv(batch_dir / "factor_summary.csv", results)
    _write_markdown_report(batch_dir / "README.md", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


def _selected_factor_entries(
    *,
    factor_pool_root: Path,
    pools: set[str],
    factor_id: int | None,
) -> list[dict[str, Any]]:
    index = json.loads((factor_pool_root / "index.json").read_text(encoding="utf-8"))
    entries = [
        entry
        for entry in index.get("factors", [])
        if str(entry.get("pool")) in pools
        and (factor_id is None or int(entry.get("factor_id", -1)) == factor_id)
    ]
    if factor_id is not None and not entries:
        raise ValueError(
            f"factor_id {factor_id} not found in pools {sorted(pools)} under {factor_pool_root}"
        )
    selected_by_candidate: dict[str, dict[str, Any]] = {}
    pool_rank = {"elite": 0, "qualified": 1}
    for entry in sorted(
        entries,
        key=lambda item: (
            pool_rank.get(str(item.get("pool")), 99),
            int(item.get("factor_id", 0)),
        ),
    ):
        selected_by_candidate.setdefault(str(entry["candidate_id"]), entry)
    return sorted(selected_by_candidate.values(), key=lambda item: int(item["factor_id"]))


def _completed_factor_reports(
    output_root: Path,
    *,
    start_date: str,
    end_date: str,
    quantiles: int,
) -> dict[int, Path]:
    """Find completed factor reports for the same requested analysis setup."""

    completed: dict[int, Path] = {}
    if not output_root.exists():
        return completed
    for batch_report_path in sorted(output_root.glob("*/batch_report.json")):
        try:
            batch = json.loads(batch_report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (
            str(batch.get("start_date_requested")) != start_date
            or str(batch.get("end_date_requested")) != end_date
            or int(batch.get("quantiles", -1)) != int(quantiles)
        ):
            continue
        batch_dir = batch_report_path.parent
        for result in batch.get("results", []):
            try:
                factor_id = int(result["factor_id"])
            except (KeyError, TypeError, ValueError):
                continue
            report_path = batch_dir / str(result.get("report_path", ""))
            if report_path.exists():
                completed.setdefault(factor_id, report_path)
    return completed


def _write_summary_csv(path: Path, results: list[dict[str, Any]]) -> None:
    rows = []
    for result in results:
        row = {
            key: result[key]
            for key in (
                "factor_id",
                "candidate_id",
                "factor_name",
                "pool",
                "domain_agent",
                "run_id",
                "start_date",
                "end_date",
                "report_path",
                "tear_sheets",
                "factor_direction",
            )
        }
        row["tear_sheets"] = ";".join(result["tear_sheets"])
        row.update(result["summary"])
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_markdown_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Full Factor Pool Backtest",
        "",
        f"- Batch: `{summary['batch_id']}`",
        f"- Pools: `{', '.join(summary['pools'])}`",
        f"- Requested window: `{summary['start_date_requested']}` to `{summary['end_date_requested']}`",
        f"- Quantiles: `{summary['quantiles']}`",
        f"- Memory update: `{summary['memory_update']}`",
        f"- Skipped already completed: `{len(summary['skipped'])}`",
        "",
        "## Factors",
        "",
        "| factor_id | pool | dir | factor | RankIC | RankICIR | top excess mean | top excess positive | long-short mean | long-short positive | tear sheet |",
        "|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for result in summary["results"]:
        item = result["summary"]
        tear_sheet = result["tear_sheets"][0] if result["tear_sheets"] else ""
        lines.append(
            "| {factor_id} | {pool} | {direction} | {factor_name} | {ic:.4f} | "
            "{icir:.4f} | {top_mean:.6f} | {top_pos:.4f} | {ls_mean:.6f} | "
            "{ls_pos:.4f} | [{tear_sheet}]({tear_sheet}) |".format(
                factor_id=result["factor_id"],
                pool=result["pool"],
                direction=result["factor_direction"],
                factor_name=result["factor_name"],
                ic=float(item["rank_ic_mean"]),
                icir=float(item["rank_icir"]),
                top_mean=float(item["top_quantile_excess_mean_return"]),
                top_pos=float(item["top_quantile_excess_positive_rate"]),
                ls_mean=float(item["long_short_mean_return"]),
                ls_pos=float(item["long_short_positive_rate"]),
                tear_sheet=tear_sheet,
            )
        )
    if summary["skipped"]:
        lines.extend(
            [
                "",
                "## Skipped",
                "",
                "| factor_id | pool | factor | reason | existing report |",
                "|---:|---|---|---|---|",
            ]
        )
        for result in summary["skipped"]:
            existing = result["existing_report_path"]
            lines.append(
                "| {factor_id} | {pool} | {factor_name} | {reason} | {existing} |".format(
                    factor_id=result["factor_id"],
                    pool=result["pool"],
                    factor_name=result["factor_name"],
                    reason=result["reason"],
                    existing=existing,
                )
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
