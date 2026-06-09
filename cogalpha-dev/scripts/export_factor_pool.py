"""Export factor_pool artifacts from a formal MVP run directory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cogalpha.factor_pool import export_factor_pool


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export factor_pool JSON artifacts from a formal MVP final_state.json."
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Formal MVP run directory containing final_state.json.",
    )
    parser.add_argument(
        "--rejected-limit",
        type=int,
        default=8,
        help=(
            "Number of selected rejected_by_fitness factors to export, "
            "mixing weak and promising rejected samples."
        ),
    )
    parser.add_argument(
        "--output-root",
        default="outputs/factor_pool",
        help="Shared factor_pool output root.",
    )
    args = parser.parse_args()

    result = export_factor_pool(
        Path(args.run_dir),
        output_root=Path(args.output_root),
        rejected_limit=args.rejected_limit,
    )
    print(
        json.dumps(
            {
                "run_dir": str(result.run_dir),
                "factor_pool_dir": str(result.factor_pool_dir),
                "counts": result.counts,
                "index_path": str(result.index_path),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
