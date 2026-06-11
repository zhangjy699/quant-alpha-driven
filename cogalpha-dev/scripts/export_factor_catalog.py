"""Export elite/qualified factor_pool entries to an incremental Markdown catalog."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cogalpha.factor_pool import DEFAULT_FACTOR_POOL_ROOT  # noqa: E402
from cogalpha.factor_pool_catalog import (  # noqa: E402
    DEFAULT_CATALOG_FILENAME,
    DEFAULT_CATALOG_OUTPUT_DIR,
    append_factor_catalog,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Append elite/qualified factor_pool JSON entries to a readable "
            "Markdown catalog (formula, rationale, code only)."
        )
    )
    parser.add_argument(
        "--factor-pool",
        default=str(DEFAULT_FACTOR_POOL_ROOT),
        help="Shared factor_pool root containing index.json.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_CATALOG_OUTPUT_DIR),
        help="Directory for factors.md and export_state.json.",
    )
    parser.add_argument(
        "--output-file",
        default=DEFAULT_CATALOG_FILENAME,
        help="Markdown catalog filename inside --output-dir.",
    )
    parser.add_argument(
        "--pools",
        nargs="+",
        default=["elite", "qualified"],
        help="Factor pools to include. Default: elite qualified.",
    )
    args = parser.parse_args()

    result = append_factor_catalog(
        factor_pool_root=Path(args.factor_pool),
        output_dir=Path(args.output_dir),
        output_file=args.output_file,
        pools=tuple(args.pools),
    )

    if result.appended_factor_ids:
        print(
            f"Appended {len(result.appended_factor_ids)} factors to {result.catalog_path}"
        )
    else:
        print(f"No new factors to append; catalog is up to date at {result.catalog_path}")

    if result.skipped_factor_ids:
        print(
            "Skipped factor_ids without code:",
            ", ".join(str(value) for value in result.skipped_factor_ids),
        )


if __name__ == "__main__":
    main()
