from pathlib import Path

import pandas as pd

from factor_backtest.compact_output import merge_overall_csv, write_overall_csv_from_records


def test_merge_overall_csv_creates_file_when_missing(tmp_path: Path) -> None:
    path = tmp_path / "overall.csv"
    new_table = pd.DataFrame(
        [
            {"factor_id": 1, "ic": 0.03, "icir": 0.3},
            {"factor_id": 2, "ic": 0.02, "icir": 0.2},
        ]
    )

    merged = merge_overall_csv(path, new_table)

    assert len(merged) == 2
    assert list(merged["factor_id"]) == [1, 2]


def test_write_overall_csv_appends_new_factors(tmp_path: Path) -> None:
    output_dir = tmp_path / "compact"
    first = [
        {"factor_id": 1, "ic": 0.03, "icir": 0.3},
        {"factor_id": 2, "ic": 0.02, "icir": 0.2},
    ]
    second = [
        {"factor_id": 3, "ic": 0.01, "icir": 0.1},
    ]

    write_overall_csv_from_records(first, input_dir=tmp_path, output_dir=output_dir)
    write_overall_csv_from_records(second, input_dir=tmp_path, output_dir=output_dir)

    table = pd.read_csv(output_dir / "overall.csv")
    assert list(table["factor_id"]) == [1, 2, 3]


def test_write_overall_csv_replaces_same_factor_id(tmp_path: Path) -> None:
    output_dir = tmp_path / "compact"
    first = [
        {"factor_id": 1, "ic": 0.03, "icir": 0.3},
        {"factor_id": 2, "ic": 0.02, "icir": 0.2},
    ]
    rerun = [
        {"factor_id": 2, "ic": 0.05, "icir": 0.5},
        {"factor_id": 4, "ic": 0.04, "icir": 0.4},
    ]

    write_overall_csv_from_records(first, input_dir=tmp_path, output_dir=output_dir)
    write_overall_csv_from_records(rerun, input_dir=tmp_path, output_dir=output_dir)

    table = pd.read_csv(output_dir / "overall.csv")
    assert list(table["factor_id"]) == [1, 2, 4]
    row_two = table.loc[table["factor_id"] == 2].iloc[0]
    assert row_two["ic"] == 0.05
    assert row_two["icir"] == 0.5
