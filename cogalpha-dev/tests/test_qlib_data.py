import pandas as pd

from cogalpha.data import (
    QlibInstrumentRange,
    filter_panel_by_instrument_ranges,
    load_qlib_daily_pv_hdf,
)


def test_load_qlib_daily_pv_hdf_normalizes_columns_and_filters_index_symbols(tmp_path):
    index = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2024-01-01"), "SH000300"),
            (pd.Timestamp("2024-01-01"), "SH600000"),
            (pd.Timestamp("2024-01-02"), "SH600000"),
        ],
        names=["datetime", "instrument"],
    )
    raw = pd.DataFrame(
        {
            "$open": [1.0, 10.0, 11.0],
            "$close": [1.1, 10.5, 11.5],
            "$high": [1.2, 11.0, 12.0],
            "$low": [0.9, 9.0, 10.0],
            "$volume": [1000.0, 100.0, 110.0],
            "$factor": [0.1, 0.2, 0.3],
        },
        index=index,
    )
    path = tmp_path / "daily_pv_debug.h5"
    raw.to_hdf(path, key="data")

    panel = load_qlib_daily_pv_hdf(path)

    assert panel.index.names == ["date", "asset"]
    assert panel.index.get_level_values("asset").unique().tolist() == ["SH600000"]
    assert list(panel.columns) == ["open", "high", "low", "close", "volume"]


def test_filter_panel_by_instrument_ranges_uses_point_in_time_membership():
    index = pd.MultiIndex.from_product(
        [
            pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            ["SH600000", "SH600001"],
        ],
        names=["date", "asset"],
    )
    panel = pd.DataFrame(
        {
            "open": [1.0] * len(index),
            "high": [2.0] * len(index),
            "low": [0.5] * len(index),
            "close": [1.5] * len(index),
            "volume": [100.0] * len(index),
        },
        index=index,
    )

    filtered = filter_panel_by_instrument_ranges(
        panel,
        [
            QlibInstrumentRange(
                asset="SH600000",
                start=pd.Timestamp("2024-01-02"),
                end=pd.Timestamp("2024-01-03"),
            ),
            QlibInstrumentRange(
                asset="SH600001",
                start=pd.Timestamp("2024-01-01"),
                end=pd.Timestamp("2024-01-01"),
            ),
        ],
    )

    assert list(filtered.index) == [
        (pd.Timestamp("2024-01-01"), "SH600001"),
        (pd.Timestamp("2024-01-02"), "SH600000"),
        (pd.Timestamp("2024-01-03"), "SH600000"),
    ]
