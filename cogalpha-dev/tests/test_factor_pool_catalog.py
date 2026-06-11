import hashlib
import json

from cogalpha.factor_pool_catalog import append_factor_catalog, load_export_state


def test_append_factor_catalog_exports_elite_and_qualified(tmp_path):
    factor_pool = _setup_factor_pool(tmp_path)

    result = append_factor_catalog(
        factor_pool_root=factor_pool,
        output_dir=tmp_path / "catalog",
    )

    assert result.appended_factor_ids == [0, 2]
    assert result.skipped_factor_ids == []
    catalog = (tmp_path / "catalog" / "factors.md").read_text(encoding="utf-8")
    assert catalog.count("## Factor ") == 2
    assert "## Factor 0" in catalog
    assert "## Factor 2" in catalog
    assert "**Formula:** `slope(close, 60) / close`" in catalog
    assert "**Rationale:** Long trend strength." in catalog
    assert "def factor_trend_strength(df):" in catalog
    assert "metrics" not in catalog
    assert "run_id" not in catalog
    assert load_export_state(tmp_path / "catalog" / "export_state.json") == {0, 2}


def test_append_factor_catalog_incrementally_appends_new_factors(tmp_path):
    factor_pool = _setup_factor_pool(tmp_path)
    catalog_dir = tmp_path / "catalog"

    first = append_factor_catalog(factor_pool_root=factor_pool, output_dir=catalog_dir)
    first_text = (catalog_dir / "factors.md").read_text(encoding="utf-8")

    _write_factor(
        factor_pool,
        "qualified/alpha-reversal/5.json",
        formula="close - open",
        rationale="Simple gap.",
        code="def factor_gap(df):\n    return df['close'] - df['open']\n",
    )
    _write_index(
        factor_pool,
        [
            _entry(0, "elite/alpha-market-cycle/0.json", "elite"),
            _entry(1, "rejected/alpha-market-cycle/1.json", "rejected"),
            _entry(2, "qualified/alpha-reversal/2.json", "qualified"),
            _entry(5, "qualified/alpha-reversal/5.json", "qualified"),
        ],
    )

    second = append_factor_catalog(factor_pool_root=factor_pool, output_dir=catalog_dir)
    second_text = (catalog_dir / "factors.md").read_text(encoding="utf-8")

    assert first.appended_factor_ids == [0, 2]
    assert second.appended_factor_ids == [5]
    assert second_text.startswith(first_text)
    assert second_text.count("## Factor ") == 3
    assert "## Factor 5" in second_text
    assert load_export_state(catalog_dir / "export_state.json") == {0, 2, 5}


def test_append_factor_catalog_skips_rejected_pool(tmp_path):
    factor_pool = tmp_path / "outputs" / "factor_pool"
    _write_factor(
        factor_pool,
        "rejected/alpha-market-cycle/1.json",
        formula="high - low",
        rationale="Rejected only.",
    )
    _write_index(
        factor_pool,
        [_entry(1, "rejected/alpha-market-cycle/1.json", "rejected")],
    )

    result = append_factor_catalog(
        factor_pool_root=factor_pool,
        output_dir=tmp_path / "catalog",
    )

    assert result.appended_factor_ids == []
    assert not (tmp_path / "catalog" / "factors.md").exists()


def test_append_factor_catalog_is_idempotent_when_no_new_factors(tmp_path):
    factor_pool = _setup_factor_pool(tmp_path)
    catalog_dir = tmp_path / "catalog"

    append_factor_catalog(factor_pool_root=factor_pool, output_dir=catalog_dir)
    first_hash = hashlib.sha256(
        (catalog_dir / "factors.md").read_bytes()
    ).hexdigest()

    second = append_factor_catalog(factor_pool_root=factor_pool, output_dir=catalog_dir)
    second_hash = hashlib.sha256(
        (catalog_dir / "factors.md").read_bytes()
    ).hexdigest()

    assert second.appended_factor_ids == []
    assert first_hash == second_hash


def _setup_factor_pool(tmp_path):
    factor_pool = tmp_path / "outputs" / "factor_pool"
    _write_factor(
        factor_pool,
        "elite/alpha-market-cycle/0.json",
        formula="slope(close, 60) / close",
        rationale="Long trend strength.",
        code="def factor_trend_strength(df):\n    return df['close']\n",
    )
    _write_factor(
        factor_pool,
        "rejected/alpha-market-cycle/1.json",
        formula="high - low",
        rationale="Rejected only.",
    )
    _write_factor(
        factor_pool,
        "qualified/alpha-reversal/2.json",
        formula="close - ts_mean(close, 5)",
        rationale="Short reversal.",
        code="def factor_reversal(df):\n    return df['close']\n",
    )
    _write_index(
        factor_pool,
        [
            _entry(0, "elite/alpha-market-cycle/0.json", "elite"),
            _entry(1, "rejected/alpha-market-cycle/1.json", "rejected"),
            _entry(2, "qualified/alpha-reversal/2.json", "qualified"),
        ],
    )
    return factor_pool


def _write_index(factor_pool, entries):
    payload = {
        "next_factor_id": max(int(entry["factor_id"]) for entry in entries) + 1,
        "counts": {},
        "factors": entries,
    }
    factor_pool.mkdir(parents=True, exist_ok=True)
    (factor_pool / "index.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _entry(factor_id, file, pool):
    return {
        "factor_id": factor_id,
        "file": file,
        "pool": pool,
        "domain_agent": file.split("/")[1],
        "run_id": "run-1",
        "candidate_id": f"candidate-{factor_id}",
        "factor_name": f"factor_{factor_id}",
    }


def _write_factor(
    factor_pool,
    relative_path,
    *,
    formula,
    rationale,
    code=None,
):
    path = factor_pool / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "factor_name": f"factor_{path.stem}",
        "formula": formula,
        "rationale": rationale,
        "required_columns": ["open", "high", "low", "close", "volume"],
        "allowed_libraries": ["np", "pd"],
        "metrics": {"ic": 0.01},
    }
    if code is not None:
        payload["code"] = code
    else:
        payload["code"] = f"def factor_{path.stem}(df):\n    return df['close']\n"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
