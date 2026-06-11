"""Incremental human-readable export for elite/qualified factor_pool entries."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cogalpha.factor_pool import DEFAULT_FACTOR_POOL_ROOT, POOL_NAMES

DEFAULT_CATALOG_OUTPUT_DIR = DEFAULT_FACTOR_POOL_ROOT / "catalog"
DEFAULT_CATALOG_FILENAME = "factors.md"
DEFAULT_STATE_FILENAME = "export_state.json"
CATALOG_HEADER = """# CogAlpha Factor Catalog

Incremental export of elite/qualified factors (formula, rationale, code only).
"""


@dataclass(frozen=True)
class FactorCatalogExportResult:
    """Summary of one incremental catalog append."""

    factor_pool_root: Path
    output_dir: Path
    catalog_path: Path
    state_path: Path
    appended_factor_ids: list[int]
    skipped_factor_ids: list[int]


def list_pending_catalog_entries(
    factor_pool_root: Path,
    *,
    pools: tuple[str, ...] = ("elite", "qualified"),
    exported_factor_ids: set[int],
) -> list[dict[str, Any]]:
    """Return index entries in ``pools`` that are not yet exported."""

    index_path = factor_pool_root / "index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"factor_pool index not found: {index_path}")

    index = json.loads(index_path.read_text(encoding="utf-8"))
    allowed_pools = set(pools)
    pending = [
        entry
        for entry in index.get("factors", [])
        if str(entry.get("pool")) in allowed_pools
        and int(entry["factor_id"]) not in exported_factor_ids
    ]
    pending.sort(key=lambda entry: int(entry["factor_id"]))
    return pending


def render_factor_section(*, factor_id: int, factor: dict[str, Any]) -> str:
    """Render one Markdown section with formula, rationale, and code only."""

    formula = factor.get("formula")
    rationale = factor.get("rationale")
    code = factor.get("code")

    formula_text = str(formula).strip() if formula not in (None, "") else "(none)"
    rationale_text = (
        str(rationale).strip() if rationale not in (None, "") else "(none)"
    )
    code_text = str(code).rstrip() if code not in (None, "") else ""

    lines = [
        "---",
        "",
        f"## Factor {factor_id}",
        "",
        f"**Formula:** `{formula_text}`",
        "",
        f"**Rationale:** {rationale_text}",
        "",
    ]
    if code_text:
        lines.extend(["```python", code_text, "```", ""])
    return "\n".join(lines)


def load_export_state(state_path: Path) -> set[int]:
    """Load exported factor_id set from disk."""

    if not state_path.exists():
        return set()
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    return {int(value) for value in payload.get("exported_factor_ids", [])}


def save_export_state(state_path: Path, exported_factor_ids: set[int]) -> None:
    """Persist exported factor_id set to disk."""

    state_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "exported_factor_ids": sorted(exported_factor_ids),
    }
    state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def append_factor_catalog(
    *,
    factor_pool_root: str | Path = DEFAULT_FACTOR_POOL_ROOT,
    output_dir: str | Path = DEFAULT_CATALOG_OUTPUT_DIR,
    output_file: str = DEFAULT_CATALOG_FILENAME,
    state_file: str = DEFAULT_STATE_FILENAME,
    pools: tuple[str, ...] = ("elite", "qualified"),
) -> FactorCatalogExportResult:
    """Append newly exported elite/qualified factors to a Markdown catalog."""

    for pool_name in pools:
        if pool_name not in POOL_NAMES:
            raise ValueError(f"Unsupported pool {pool_name!r}; expected one of {POOL_NAMES}.")

    pool_root = Path(factor_pool_root)
    out_dir = Path(output_dir)
    catalog_path = out_dir / output_file
    state_path = out_dir / state_file

    exported_ids = load_export_state(state_path)
    pending = list_pending_catalog_entries(
        pool_root,
        pools=pools,
        exported_factor_ids=exported_ids,
    )

    appended_ids: list[int] = []
    skipped_ids: list[int] = []
    sections: list[str] = []

    for entry in pending:
        factor_id = int(entry["factor_id"])
        factor_path = pool_root / str(entry["file"])
        factor = json.loads(factor_path.read_text(encoding="utf-8"))
        code = factor.get("code")
        if code in (None, ""):
            print(f"skip factor_id {factor_id}: missing code in {factor_path}")
            skipped_ids.append(factor_id)
            continue
        sections.append(render_factor_section(factor_id=factor_id, factor=factor))
        appended_ids.append(factor_id)

    if sections:
        out_dir.mkdir(parents=True, exist_ok=True)
        if not catalog_path.exists():
            catalog_path.write_text(CATALOG_HEADER + "\n", encoding="utf-8")
        with catalog_path.open("a", encoding="utf-8") as handle:
            handle.write("\n".join(sections))
            if not sections[-1].endswith("\n"):
                handle.write("\n")
        exported_ids.update(appended_ids)
        save_export_state(state_path, exported_ids)

    return FactorCatalogExportResult(
        factor_pool_root=pool_root,
        output_dir=out_dir,
        catalog_path=catalog_path,
        state_path=state_path,
        appended_factor_ids=appended_ids,
        skipped_factor_ids=skipped_ids,
    )
