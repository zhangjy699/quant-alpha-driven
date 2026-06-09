"""Load factor_pool entries for independent backtesting."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cogalpha.schemas import AlphaCandidate, AlphaFunction


@dataclass(frozen=True)
class FactorBacktestInput:
    """One factor_pool record reconstructed as an executable candidate."""

    factor_id: int
    factor_file: Path
    pool: str
    domain_agent: str
    run_id: str
    candidate_id: str
    factor_name: str
    fitness_direction: int
    factor: dict[str, Any]
    candidate: AlphaCandidate


def load_factor_from_pool(
    *,
    factor_id: int,
    factor_pool_root: str | Path,
) -> FactorBacktestInput:
    """Load one factor by global factor_id from outputs/factor_pool."""

    pool_root = Path(factor_pool_root)
    index_path = pool_root / "index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"factor_pool index not found: {index_path}")

    index = json.loads(index_path.read_text(encoding="utf-8"))
    entry = next(
        (
            item
            for item in index.get("factors", [])
            if int(item.get("factor_id", -1)) == int(factor_id)
        ),
        None,
    )
    if entry is None:
        raise ValueError(f"factor_id {factor_id} not found in {index_path}")

    factor_file = pool_root / str(entry["file"])
    factor = json.loads(factor_file.read_text(encoding="utf-8"))
    factor_name = str(factor.get("factor_name") or entry["factor_name"])
    fitness_direction = int(factor.get("fitness_direction", 1) or 1)
    if fitness_direction not in {-1, 1}:
        fitness_direction = 1
    candidate = AlphaCandidate(
        candidate_id=str(entry["candidate_id"]),
        alpha=AlphaFunction(
            name=factor_name,
            code=str(factor["code"]),
            formula=factor.get("formula"),
            rationale=str(factor.get("rationale") or "No rationale provided."),
            required_columns=list(factor.get("required_columns") or []),
            allowed_libraries=list(factor.get("allowed_libraries") or []),
        ),
        metadata={
            "factor_id": int(entry["factor_id"]),
            "factor_pool": str(entry["pool"]),
            "domain_agent": str(entry["domain_agent"]),
            "run_id": str(entry["run_id"]),
            "fitness_direction": fitness_direction,
        },
    )
    return FactorBacktestInput(
        factor_id=int(entry["factor_id"]),
        factor_file=factor_file,
        pool=str(entry["pool"]),
        domain_agent=str(entry["domain_agent"]),
        run_id=str(entry["run_id"]),
        candidate_id=str(entry["candidate_id"]),
        factor_name=factor_name,
        fitness_direction=fitness_direction,
        factor=factor,
        candidate=candidate,
    )
