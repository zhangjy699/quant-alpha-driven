"""Runtime numerical guard for generated alpha functions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from cogalpha.execution import AlphaExecutionError, execute_alpha_candidate
from cogalpha.schemas import AlphaCandidate, GuardIssue, GuardReport, GuardStatus


@dataclass(frozen=True)
class RuntimeAlphaGuardResult:
    """Runtime guard report plus reusable factor values when execution succeeds."""

    report: GuardReport
    factor_values: pd.DataFrame | None = None


def run_runtime_alpha_code_guard(
    candidate: AlphaCandidate,
    ohlcv_panel: pd.DataFrame,
    max_nan_fraction: float = 0.30,
) -> GuardReport:
    """Execute an AlphaCandidate and reject unstable numerical outputs."""

    return run_runtime_alpha_code_guard_with_values(
        candidate,
        ohlcv_panel,
        max_nan_fraction=max_nan_fraction,
    ).report


def run_runtime_alpha_code_guard_with_values(
    candidate: AlphaCandidate,
    ohlcv_panel: pd.DataFrame,
    max_nan_fraction: float = 0.30,
) -> RuntimeAlphaGuardResult:
    """Execute an AlphaCandidate once and return guard status plus factor values."""

    try:
        factor_values = execute_alpha_candidate(candidate, ohlcv_panel)
    except AlphaExecutionError as exc:
        return RuntimeAlphaGuardResult(
            report=GuardReport(
                guard_name="runtime_alpha_code",
                status=GuardStatus.FAIL,
                issues=[
                    GuardIssue(
                        code="runtime_error",
                        message=str(exc),
                        location=candidate.candidate_id,
                    )
                ],
            ),
        )

    issues: list[GuardIssue] = []
    numeric_values = factor_values.apply(lambda series: pd.to_numeric(series, errors="coerce"))
    membership_mask = _membership_mask(ohlcv_panel, numeric_values)
    raw_values = numeric_values.to_numpy(dtype=float)[membership_mask.to_numpy(dtype=bool)]
    total_count = raw_values.size

    if total_count == 0:
        issues.append(
            GuardIssue(
                code="empty_output",
                message="Alpha execution produced an empty factor panel.",
                location=candidate.candidate_id,
            )
        )
        nan_fraction = 1.0
        inf_count = 0
    else:
        nan_count = int(np.isnan(raw_values).sum())
        inf_count = int(np.isinf(raw_values).sum())
        nan_fraction = nan_count / total_count
        if nan_count == total_count:
            issues.append(
                GuardIssue(
                    code="all_nan_output",
                    message="Alpha execution produced only NaN values.",
                    location=candidate.candidate_id,
                )
            )
        elif nan_fraction > max_nan_fraction:
            issues.append(
                GuardIssue(
                    code="too_many_nan_values",
                    message=(
                        f"Alpha execution produced NaN fraction {nan_fraction:.3f}, "
                        f"above limit {max_nan_fraction:.3f}."
                    ),
                    location=candidate.candidate_id,
                )
            )
        if inf_count:
            issues.append(
                GuardIssue(
                    code="non_finite_output",
                    message="Alpha execution produced infinite values.",
                    location=candidate.candidate_id,
                )
            )

    status = GuardStatus.FAIL if issues else GuardStatus.PASS
    return RuntimeAlphaGuardResult(
        report=GuardReport(
            guard_name="runtime_alpha_code",
            status=status,
            issues=issues,
            metadata={
                "rows": int(numeric_values.shape[0]),
                "assets": int(numeric_values.shape[1]),
                "evaluated_count": int(total_count),
                "nan_fraction": float(nan_fraction),
                "inf_count": int(inf_count),
            },
        ),
        factor_values=factor_values,
    )


def _membership_mask(
    ohlcv_panel: pd.DataFrame,
    factor_values: pd.DataFrame,
) -> pd.DataFrame:
    membership = pd.Series(
        True,
        index=pd.MultiIndex.from_frame(
            ohlcv_panel.index.to_frame(index=False).loc[:, ["date", "asset"]]
        ),
    )
    membership = membership[~membership.index.duplicated()]
    return (
        membership.unstack("asset")
        .reindex(index=factor_values.index, columns=factor_values.columns)
        .fillna(False)
        .astype(bool)
    )
