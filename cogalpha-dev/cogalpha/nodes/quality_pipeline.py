"""Quality pipeline node for generated alpha candidates."""

from __future__ import annotations

from dataclasses import dataclass, field

from cogalpha.candidate_lifecycle import (
    record_quality_acceptance,
    record_quality_rejection,
    record_repair,
)
from cogalpha.config import MVPLoopConfig
from cogalpha.guards import DeterministicGuardPipeline
from cogalpha.schemas import (
    AlphaCandidate,
    CogAlphaState,
    DAGNodeResult,
    GuardIssue,
    GuardReport,
    GuardStatus,
    QualityDecision,
    QualitySkillRequest,
    QualityVerdict,
)
from cogalpha.skill_nodes import SkillNodeRuntime, StructuredArtifactInvoker


@dataclass
class QualityPipelineNode:
    """Run deterministic guards and quality skills in the first-version order."""

    invoker: StructuredArtifactInvoker
    config: MVPLoopConfig
    guard_pipeline: DeterministicGuardPipeline = field(default_factory=DeterministicGuardPipeline)

    def __call__(self, state: dict) -> dict:
        parsed = CogAlphaState.model_validate(state)
        accepted: list[AlphaCandidate] = []
        rejected: list[AlphaCandidate] = []
        all_guard_reports: list[GuardReport] = []
        all_quality_decisions: list[QualityDecision] = []

        for candidate in parsed.candidates:
            try:
                result = self._process_candidate(candidate)
            except Exception as exc:  # noqa: BLE001 - isolate one bad LLM artifact
                rejected_candidate = record_quality_rejection(candidate)
                result = _QualityResult(
                    accepted=None,
                    rejected=rejected_candidate,
                    guard_reports=[
                        GuardReport(
                            guard_name="quality_pipeline_exception",
                            status=GuardStatus.FAIL,
                            issues=[
                                GuardIssue(
                                    code="quality_skill_error",
                                    message=str(exc),
                                    location=candidate.candidate_id,
                                )
                            ],
                        )
                    ],
                    quality_decisions=[],
                )
            all_guard_reports.extend(result.guard_reports)
            all_quality_decisions.extend(result.quality_decisions)
            if result.accepted is not None:
                accepted.append(result.accepted)
            else:
                rejected.append(result.rejected or candidate)

        parsed.candidates = accepted
        parsed.rejected_pool.extend(rejected)
        parsed.node_history.append(
            DAGNodeResult(
                node_name="quality_pipeline",
                candidates=accepted,
                guard_reports=all_guard_reports,
                quality_decisions=all_quality_decisions,
                metadata={
                    "accepted": len(accepted),
                    "rejected": len(rejected),
                },
            )
        )
        return parsed.model_dump(mode="python")

    def _process_candidate(self, candidate: AlphaCandidate) -> _QualityResult:
        guard_reports: list[GuardReport] = []
        quality_decisions: list[QualityDecision] = []

        current = candidate
        guard_outcome = self.guard_pipeline.run(current)
        guard_reports.extend(guard_outcome.reports)
        if guard_outcome.failed:
            repaired = self._attempt_code_repair(current, guard_reports, quality_decisions)
            if repaired is None:
                current = record_quality_rejection(current)
                return _QualityResult(None, current, guard_reports, quality_decisions)
            current = repaired
            guard_outcome = self.guard_pipeline.run(current)
            guard_reports.extend(guard_outcome.reports)
            if guard_outcome.failed:
                current = record_quality_rejection(current)
                return _QualityResult(None, current, guard_reports, quality_decisions)

        code_quality = self._invoke_quality_skill(
            "alpha-code-quality",
            current,
            guard_reports,
            quality_decisions,
        )
        quality_decisions.append(code_quality)
        if code_quality.verdict == QualityVerdict.REJECT:
            current = record_quality_rejection(current)
            return _QualityResult(None, current, guard_reports, quality_decisions)
        if code_quality.verdict == QualityVerdict.REPAIR:
            repaired = self._attempt_code_repair(current, guard_reports, quality_decisions)
            if repaired is None:
                current = record_quality_rejection(current)
                return _QualityResult(None, current, guard_reports, quality_decisions)
            current = repaired
            guard_outcome = self.guard_pipeline.run(current)
            guard_reports.extend(guard_outcome.reports)
            if guard_outcome.failed:
                current = record_quality_rejection(current)
                return _QualityResult(None, current, guard_reports, quality_decisions)

        judge = self._invoke_quality_skill("alpha-judge", current, guard_reports, quality_decisions)
        quality_decisions.append(judge)
        if judge.verdict == QualityVerdict.REJECT:
            current = record_quality_rejection(current)
            return _QualityResult(None, current, guard_reports, quality_decisions)
        if judge.verdict == QualityVerdict.REPAIR:
            improved = self._attempt_logic_improvement(current, guard_reports, quality_decisions)
            if improved is None:
                current = record_quality_rejection(current)
                return _QualityResult(None, current, guard_reports, quality_decisions)
            current = improved
            guard_outcome = self.guard_pipeline.run(current)
            guard_reports.extend(guard_outcome.reports)
            if guard_outcome.failed:
                current = record_quality_rejection(current)
                return _QualityResult(None, current, guard_reports, quality_decisions)

        current = record_quality_acceptance(current)
        return _QualityResult(current, None, guard_reports, quality_decisions)

    def _attempt_code_repair(
        self,
        candidate: AlphaCandidate,
        guard_reports: list[GuardReport],
        quality_decisions: list[QualityDecision],
    ) -> AlphaCandidate | None:
        for attempt in range(self.config.max_repair_attempts):
            decision = self._invoke_quality_skill(
                "alpha-code-repair",
                candidate,
                guard_reports,
                quality_decisions,
                attempt=attempt,
            )
            quality_decisions.append(decision)
            if decision.repaired_candidate is not None:
                return record_repair(decision.repaired_candidate)
            if decision.verdict == QualityVerdict.REJECT:
                return None
        return None

    def _attempt_logic_improvement(
        self,
        candidate: AlphaCandidate,
        guard_reports: list[GuardReport],
        quality_decisions: list[QualityDecision],
    ) -> AlphaCandidate | None:
        decision = self._invoke_quality_skill(
            "alpha-logic-improvement",
            candidate,
            guard_reports,
            quality_decisions,
        )
        quality_decisions.append(decision)
        if decision.repaired_candidate is not None:
            return record_repair(decision.repaired_candidate)
        return None

    def _invoke_quality_skill(
        self,
        skill_name: str,
        candidate: AlphaCandidate,
        guard_reports: list[GuardReport],
        quality_decisions: list[QualityDecision],
        attempt: int = 0,
    ) -> QualityDecision:
        request = QualitySkillRequest(
            candidate=candidate,
            guard_reports=guard_reports,
            previous_decisions=quality_decisions,
            attempt=attempt,
        )
        return SkillNodeRuntime(self.invoker).quality_decision(skill_name, request)


@dataclass(frozen=True)
class _QualityResult:
    accepted: AlphaCandidate | None
    rejected: AlphaCandidate | None
    guard_reports: list[GuardReport]
    quality_decisions: list[QualityDecision]
