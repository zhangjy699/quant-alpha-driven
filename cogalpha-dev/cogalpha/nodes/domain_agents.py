"""Domain-agent skill node for initial alpha candidate generation."""

from __future__ import annotations

from dataclasses import dataclass

from cogalpha.candidate_lifecycle import record_domain_generation
from cogalpha.config import MVPLoopConfig
from cogalpha.registry import DOMAIN_AGENT_SPECS, DomainAgentSpec
from cogalpha.schemas import (
    AlphaCandidate,
    CogAlphaState,
    DAGNodeResult,
    DomainAgentRequest,
)
from cogalpha.skill_nodes import SkillNodeRuntime, StructuredArtifactInvoker


@dataclass
class DomainAgentNode:
    """Invoke every paper-defined Domain Agent Skill and collect candidates."""

    invoker: StructuredArtifactInvoker
    config: MVPLoopConfig
    agent_specs: tuple[DomainAgentSpec, ...] = DOMAIN_AGENT_SPECS

    def __call__(self, state: dict) -> dict:
        parsed = CogAlphaState.model_validate(state)
        skill_runtime = SkillNodeRuntime(self.invoker)
        generated: list[AlphaCandidate] = []
        errors: list[dict[str, str]] = []

        for spec in self.agent_specs:
            request = DomainAgentRequest(
                skill_name=spec.skill_name,
                paper_agent_name=spec.paper_name,
                level=spec.level,
                layer=spec.layer,
                focus=spec.focus,
                num_candidates=self.config.alphas_per_domain_agent,
                generation=parsed.generation,
                effective_feedback_summary=parsed.feedback.effective_feedback_summary,
                ineffective_feedback_summary=parsed.feedback.ineffective_feedback_summary,
            )
            try:
                batch = skill_runtime.candidate_batch(spec.skill_name, request)
            except Exception as exc:  # noqa: BLE001 - node records per-skill failures
                errors.append({"skill": spec.skill_name, "error": str(exc)})
                continue

            generated.extend(
                record_domain_generation(
                    candidate,
                    skill_name=spec.skill_name,
                    generation=parsed.generation,
                    guidance_mode=request.guidance_mode,
                )
                for candidate in batch.candidates
            )

        parsed.candidates.extend(generated)
        parsed.node_history.append(
            DAGNodeResult(
                node_name="domain_agents",
                candidates=generated,
                metadata={
                    "skills_invoked": len(self.agent_specs),
                    "candidates_generated": len(generated),
                    "errors": errors,
                },
            )
        )
        return parsed.model_dump(mode="python")
