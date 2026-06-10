"""Run the CogAlpha MVP graph with real skills and prepared market data."""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path

from cogalpha.config import MVPLoopConfig
from cogalpha.data import load_prepared_baseline_market_data
from cogalpha.evaluation import EvaluationCache, PanelBackedMetricsProvider
from cogalpha.graph import build_mvp_graph
from cogalpha.guards import DeterministicGuardPipeline
from cogalpha.instrumentation import InvocationRecorder, RecordingInvoker
from cogalpha.llm import OpenAICompatibleClient
from cogalpha.manifest import build_run_manifest, write_run_manifest
from cogalpha.registry import DOMAIN_AGENT_SPECS, PROJECT_ROOT, all_skill_refs
from cogalpha.reporting import (
    EvaluationRunReport,
    ReportLayer,
    StopGoDecision,
    write_evaluation_run_report,
)
from cogalpha.schemas import CandidateStage, CogAlphaState
from cogalpha.skill_invocation import SkillInvoker
from cogalpha.skill_loader import StandardSkillLoader

DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-pro"
DEFAULT_DEEPSEEK_REASONING_EFFORT = "high"
DEFAULT_DEEPSEEK_THINKING = "enabled"
DEFAULT_OPENAI_CHAT_MODEL = "gpt-4o-mini-2024-07-18"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a formal CogAlpha MVP experiment.")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--data-dir", default="data/processed/company_all_a")
    parser.add_argument("--split", choices=["train", "valid", "test"], default="valid")
    parser.add_argument("--output-root", default="outputs/experiments")
    parser.add_argument("--agent-limit", type=int, default=3)
    parser.add_argument("--max-generations", type=int, default=1)
    parser.add_argument("--alphas-per-agent", type=int, default=1)
    parser.add_argument("--parent-pool-size", type=int, default=4)
    parser.add_argument("--max-repair-attempts", type=int, default=1)
    parser.add_argument("--inline-references", action="store_true")
    parser.add_argument("--use-factor-memory", action="store_true")
    parser.add_argument("--factor-memory-root", default="outputs/factor_memory")
    parser.add_argument("--key-file", default="KEY.md")
    parser.add_argument("--provider", choices=["deepseek", "openai", "custom"], default="deepseek")
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--reasoning-effort", default=None)
    parser.add_argument("--thinking", choices=["enabled", "disabled"], default=None)
    parser.add_argument("--max-tokens", type=int, default=8192)
    args = parser.parse_args()

    _load_key_file(args.key_file)
    _configure_llm_provider(args)
    run_id = args.run_id or datetime.now(UTC).strftime("formal-mvp-%Y%m%d-%H%M%S")
    run_dir = Path(args.output_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    metadata = json.loads((Path(args.data_dir) / "metadata.json").read_text(encoding="utf-8"))
    market_data = load_prepared_baseline_market_data(args.data_dir)
    data_split = market_data.split(args.split)
    cache = EvaluationCache(run_dir / "evaluation_cache.jsonl")
    metrics_provider = PanelBackedMetricsProvider.from_split(
        data_split,
        data_version=metadata["data_version"],
        cache=cache,
    )
    guard_pipeline = DeterministicGuardPipeline(
        runtime_ohlcv_panel=data_split.ohlcv_panel,
    )
    client = OpenAICompatibleClient.from_env()
    skill_invoker = SkillInvoker(
        loader=StandardSkillLoader(PROJECT_ROOT / "skills"),
        client=client,
        inline_references=args.inline_references,
        retrieval_cache_root=(
            Path(args.factor_memory_root) / "retrieval_cache"
            if args.use_factor_memory
            else None
        ),
    )
    invoker = RecordingInvoker(
        inner=skill_invoker,
        recorder=InvocationRecorder(run_dir / "skill_invocations.jsonl"),
        context_variant=_context_variant(args),
    )
    config = MVPLoopConfig(
        alphas_per_domain_agent=args.alphas_per_agent,
        max_generations=args.max_generations,
        max_repair_attempts=args.max_repair_attempts,
        parent_pool_size=args.parent_pool_size,
    )
    (run_dir / "actual_config.json").write_text(
        config.model_dump_json(indent=2),
        encoding="utf-8",
    )
    agent_specs = DOMAIN_AGENT_SPECS[: args.agent_limit]
    graph = build_mvp_graph(
        invoker=invoker,
        config=config,
        metrics_provider=metrics_provider,
        guard_pipeline=guard_pipeline,
        agent_specs=agent_specs,
    )
    initial_state = CogAlphaState(metadata={"run_id": run_id, "split": args.split})
    result = CogAlphaState.model_validate(graph.invoke(initial_state.model_dump(mode="python")))
    (run_dir / "final_state.json").write_text(
        result.model_dump_json(indent=2),
        encoding="utf-8",
    )
    _assert_formal_run_complete(result)
    summary = _summarize_run(result, invoker, metrics_provider, args, metadata)
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    manifest = build_run_manifest(
        manifest_id=run_id,
        purpose="Formal MVP graph run",
        data_version=metadata["data_version"],
        config_paths=["configs/baseline.yaml", "configs/mvp.yaml"],
        skill_paths=[ref.path for ref in all_skill_refs()],
        code_paths=[
            "scripts/run_formal_mvp.py",
            "cogalpha/config.py",
            "cogalpha/schemas.py",
            "cogalpha/graph/mvp.py",
            "cogalpha/nodes/domain_agents.py",
            "cogalpha/nodes/quality_pipeline.py",
            "cogalpha/nodes/fitness.py",
            "cogalpha/nodes/evolution.py",
            "cogalpha/candidate_lifecycle.py",
            "cogalpha/data.py",
            "cogalpha/evaluation.py",
            "cogalpha/fitness.py",
            "cogalpha/feedback.py",
            "cogalpha/guards/pipeline.py",
            "cogalpha/guards/alpha_code.py",
            "cogalpha/guards/alpha_runtime.py",
            "cogalpha/execution.py",
            "cogalpha/instrumentation.py",
            "cogalpha/llm/client.py",
            "cogalpha/skill_invocation.py",
            "cogalpha/skill_loader.py",
            "cogalpha/skill_nodes.py",
        ],
        fixed_inputs=[
            args.split,
            str(Path(args.data_dir) / "metadata.json"),
            str(Path(args.data_dir) / f"{args.split}_ohlcv.parquet"),
            str(Path(args.data_dir) / f"{args.split}_forward_returns.parquet"),
            *(
                [str(Path(args.factor_memory_root) / "retrieval_cache")]
                if args.use_factor_memory
                else []
            ),
            *[spec.skill_name for spec in agent_specs],
        ],
        model_settings={
            "model": client.model,
            "base_url": client.base_url,
            "temperature": str(client.temperature),
            "reasoning_effort": str(client.reasoning_effort),
            "thinking": str(client.thinking),
            "max_tokens": str(client.max_tokens),
            "inline_references": str(args.inline_references),
            "use_factor_memory": str(args.use_factor_memory),
            "factor_memory_root": (
                str(Path(args.factor_memory_root)) if args.use_factor_memory else ""
            ),
        },
        notes="Real LLM Skill-Driven DAG run. Test split must not be used for tuning.",
    )
    write_run_manifest(run_dir / "run_manifest.json", manifest)
    report = _build_formal_run_report(
        summary=summary,
        data_version=metadata["data_version"],
        manifest_path=run_dir / "run_manifest.json",
    )
    write_evaluation_run_report(run_dir / "evaluation_run_report.json", report)
    print(json.dumps(summary, indent=2, sort_keys=True))


def _load_key_file(path: str) -> None:
    key_path = Path(path)
    if not key_path.exists():
        return
    parsed: dict[str, str] = {}
    for raw_line in key_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.match(r"(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*[:=]\s*(.+)", line)
        if not match:
            continue
        name, value = match.groups()
        parsed[name] = value.strip().strip('"').strip("'")

    for name, value in parsed.items():
        canonical_name = _canonical_llm_env_name(name)
        if canonical_name:
            os.environ.setdefault(canonical_name, value)
        elif name.isupper():
            os.environ.setdefault(name, value)

def _configure_llm_provider(args: argparse.Namespace) -> None:
    if args.provider == "deepseek":
        os.environ.setdefault("COGALPHA_LLM_BASE_URL", DEFAULT_DEEPSEEK_BASE_URL)
        os.environ.setdefault("COGALPHA_LLM_MODEL", DEFAULT_DEEPSEEK_MODEL)
        os.environ.setdefault("COGALPHA_LLM_REASONING_EFFORT", DEFAULT_DEEPSEEK_REASONING_EFFORT)
        os.environ.setdefault("COGALPHA_LLM_THINKING", DEFAULT_DEEPSEEK_THINKING)
        if args.max_tokens is not None:
            os.environ.setdefault("COGALPHA_LLM_MAX_TOKENS", str(args.max_tokens))
    elif args.provider == "openai":
        os.environ.setdefault("COGALPHA_LLM_MODEL", DEFAULT_OPENAI_CHAT_MODEL)

    if args.model:
        os.environ["COGALPHA_LLM_MODEL"] = args.model
    if args.base_url:
        os.environ["COGALPHA_LLM_BASE_URL"] = args.base_url
    if args.reasoning_effort:
        os.environ["COGALPHA_LLM_REASONING_EFFORT"] = args.reasoning_effort
    if args.thinking:
        os.environ["COGALPHA_LLM_THINKING"] = args.thinking
    if args.max_tokens is not None:
        os.environ["COGALPHA_LLM_MAX_TOKENS"] = str(args.max_tokens)


def _canonical_llm_env_name(name: str) -> str | None:
    normalized = name.lower().replace("-", "_")
    if normalized in {
        "key",
        "api_key",
        "llm_api_key",
        "deepseek_api_key",
        "openai_api_key",
    }:
        return "COGALPHA_LLM_API_KEY"
    if normalized in {"model", "llm_model", "chat_model", "deepseek_model", "openai_model"}:
        return "COGALPHA_LLM_MODEL"
    if normalized in {
        "base_url",
        "api_base",
        "llm_base_url",
        "deepseek_base_url",
        "openai_base_url",
    }:
        return "COGALPHA_LLM_BASE_URL"
    if normalized in {"reasoning_effort", "deepseek_reasoning_effort"}:
        return "COGALPHA_LLM_REASONING_EFFORT"
    if normalized in {"thinking", "deepseek_thinking"}:
        return "COGALPHA_LLM_THINKING"
    if normalized in {"max_tokens", "llm_max_tokens", "deepseek_max_tokens"}:
        return "COGALPHA_LLM_MAX_TOKENS"
    return None


def _is_openai_base_url(base_url: str) -> bool:
    return "api.openai.com" in base_url.lower()


def _context_variant(args: argparse.Namespace) -> str:
    variant = "inline_references" if args.inline_references else "minimal"
    if args.use_factor_memory:
        variant = f"{variant}+factor_memory"
    return variant


def _summarize_run(
    state: CogAlphaState,
    invoker: RecordingInvoker,
    metrics_provider: PanelBackedMetricsProvider,
    args: argparse.Namespace,
    metadata: dict,
) -> dict:
    return {
        "run_id": state.metadata.get("run_id"),
        "created_at": datetime.now(UTC).isoformat(),
        "split": args.split,
        "data_version": metadata["data_version"],
        "agent_limit": args.agent_limit,
        "max_generations": args.max_generations,
        "use_factor_memory": args.use_factor_memory,
        "factor_memory_root": (
            str(Path(args.factor_memory_root)) if args.use_factor_memory else None
        ),
        "node_history": [entry.node_name for entry in state.node_history],
        "skill_calls": len(invoker.calls),
        "skill_errors": sum(1 for call in invoker.calls if call["status"] != "ok"),
        "qualified": len(state.qualified_pool),
        "elite": len(state.elite_pool),
        "parent_pool": len(state.parent_pool),
        "rejected": len(state.rejected_pool),
        "remaining_candidates": len(state.candidates),
        "cache_hits": metrics_provider.cache_hits_by_candidate_id,
        "errors": metrics_provider.errors_by_candidate_id,
    }


def _build_formal_run_report(
    *,
    summary: dict,
    data_version: str,
    manifest_path: Path,
) -> EvaluationRunReport:
    blockers: list[str] = []
    workflow_status = StopGoDecision.GO
    if summary["skill_errors"] or summary["remaining_candidates"]:
        workflow_status = StopGoDecision.STOP
        blockers.append("Workflow ended with skill errors or unevaluated candidates.")

    effect_status = StopGoDecision.GO if summary["qualified"] else StopGoDecision.HOLD
    if not summary["qualified"]:
        blockers.append(
            "No candidate qualified on validation; do not treat run as performance evidence."
        )

    promotion_status = StopGoDecision.HOLD
    blockers.append("Promotion requires fixed comparison, review, and rollback pointer.")

    decision = (
        StopGoDecision.STOP if workflow_status == StopGoDecision.STOP else StopGoDecision.HOLD
    )
    return EvaluationRunReport(
        report_id=f"{summary['run_id']}-report",
        purpose="Formal MVP workflow run report",
        data_version=data_version,
        manifest_path=str(manifest_path),
        layers=[
            ReportLayer(
                name="data_contract",
                status=StopGoDecision.GO,
                summary=(
                    "Prepared market-data split uses configured next-open forward returns and recorded "
                    "data_version."
                ),
                artifacts=[str(manifest_path)],
            ),
            ReportLayer(
                name="workflow_execution",
                status=workflow_status,
                summary=(
                    f"Nodes executed: {summary['node_history']}; skill_errors="
                    f"{summary['skill_errors']}; remaining_candidates="
                    f"{summary['remaining_candidates']}."
                ),
                artifacts=["summary.json", "final_state.json", "skill_invocations.jsonl"],
            ),
            ReportLayer(
                name="effect_evaluation",
                status=effect_status,
                summary=(
                    f"qualified={summary['qualified']}, elite={summary['elite']}, "
                    f"rejected={summary['rejected']} on split={summary['split']}."
                ),
                artifacts=["evaluation_cache.jsonl", "final_state.json"],
            ),
            ReportLayer(
                name="promotion_governance",
                status=promotion_status,
                summary="Single run cannot promote prompt, topology, gate, or default behavior.",
                artifacts=["run_manifest.json"],
            ),
        ],
        decision=decision,
        blockers=blockers,
    )


def _assert_formal_run_complete(state: CogAlphaState) -> None:
    """Fail fast when a formal graph run ends before candidates are scored."""

    if not state.node_history:
        raise RuntimeError("Formal run produced no node history.")
    final_node = state.node_history[-1]
    if final_node.node_name != "fitness_gate":
        raise RuntimeError(
            f"Formal run ended at {final_node.node_name!r}; expected final fitness_gate."
        )
    if state.candidates:
        raise RuntimeError(
            f"Formal run ended with {len(state.candidates)} unevaluated candidate(s)."
        )

    evaluated_ids = {
        result.candidate_id
        for node in state.node_history
        for result in node.evaluation_results
    }
    final_pool_ids = {
        candidate.candidate_id
        for candidate in state.qualified_pool + state.elite_pool + state.rejected_pool
        if candidate.stage
        in {
            CandidateStage.QUALIFIED,
            CandidateStage.ELITE,
            CandidateStage.REJECTED_BY_FITNESS,
        }
    }
    missing_ids = sorted(final_pool_ids - evaluated_ids)
    if missing_ids:
        raise RuntimeError(
            "Formal run has final-pool candidate(s) without evaluation results: "
            + ", ".join(missing_ids)
        )


if __name__ == "__main__":
    main()
