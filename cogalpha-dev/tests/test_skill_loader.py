from cogalpha.registry import PROJECT_ROOT
from cogalpha.skill_loader import StandardSkillLoader, _parse_simple_frontmatter


def test_discover_reads_standard_skill_frontmatter():
    loader = StandardSkillLoader(PROJECT_ROOT / "skills")
    skills = loader.discover()

    assert "alpha-market-cycle" in skills
    assert skills["alpha-market-cycle"].description.startswith("Generates OHLCV alpha candidates")


def test_frontmatter_parser_preserves_cc_skill_fields():
    data = _parse_simple_frontmatter(
        """
name: demo
description: Demo skill
allowed-tools:
  - Read
  - Grep
argument-hint: "<topic>"
arguments: [topic, depth]
model: opus
effort: high
context: fork
disable-model-invocation: true
user-invocable: false
paths:
  - "src/**/*.py"
"""
    )

    assert data["allowed-tools"] == ["Read", "Grep"]
    assert data["arguments"] == ["topic", "depth"]
    assert data["disable-model-invocation"] == "true"
    assert data["paths"] == ["src/**/*.py"]


def test_load_includes_direct_references_only():
    loader = StandardSkillLoader(PROJECT_ROOT / "skills")
    loaded = loader.load("alpha-market-cycle")

    reference_names = {path.name for path in loaded.references}
    assert "alpha-factor-contract.md" in reference_names
    assert "structured-artifacts.md" in reference_names
    assert "diversified-guidance.md" in reference_names

    # Direct references are loaded, but links inside references are not recursively expanded.
    assert all(path.parent.name == "references" for path in loaded.references)


def test_assemble_context_matches_cc_default_shape():
    loader = StandardSkillLoader(PROJECT_ROOT / "skills")
    context = loader.assemble_context(
        "alpha-market-cycle",
        runtime_payload='{"num_candidates": 1}',
        output_schema="AlphaCandidate",
    )

    invoked_idx = context.index("# Invoked Skill")
    payload_idx = context.index("# Runtime Payload")
    schema_idx = context.index("# Required Output Schema")
    assert invoked_idx < payload_idx < schema_idx
    assert "Base directory for this skill:" in context
    assert "# Direct References" not in context


def test_assemble_context_can_inline_references_as_explicit_fallback():
    loader = StandardSkillLoader(PROJECT_ROOT / "skills")
    context = loader.assemble_context(
        "alpha-market-cycle",
        runtime_payload='{"num_candidates": 1}',
        output_schema="AlphaCandidate",
        inline_references=True,
    )

    references_idx = context.index("# Direct References")
    payload_idx = context.index("# Runtime Payload")
    assert references_idx < payload_idx
