from cogalpha.registry import (
    DOMAIN_AGENT_SPECS,
    EVOLUTION_SKILLS,
    PROJECT_ROOT,
    QUALITY_SKILLS,
    all_skill_refs,
)

REQUIRED_SKILL_HEADINGS = [
    "## Role",
    "## Inputs",
    "## Workflow",
    "## Anti-Leakage Rules",
    "## Metric Objective",
    "## Self-Check",
    "## Trace Expectations",
    "## Output",
]

REQUIRED_REFERENCE_LINKS = [
    "../references/agentic-workflow.md",
    "../references/metric-objectives.md",
    "../references/trace-grounded-learning.md",
]

REFERENCE_FILES = {
    "agentic-workflow.md": ["## Observe", "## Plan", "## Self-Check", "## Trace Expectations"],
    "metric-objectives.md": ["IC", "RankIC", "ICIR", "RankICIR", "MI"],
    "trace-grounded-learning.md": ["evidence_id", "reviewer", "rollback"],
}


def test_registered_skills_are_loaded_for_richness_contract():
    skills = all_skill_refs()

    assert len(skills) == 27
    for skill in skills:
        assert (PROJECT_ROOT / skill.path).read_text(encoding="utf-8")


def test_shared_rich_skill_references_exist_and_define_contract_terms():
    references_root = PROJECT_ROOT / "skills" / "references"

    for filename, required_terms in REFERENCE_FILES.items():
        text = (references_root / filename).read_text(encoding="utf-8")
        for term in required_terms:
            assert term in text


def test_all_registered_skills_satisfy_rich_workflow_contract():
    for skill in all_skill_refs():
        text = _skill_text(skill.path)
        for heading in REQUIRED_SKILL_HEADINGS:
            assert heading in text, f"{skill.name} missing {heading}"
        for reference_link in REQUIRED_REFERENCE_LINKS:
            assert reference_link in text, f"{skill.name} missing {reference_link}"


def test_domain_skills_retain_paper_agent_focus():
    for spec in DOMAIN_AGENT_SPECS:
        text = _skill_text(spec.to_skill_ref().path)

        assert spec.paper_name in text
        assert spec.layer in text
        assert spec.focus in text


def test_quality_skills_define_accept_repair_reject_guidance():
    for skill in QUALITY_SKILLS:
        text = _skill_text(skill.path)

        assert "accept" in text
        assert "repair" in text
        assert "reject" in text


def test_evolution_skills_preserve_lineage_and_parent_strength_targets():
    for skill in EVOLUTION_SKILLS:
        text = _skill_text(skill.path)

        assert "lineage" in text
        assert "parent_strength_target" in text


def _skill_text(path: str) -> str:
    return (PROJECT_ROOT / path).read_text(encoding="utf-8")
