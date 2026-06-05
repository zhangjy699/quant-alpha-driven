from cogalpha.registry import DOMAIN_AGENT_SPECS, PROJECT_ROOT, all_skill_refs


def test_domain_agent_registry_matches_paper_count():
    assert len(DOMAIN_AGENT_SPECS) == 21


def test_all_registered_skill_files_exist():
    for skill in all_skill_refs():
        assert skill.path.endswith("/SKILL.md")
        assert (PROJECT_ROOT / skill.path).exists()
