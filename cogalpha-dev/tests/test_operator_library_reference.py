from pathlib import Path

from cogalpha.registry import DOMAIN_AGENT_SPECS, EVOLUTION_SKILLS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REFERENCES_DIR = PROJECT_ROOT / "skills" / "references"
OPERATOR_REFERENCE = "operator-library.md"
OPERATOR_MANIFEST = "operator-manifest.txt"

CODE_GENERATION_SKILLS = tuple(spec.skill_name for spec in DOMAIN_AGENT_SPECS) + (
    "alpha-mutation",
    "alpha-crossover",
    "alpha-code-repair",
    "alpha-logic-improvement",
)

FORBIDDEN_SUBSTRINGS = (
    "_stub",
    "group_",
    "trade_when",
    "operator_registry",
    "api/operators",
    "Operator Entry Template",
)

FORBIDDEN_CATALOG_ROWS = (
    "| `rank` |",
    "| `zscore` |",
    "| `normalize` |",
    "| `quantile` |",
    "| `scale` |",
    "| `winsorize` |",
    "| `neutralize` |",
    "| `bucket` |",
    "| `densify` |",
    "| `trade_when` |",
    "| `vec_avg` |",
    "| `vec_sum` |",
    "| `orthogonalize` |",
    "| `change_instrument` |",
)


def _load_manifest_names() -> list[str]:
    path = REFERENCES_DIR / OPERATOR_MANIFEST
    names = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        names.append(stripped)
    return names


def _catalog_section(text: str) -> str:
    start = text.index("## Operator Catalog")
    end = text.index("## Pitfalls")
    return text[start:end]


def test_operator_library_and_manifest_exist() -> None:
    assert (REFERENCES_DIR / OPERATOR_REFERENCE).exists()
    assert (REFERENCES_DIR / OPERATOR_MANIFEST).exists()


def test_manifest_covers_operator_library_catalog() -> None:
    doc = (REFERENCES_DIR / OPERATOR_REFERENCE).read_text(encoding="utf-8")
    catalog = _catalog_section(doc)
    missing = [name for name in _load_manifest_names() if f"`{name}`" not in catalog]
    assert not missing, f"Catalog missing operators: {missing[:10]}"


def test_each_manifest_operator_appears_once_in_catalog() -> None:
    catalog = _catalog_section(
        (REFERENCES_DIR / OPERATOR_REFERENCE).read_text(encoding="utf-8")
    )
    duplicates = []
    for name in _load_manifest_names():
        count = catalog.count(f"`{name}`")
        if count != 1:
            duplicates.append((name, count))
    assert not duplicates, f"Operators not appearing exactly once in catalog: {duplicates[:10]}"


def test_operator_library_has_no_forbidden_content() -> None:
    text = (REFERENCES_DIR / OPERATOR_REFERENCE).read_text(encoding="utf-8")
    for forbidden in FORBIDDEN_SUBSTRINGS:
        assert forbidden not in text, f"Forbidden substring found: {forbidden!r}"
    catalog = _catalog_section(text)
    for row in FORBIDDEN_CATALOG_ROWS:
        assert row not in catalog, f"Forbidden catalog row found: {row}"


def test_operator_library_structure() -> None:
    text = (REFERENCES_DIR / OPERATOR_REFERENCE).read_text(encoding="utf-8")
    assert "Raw Factor Policy" in text
    assert "Shared Implementation Patterns" in text
    assert "Operator Catalog" in text
    assert "Pitfalls" in text
    assert len(text.splitlines()) <= 280


def test_code_generation_skills_reference_operator_library() -> None:
    evolution_names = {skill.name for skill in EVOLUTION_SKILLS}
    assert {"alpha-mutation", "alpha-crossover"} <= evolution_names

    for skill_name in CODE_GENERATION_SKILLS:
        skill_path = PROJECT_ROOT / "skills" / skill_name / "SKILL.md"
        text = skill_path.read_text(encoding="utf-8")
        assert OPERATOR_REFERENCE in text, f"Missing operator library reference in {skill_path}"
