import json

from cogalpha.registry import PROJECT_ROOT
from cogalpha.schemas import AlphaCandidate
from cogalpha.skill_invocation import SkillInvoker
from cogalpha.skill_loader import StandardSkillLoader


class FakeJSONClient:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls = []

    def complete_json(self, context: str, schema_name: str, metadata=None) -> str:
        self.calls.append((context, schema_name, metadata))
        return self.response


class SequentialJSONClient:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls = []

    def complete_json(self, context: str, schema_name: str, metadata=None) -> str:
        self.calls.append((context, schema_name, metadata))
        return self.responses.pop(0)


def test_skill_invoker_validates_json_artifact():
    code = (
        "def factor_close_open_gap(df):\n"
        "    df_copy = df.copy()\n"
        '    df_copy["factor_close_open_gap"] = df_copy["close"] - df_copy["open"]\n'
        '    return df_copy["factor_close_open_gap"]\n'
    )
    response = json.dumps(
        {
            "candidate_id": "candidate-1",
            "alpha": {
                "name": "factor_close_open_gap",
                "code": code,
                "rationale": "Measures intraday body direction.",
            },
        }
    )
    client = FakeJSONClient(response)
    invoker = SkillInvoker(
        loader=StandardSkillLoader(PROJECT_ROOT / "skills"),
        client=client,
    )

    candidate = invoker.invoke(
        "alpha-market-cycle",
        runtime_payload='{"num_candidates": 1}',
        output_schema=AlphaCandidate,
    )

    assert candidate.candidate_id == "candidate-1"
    assert client.calls[0][1] == "AlphaCandidate"
    assert client.calls[0][2]["allowed_tools"] == []


def test_skill_invoker_retries_once_after_schema_validation_error():
    code = (
        "def factor_close_open_gap(df):\n"
        "    df_copy = df.copy()\n"
        '    df_copy["factor_close_open_gap"] = df_copy["close"] - df_copy["open"]\n'
        '    return df_copy["factor_close_open_gap"]\n'
    )
    fixed_response = json.dumps(
        {
            "candidate_id": "candidate-1",
            "alpha": {
                "name": "factor_close_open_gap",
                "code": code,
                "rationale": "Measures intraday body direction.",
            },
        }
    )
    client = SequentialJSONClient(
        [
            json.dumps({"candidate_id": "candidate-1"}),
            fixed_response,
        ]
    )
    invoker = SkillInvoker(
        loader=StandardSkillLoader(PROJECT_ROOT / "skills"),
        client=client,
    )

    candidate = invoker.invoke(
        "alpha-market-cycle",
        runtime_payload='{"num_candidates": 1}',
        output_schema=AlphaCandidate,
    )

    assert candidate.candidate_id == "candidate-1"
    assert len(client.calls) == 2
    assert "Previous Invalid JSON Artifact" in client.calls[1][0]
    assert client.calls[1][2]["retry_reason"] == "schema_validation_error"


def test_skill_invoker_blocks_disabled_model_invocation(tmp_path):
    skill_dir = tmp_path / "skills" / "disabled-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        """---
name: disabled-skill
description: Disabled test skill.
disable-model-invocation: true
---

# Disabled
""",
        encoding="utf-8",
    )

    client = FakeJSONClient("{}")
    invoker = SkillInvoker(loader=StandardSkillLoader(tmp_path / "skills"), client=client)

    try:
        invoker.invoke(
            "disabled-skill",
            runtime_payload="{}",
            output_schema=AlphaCandidate,
        )
    except PermissionError:
        pass
    else:
        raise AssertionError("Expected PermissionError")

    assert client.calls == []


def test_skill_invoker_exposes_frontmatter_policy_metadata(tmp_path):
    skill_dir = tmp_path / "skills" / "policy-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        """---
name: policy-skill
description: Policy test skill.
allowed-tools:
  - Read
  - Grep
model: opus
effort: high
context: fork
---

# Policy
""",
        encoding="utf-8",
    )

    invoker = SkillInvoker(
        loader=StandardSkillLoader(tmp_path / "skills"),
        client=FakeJSONClient("{}"),
    )

    context = invoker.prepare_context(
        "policy-skill",
        runtime_payload="{}",
        output_schema=AlphaCandidate,
    )

    assert context.metadata["allowed_tools"] == ["Read", "Grep"]
    assert context.metadata["model"] == "opus"
    assert context.metadata["effort"] == "high"
    assert context.metadata["context"] == "fork"


def test_skill_invoker_appends_retrieval_context_when_available(tmp_path):
    skill_dir = tmp_path / "skills" / "policy-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        """---
name: policy-skill
description: Policy test skill.
---

# Policy
""",
        encoding="utf-8",
    )
    retrieval_root = tmp_path / "factor_memory" / "retrieval_cache"
    retrieval_root.mkdir(parents=True)
    (retrieval_root / "policy-skill.md").write_text(
        "# Prior Lessons for policy-skill\n\n"
        "## Success Patterns\n"
        "- Preserve normalized pressure signals. Evidence: [1]\n",
        encoding="utf-8",
    )
    invoker = SkillInvoker(
        loader=StandardSkillLoader(tmp_path / "skills"),
        client=FakeJSONClient("{}"),
        retrieval_cache_root=retrieval_root,
    )

    context = invoker.prepare_context(
        "policy-skill",
        runtime_payload="{}",
        output_schema=AlphaCandidate,
    )

    assert "# Retrieved Factor Memory" in context.prompt
    assert "# Prior Lessons for policy-skill" in context.prompt
    assert "Evidence: [1]" in context.prompt


def test_skill_invoker_omits_missing_retrieval_context(tmp_path):
    skill_dir = tmp_path / "skills" / "policy-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        """---
name: policy-skill
description: Policy test skill.
---

# Policy
""",
        encoding="utf-8",
    )
    invoker = SkillInvoker(
        loader=StandardSkillLoader(tmp_path / "skills"),
        client=FakeJSONClient("{}"),
        retrieval_cache_root=tmp_path / "factor_memory" / "retrieval_cache",
    )

    context = invoker.prepare_context(
        "policy-skill",
        runtime_payload="{}",
        output_schema=AlphaCandidate,
    )

    assert "# Retrieved Factor Memory" not in context.prompt


def test_skill_invoker_can_inline_direct_references_for_no_tool_clients(tmp_path):
    skill_dir = tmp_path / "skills" / "inline-test"
    skill_dir.mkdir(parents=True)
    (skill_dir / "reference.md").write_text("Reference contract.", encoding="utf-8")
    (skill_dir / "SKILL.md").write_text(
        """---
name: inline-test
description: Inline references for clients without file tools.
---

Read [Reference](reference.md).
""",
        encoding="utf-8",
    )
    invoker = SkillInvoker(
        loader=StandardSkillLoader(tmp_path / "skills"),
        client=FakeJSONClient("{}"),
        inline_references=True,
    )

    context = invoker.prepare_context(
        "inline-test",
        runtime_payload="{}",
        output_schema=AlphaCandidate,
    )

    assert "# Direct References" in context.prompt
    assert "Reference contract." in context.prompt
