from cogalpha.guards import run_static_alpha_code_guard
from cogalpha.preflight import fixed_preflight_candidates
from cogalpha.schemas import GuardStatus


def test_fixed_preflight_candidates_are_static_guard_clean():
    candidates = fixed_preflight_candidates()

    assert len(candidates) >= 3
    for candidate in candidates:
        report = run_static_alpha_code_guard(candidate.alpha.code, candidate.alpha.name)
        assert report.status == GuardStatus.PASS


def test_docs_describe_agentic_runner_and_trace_artifacts():
    readme = _project_file("README.md").read_text(encoding="utf-8")
    manifest = _project_file("MANIFEST.md").read_text(encoding="utf-8")

    assert "run_agentic_mvp.py" in readme
    assert "trace.jsonl" in readme
    assert "trace_verification.json" in readme
    assert "run_agentic_mvp.py" in manifest
    assert "cogalpha/tracing.py" in manifest
    assert "cogalpha/verification/trace_verifier.py" in manifest
    assert "compatibility" in readme.lower()


def test_docs_do_not_claim_hidden_test_or_failed_overlay_promotion():
    combined_docs = (
        _project_file("README.md").read_text(encoding="utf-8")
        + "\n"
        + _project_file("MANIFEST.md").read_text(encoding="utf-8")
    ).lower()

    assert "hidden-test tuning" not in combined_docs
    assert "rankguard slowtrend" not in combined_docs
    assert "f1 paper-scale" not in combined_docs


def _project_file(path: str):
    return __import__("pathlib").Path(__file__).resolve().parents[1] / path
