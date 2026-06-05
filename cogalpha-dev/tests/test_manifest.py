from cogalpha.manifest import build_run_manifest, file_fingerprint, write_run_manifest


def test_file_fingerprint_changes_with_content(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("a: 1\n", encoding="utf-8")
    first = file_fingerprint(path)
    path.write_text("a: 2\n", encoding="utf-8")
    second = file_fingerprint(path)

    assert first.sha256 != second.sha256


def test_run_manifest_records_all_input_surfaces(tmp_path):
    config = tmp_path / "config.yaml"
    skill = tmp_path / "SKILL.md"
    code = tmp_path / "evaluation.py"
    config.write_text("dataset: CSI300\n", encoding="utf-8")
    skill.write_text("---\nname: test\ndescription: test\n---\n", encoding="utf-8")
    code.write_text("print('ok')\n", encoding="utf-8")

    manifest = build_run_manifest(
        manifest_id="preflight-valid",
        purpose="fixed preflight",
        data_version="data-v1",
        config_paths=[config],
        skill_paths=[skill],
        code_paths=[code],
        fixed_inputs=["preflight_intraday_body"],
        model_settings={"model": "none"},
    )
    output = tmp_path / "manifest.json"
    write_run_manifest(output, manifest)

    assert output.exists()
    assert manifest.config_files[0].path.endswith("config.yaml")
    assert manifest.fixed_inputs == ["preflight_intraday_body"]
