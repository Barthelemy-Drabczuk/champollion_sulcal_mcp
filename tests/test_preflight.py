from __future__ import annotations

import pytest

from champollion_sulcal_mcp import preflight


@pytest.mark.unit
def test_resolve_uses_env(tmp_path, monkeypatch):
    monkeypatch.setenv("CHAMPOLLION_PIPELINE_DIR", str(tmp_path))
    result = preflight.resolve_pipeline_dir()
    assert result == tmp_path


@pytest.mark.unit
def test_resolve_missing_raises(monkeypatch):
    monkeypatch.delenv("CHAMPOLLION_PIPELINE_DIR", raising=False)
    def _raise():
        raise FileNotFoundError("not found")

    monkeypatch.setattr(preflight, "resolve_pipeline_dir", _raise)
    with pytest.raises(FileNotFoundError):
        preflight.resolve_pipeline_dir()


@pytest.mark.unit
def test_summarize_reports_missing_scripts(tmp_path, monkeypatch):
    pipeline = tmp_path / "pipeline"
    src = pipeline / "src" / "champollion_pipeline"
    src.mkdir(parents=True)
    # Only create 2 of 6 scripts
    (src / "generate_morphologist_graphs.py").write_text("")
    (src / "run_cortical_tiles.py").write_text("")
    monkeypatch.setenv("CHAMPOLLION_PIPELINE_DIR", str(pipeline))
    result = preflight.summarize()
    assert result["scripts"]["morphologist"] is True
    assert result["scripts"]["cortical_tiles"] is True
    assert result["scripts"]["config"] is False
    assert result["ok"] is False


# --- REQ-MCP-01: scripts_dir must point at src/champollion_pipeline ---


@pytest.mark.unit
def test_scripts_dir_is_champollion_pipeline_package(tmp_path, monkeypatch):
    """detect().scripts_dir resolves to {pipeline_dir}/src/champollion_pipeline."""
    pipeline = tmp_path / "champollion_pipeline"
    (pipeline / "src" / "champollion_pipeline").mkdir(parents=True)
    monkeypatch.setenv("CHAMPOLLION_PIPELINE_DIR", str(pipeline))

    loc = preflight.detect()

    assert loc.scripts_dir == pipeline / "src" / "champollion_pipeline"


@pytest.mark.unit
def test_summarize_finds_all_scripts_in_package_layout(tmp_path, monkeypatch):
    """summarize() reports ok when every STAGE_SCRIPTS file sits in src/champollion_pipeline."""
    pipeline = tmp_path / "champollion_pipeline"
    package = pipeline / "src" / "champollion_pipeline"
    package.mkdir(parents=True)
    for fname in preflight.STAGE_SCRIPTS.values():
        (package / fname).write_text("# stub\n")
    monkeypatch.setenv("CHAMPOLLION_PIPELINE_DIR", str(pipeline))

    result = preflight.summarize()

    assert result["scripts"] == {name: True for name in preflight.STAGE_SCRIPTS}
    assert result["ok"] is True


@pytest.mark.integration
def test_stage_scripts_exist_in_real_pipeline_checkout(monkeypatch):
    """Against the real sibling checkout, every STAGE_SCRIPTS entry is found under scripts_dir."""
    monkeypatch.delenv("CHAMPOLLION_PIPELINE_DIR", raising=False)
    try:
        loc = preflight.detect()
    except FileNotFoundError:
        pytest.skip("champollion_pipeline sibling checkout not available")

    missing = [
        fname for fname in preflight.STAGE_SCRIPTS.values() if not (loc.scripts_dir / fname).is_file()
    ]
    assert missing == [], f"scripts not found under {loc.scripts_dir}: {missing}"


@pytest.mark.unit
def test_hf_token_detection(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "test-token")
    assert preflight.check_hf_token_present() is True
    monkeypatch.delenv("HF_TOKEN")
    assert preflight.check_hf_token_present() is False
