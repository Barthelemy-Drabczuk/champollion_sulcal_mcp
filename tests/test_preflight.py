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
    src = pipeline / "src"
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


@pytest.mark.unit
def test_hf_token_detection(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "test-token")
    assert preflight.check_hf_token_present() is True
    monkeypatch.delenv("HF_TOKEN")
    assert preflight.check_hf_token_present() is False
