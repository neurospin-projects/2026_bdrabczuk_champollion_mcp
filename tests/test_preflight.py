from __future__ import annotations

import os
from pathlib import Path

import pytest

from champollion_sulcal_mcp import preflight


@pytest.mark.unit
def test_resolve_uses_env(tmp_path, monkeypatch):
    monkeypatch.setenv("CHAMPOLLION_PIPELINE_DIR", str(tmp_path))
    result = preflight.resolve_pipeline_dir()
    assert result == tmp_path


# --- REQ-MCPSELF-BDRABCZUK-EFA902B509A8: CHAMPOLLION_PIPELINE_DIR is required ---


@pytest.mark.unit
@pytest.mark.parametrize("case", ["unset", "empty", "file", "missing"])
def test_resolve_without_existing_dir_raises_naming_variable(case, tmp_path, monkeypatch):
    """Unless CHAMPOLLION_PIPELINE_DIR names an existing directory, a FileNotFoundError names the variable.

    A champollion_pipeline directory is planted where a sibling-folder fallback would look
    (three levels above the module file), so no fallback may hide the missing variable.
    """
    monkeypatch.setattr(preflight, "__file__", str(tmp_path / "repo" / "src" / "pkg" / "preflight.py"))
    (tmp_path / "champollion_pipeline").mkdir()
    a_file = tmp_path / "a_file"
    a_file.write_text("")
    if case == "unset":
        monkeypatch.delenv("CHAMPOLLION_PIPELINE_DIR", raising=False)
    else:
        value = {"empty": "", "file": str(a_file), "missing": str(tmp_path / "missing")}[case]
        monkeypatch.setenv("CHAMPOLLION_PIPELINE_DIR", value)

    with pytest.raises(FileNotFoundError, match="CHAMPOLLION_PIPELINE_DIR"):
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
def test_stage_scripts_exist_in_real_pipeline_checkout():
    """Against the real checkout named by CHAMPOLLION_PIPELINE_DIR, every STAGE_SCRIPTS entry is under scripts_dir."""
    if not os.environ.get("CHAMPOLLION_PIPELINE_DIR"):
        pytest.skip("CHAMPOLLION_PIPELINE_DIR is not set")
    loc = preflight.detect()

    missing = [fname for fname in preflight.STAGE_SCRIPTS.values() if not (loc.scripts_dir / fname).is_file()]
    assert missing == [], f"scripts not found under {loc.scripts_dir}: {missing}"


@pytest.mark.unit
def test_hf_token_detection(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "test-token")
    assert preflight.check_hf_token_present() is True
    monkeypatch.delenv("HF_TOKEN")
    assert preflight.check_hf_token_present() is False


# --- REQ-MCPSELF-BDRABCZUK-2F4EAE4E90B7: README marks the variable as required ---

README = Path(__file__).resolve().parent.parent / "README.md"


@pytest.mark.unit
def test_readme_marks_pipeline_dir_required():
    """The Default cell of the README Configuration row for CHAMPOLLION_PIPELINE_DIR reads "Required"."""
    rows = [line for line in README.read_text().splitlines() if line.startswith("| `CHAMPOLLION_PIPELINE_DIR` |")]
    assert len(rows) == 1, rows
    cells = [cell.strip() for cell in rows[0].strip().strip("|").split("|")]
    assert len(cells) == 3, cells
    assert cells[2].strip("*_ ").lower() == "required", cells[2]
