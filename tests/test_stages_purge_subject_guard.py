"""purge_subject tool refuses an unsafe subject before launching a job (TASK-PATTERNS-BDRABCZUK-077C3B91AE0E).

The MCP tool mirrors rules G1-G3 of the pipeline's utils/subject_name.py::subject_name_problem
(it runs purge_subject.py by subprocess, so it cannot import them). runner.launch is replaced
by a recorder, so no pipeline script is executed.
"""

from __future__ import annotations

import os
import re

import pytest
from fastmcp.exceptions import ToolError

from champollion_sulcal_mcp.tools import stages

pytestmark = pytest.mark.unit


def _refusal(subject: str, reason: str) -> str:
    """The exact ToolError message the tool raises for `subject`."""
    return re.escape(f"Refusing subject {subject!r}: {reason}.")


class TestPurgeSubjectGuard:
    async def test_empty_subject_refused_before_launch(self, pipeline_dir, tmp_path, launches):
        """REQ-PATTERNS-BDRABCZUK-EC6A1F0884D9: an empty subject raises a ToolError; no job is launched."""
        with pytest.raises(ToolError, match=f"^{_refusal('', 'it is empty')}$"):
            await stages.purge_subject(derivatives=str(tmp_path / "deriv"), subject="")
        assert launches == []

    @pytest.mark.parametrize("subject", [".", ".."])
    async def test_dot_subject_refused_before_launch(self, pipeline_dir, tmp_path, launches, subject):
        """REQ-PATTERNS-BDRABCZUK-8888DFDC19D0: '.' or '..' raises a ToolError; no job is launched."""
        with pytest.raises(ToolError, match="^" + _refusal(subject, "it is '.' or '..'") + "$"):
            await stages.purge_subject(derivatives=str(tmp_path / "deriv"), subject=subject)
        assert launches == []

    @pytest.mark.parametrize("separator", ["sep", "altsep"])
    async def test_separator_subject_refused_before_launch(
        self, pipeline_dir, tmp_path, launches, monkeypatch, separator
    ):
        """REQ-PATTERNS-BDRABCZUK-5B394A5E0942: os.sep or os.altsep in subject raises a ToolError; no job."""
        if separator == "altsep":
            monkeypatch.setattr(os, "altsep", "\\")
        subject = f"..{getattr(os, separator)}sub-01"
        with pytest.raises(ToolError, match=f"^{_refusal(subject, 'it contains a path separator')}$"):
            await stages.purge_subject(derivatives=str(tmp_path / "deriv"), subject=subject)
        assert launches == []

    async def test_valid_subject_still_launches(self, pipeline_dir, tmp_path, launches):
        """REQ-PATTERNS-BDRABCZUK-EC6A1F0884D9: a plain subject name passes the guard and launches one job."""
        await stages.purge_subject(derivatives=str(tmp_path / "deriv"), subject="sub-01")
        assert len(launches) == 1
