"""The MCP default Mask version follows the pipeline default (TASK-MODELVER-BDRABCZUK-2D826D2F0C5C).

User decision 2026-11-24 (Q1, Q6): with masks and model_version unset the MCP server keeps defaulting the
run's Mask version, now to canonical_corrected_26_1, champollion_pipeline's
``derivatives_layout.DEFAULT_MASKS_VERSION`` since its switch (pipeline WIP 8d1a670). The pipeline value
is pinned as a literal: the sibling pipeline checkout need not sit on WIP.

- REQ-MODELVER-BDRABCZUK-26B39708A523: DEFAULT_MASKS_VERSION equals the pipeline's.
- REQ-MODELVER-BDRABCZUK-533E345C49DB: start_pipeline's description names it as the fallback V.
- REQ-MODELVER-BDRABCZUK-CD880048509C: start_config's description names it as the script's --masks default.
"""

from __future__ import annotations

import inspect

import pytest

from champollion_sulcal_mcp.tools import pipeline, stages

pytestmark = pytest.mark.unit

PIPELINE_DEFAULT_MASKS_VERSION = "canonical_corrected_26_1"  # champollion_pipeline derivatives_layout, WIP 8d1a670
OLD_DEFAULT = "canonical_25"


def test_default_masks_version_is_pipeline_default():
    """REQ-MODELVER-BDRABCZUK-26B39708A523: DEFAULT_MASKS_VERSION is canonical_corrected_26_1."""
    assert pipeline.DEFAULT_MASKS_VERSION == PIPELINE_DEFAULT_MASKS_VERSION


def test_start_pipeline_description_names_corrected_default():
    """REQ-MODELVER-BDRABCZUK-533E345C49DB: the description gives canonical_corrected_26_1 as the fallback V."""
    description = inspect.getdoc(pipeline.start_pipeline) or ""
    assert f"else ``{PIPELINE_DEFAULT_MASKS_VERSION}``" in description, description
    assert OLD_DEFAULT not in description, description


def test_start_config_description_names_corrected_default():
    """REQ-MODELVER-BDRABCZUK-CD880048509C: the description gives canonical_corrected_26_1 as the --masks default."""
    description = inspect.getdoc(stages.start_config) or ""
    assert f"script default {PIPELINE_DEFAULT_MASKS_VERSION}" in description, description
    assert OLD_DEFAULT not in description, description
