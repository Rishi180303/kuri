"""Tests for features.yaml generator and drift detector."""

from __future__ import annotations

from pathlib import Path

from trading.features.config import FeatureConfig
from trading.features.yaml_io import (
    default_yaml_path,
    diff_features_yaml,
    render_features_yaml,
    validate_features_yaml_in_sync,
    write_features_yaml,
)


def test_default_yaml_in_sync_with_code() -> None:
    """The shipped configs/features.yaml must always match code."""
    assert (
        validate_features_yaml_in_sync()
    ), "configs/features.yaml drifted from code. Run `kuri features write-yaml` to regenerate."


def test_render_includes_version_and_count() -> None:
    cfg = FeatureConfig(feature_set_version=42)
    text = render_features_yaml(cfg)
    assert "feature_set_version: 42" in text
    assert "n_features:" in text


def test_write_then_diff_returns_none(tmp_path: Path) -> None:
    p = tmp_path / "features.yaml"
    write_features_yaml(path=p)
    assert diff_features_yaml(path=p) is None


def test_diff_when_yaml_out_of_date(tmp_path: Path) -> None:
    p = tmp_path / "features.yaml"
    p.write_text("feature_set_version: 999\nfeatures: []\n", encoding="utf-8")
    diff = diff_features_yaml(path=p)
    assert diff is not None
    assert "feature_set_version" in diff


def test_diff_when_yaml_missing(tmp_path: Path) -> None:
    diff = diff_features_yaml(path=tmp_path / "nope.yaml")
    assert diff is not None
    assert "does not exist" in diff


def test_default_yaml_path_points_at_configs() -> None:
    p = default_yaml_path()
    assert p.name == "features.yaml"
    assert p.parent.name == "configs"
