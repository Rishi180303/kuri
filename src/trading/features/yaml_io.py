"""Generate `configs/features.yaml` from code, and diff it for validation.

Code is the source of truth. Each feature module exposes `get_meta()` that
returns `list[FeatureMeta]`. The pipeline aggregates them via `all_metas()`.
This module renders that list to YAML and offers a deterministic diff.

`kuri features validate-yaml` calls `validate_features_yaml_in_sync()` and
exits non-zero if the on-disk YAML drifts from code. Hook into pre-commit
to keep them aligned.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from trading.features.config import FeatureConfig
from trading.features.pipeline import all_metas


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_yaml_path() -> Path:
    return _project_root() / "configs" / "features.yaml"


def render_features_yaml(cfg: FeatureConfig | None = None) -> str:
    cfg = cfg or FeatureConfig()
    metas = all_metas(cfg)

    payload: dict[str, Any] = {
        "feature_set_version": cfg.feature_set_version,
        "n_features": len(metas),
        "features": [m.to_yaml_dict() for m in metas],
    }
    return yaml.safe_dump(payload, sort_keys=False, default_flow_style=False)


def write_features_yaml(
    path: Path | None = None,
    cfg: FeatureConfig | None = None,
) -> Path:
    target = path or default_yaml_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_features_yaml(cfg), encoding="utf-8")
    return target


def diff_features_yaml(
    path: Path | None = None,
    cfg: FeatureConfig | None = None,
) -> str | None:
    """Return None if YAML matches code, else a unified-style diff string."""
    target = path or default_yaml_path()
    expected = render_features_yaml(cfg)
    if not target.exists():
        return f"{target} does not exist; run `kuri features write-yaml`."
    actual = target.read_text(encoding="utf-8")
    if actual == expected:
        return None
    # Produce a minimal diff (line-based)
    import difflib

    diff = difflib.unified_diff(
        actual.splitlines(keepends=True),
        expected.splitlines(keepends=True),
        fromfile=str(target),
        tofile="<code-derived>",
        lineterm="",
    )
    return "".join(diff) or "differs (whitespace only)"


def validate_features_yaml_in_sync(
    path: Path | None = None,
    cfg: FeatureConfig | None = None,
) -> bool:
    return diff_features_yaml(path=path, cfg=cfg) is None
