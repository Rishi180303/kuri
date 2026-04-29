"""Tests for trading.training.data.load_training_data."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest

from trading.config import TickerEntry, UniverseConfig
from trading.features.store import FeatureStore
from trading.labels.store import LabelStore
from trading.training.data import load_training_data


def _seed_features_and_labels(
    root: Path,
    universe: UniverseConfig,
    n_days: int = 30,
) -> None:
    """Synthesise minimal feature + label parquets at `root` so the loader
    has something to join. Per-ticker has a couple feature cols + ret_5d
    as a label-ready surrogate; regime is a single date-keyed frame.
    """
    fstore = FeatureStore(root / "features", version=1)
    lstore = LabelStore(root / "labels", version=1)
    base = date(2024, 1, 1)

    # Per-ticker features
    pt_rows = []
    for sym in universe.symbols:
        for i in range(n_days):
            pt_rows.append(
                {
                    "date": base + timedelta(days=i),
                    "ticker": sym,
                    "ret_1d": 0.001 * (i % 5),
                    "rsi_14": 50.0 + (i % 10),
                }
            )
    fstore.save_per_ticker(pl.DataFrame(pt_rows))

    # Regime (one row per date)
    regime_rows = [
        {
            "date": base + timedelta(days=i),
            "vix_level": 15.0 + 0.1 * i,
            "nifty_above_sma_200": 1,
        }
        for i in range(n_days)
    ]
    fstore.save_regime(pl.DataFrame(regime_rows))

    # Labels (mirrors compute_labels output: last 5 rows null per ticker)
    lbl_rows = []
    for sym in universe.symbols:
        for i in range(n_days):
            cls_val = None if i >= n_days - 5 else (1 if (i + hash(sym)) % 2 == 0 else 0)
            reg_val = None if i >= n_days - 5 else 0.001 * (i % 5)
            lbl_rows.append(
                {
                    "date": base + timedelta(days=i),
                    "ticker": sym,
                    "outperforms_universe_median_5d": cls_val,
                    "forward_ret_5d_demeaned": reg_val,
                }
            )
    lstore.save_per_ticker(pl.DataFrame(lbl_rows))


@pytest.fixture
def small_universe(monkeypatch: pytest.MonkeyPatch) -> UniverseConfig:
    """Override the cached universe loader so the test's mini universe
    is what `load_training_data` sees for sector mapping.
    """
    universe = UniverseConfig(
        as_of=date(2024, 1, 1),
        index="MINI",
        tickers=[
            TickerEntry(symbol="AAA", sector="IT"),
            TickerEntry(symbol="BBB", sector="IT"),
            TickerEntry(symbol="CCC", sector="Banks"),
        ],
    )
    import trading.training.data as data_mod

    monkeypatch.setattr(data_mod, "get_universe_config", lambda: universe)
    return universe


def test_join_produces_expected_shape(tmp_path: Path, small_universe: UniverseConfig) -> None:
    _seed_features_and_labels(tmp_path, small_universe, n_days=30)
    df = load_training_data(
        horizons=(5,),
        data_dir=tmp_path,
    )
    # 3 tickers * (30 - 5 last-h null rows) = 75 rows after dropping label nulls
    assert df.height == 3 * (30 - 5)
    assert "sector" in df.columns
    assert "vix_level" in df.columns
    assert "rsi_14" in df.columns
    assert "outperforms_universe_median_5d" in df.columns
    assert df["outperforms_universe_median_5d"].null_count() == 0


def test_drop_label_nulls_false_keeps_all_rows(
    tmp_path: Path, small_universe: UniverseConfig
) -> None:
    _seed_features_and_labels(tmp_path, small_universe, n_days=30)
    df = load_training_data(
        horizons=(5,),
        data_dir=tmp_path,
        drop_label_nulls=False,
    )
    assert df.height == 3 * 30  # full row count, including the 5 trailing nulls each


def test_date_range_filter(tmp_path: Path, small_universe: UniverseConfig) -> None:
    _seed_features_and_labels(tmp_path, small_universe, n_days=30)
    df = load_training_data(
        start=date(2024, 1, 5),
        end=date(2024, 1, 10),
        horizons=(5,),
        data_dir=tmp_path,
    )
    assert df["date"].min() >= date(2024, 1, 5)
    assert df["date"].max() <= date(2024, 1, 10)


def test_sector_attached_correctly(tmp_path: Path, small_universe: UniverseConfig) -> None:
    _seed_features_and_labels(tmp_path, small_universe, n_days=30)
    df = load_training_data(horizons=(5,), data_dir=tmp_path)
    pairs = df.select("ticker", "sector").unique().sort("ticker").to_dicts()
    assert pairs == [
        {"ticker": "AAA", "sector": "IT"},
        {"ticker": "BBB", "sector": "IT"},
        {"ticker": "CCC", "sector": "Banks"},
    ]


def test_rejects_empty_horizons(tmp_path: Path, small_universe: UniverseConfig) -> None:
    _seed_features_and_labels(tmp_path, small_universe, n_days=10)
    with pytest.raises(ValueError):
        load_training_data(horizons=(), data_dir=tmp_path)


def test_raises_when_features_missing(tmp_path: Path, small_universe: UniverseConfig) -> None:
    # No seeding — features dir empty
    with pytest.raises(RuntimeError, match="No per-ticker features"):
        load_training_data(horizons=(5,), data_dir=tmp_path)


def test_raises_when_labels_missing(tmp_path: Path, small_universe: UniverseConfig) -> None:
    # Seed features only
    fstore = FeatureStore(tmp_path / "features", version=1)
    base = date(2024, 1, 1)
    rows = []
    for sym in small_universe.symbols:
        for i in range(10):
            rows.append({"date": base + timedelta(days=i), "ticker": sym, "ret_1d": 0.0})
    fstore.save_per_ticker(pl.DataFrame(rows))
    fstore.save_regime(
        pl.DataFrame(
            {"date": [base + timedelta(days=i) for i in range(10)], "vix_level": [15.0] * 10}
        )
    )

    with pytest.raises(RuntimeError, match="No labels"):
        load_training_data(horizons=(5,), data_dir=tmp_path)
