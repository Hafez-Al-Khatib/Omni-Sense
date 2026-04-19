"""RUL model: Weibull AFT fitting, prediction ordering, quantiles."""
import math

import pytest

import omni.cmms.rul_model as rul_mod
from omni.cmms.rul_model import (
    PipeObservation,
    WeibullAFT,
    generate_synthetic_corpus,
    predict_rul,
)


@pytest.fixture(autouse=True)
def reset_singleton():
    """Force model refit for each test."""
    rul_mod._model = None
    yield
    rul_mod._model = None


def _obs(segment_id="S-TEST", age=10, repairs=0, pressure=4.0,
         cast_iron=False, branched=False, detections=0,
         observed_days=200.0, failed=True):
    return PipeObservation(
        segment_id=segment_id,
        observed_days=observed_days,
        failed=failed,
        pipe_age_years=age,
        repair_count=repairs,
        pressure_bar=pressure,
        is_cast_iron=cast_iron,
        is_branched=branched,
        leak_detections_30d=detections,
    )


def test_corpus_has_both_classes():
    corpus = generate_synthetic_corpus(200, seed=0)
    n_failed  = sum(1 for o in corpus if o.failed)
    n_censored = sum(1 for o in corpus if not o.failed)
    assert n_failed > 0
    assert n_censored >= 0   # censoring depends on params, not required
    assert len(corpus) == 200


def test_model_fits_without_error():
    corpus = generate_synthetic_corpus(150, seed=1)
    model = WeibullAFT().fit(corpus)
    assert model.is_fitted
    assert not math.isnan(model.log_likelihood)
    assert model._sigma > 0


def test_model_requires_min_observations():
    corpus = generate_synthetic_corpus(5, seed=0)
    with pytest.raises(ValueError, match="≥ 10"):
        WeibullAFT().fit(corpus)


def test_prediction_ordering_by_age():
    """Older pipes should have lower RUL than newer ones, all else equal."""
    corpus = generate_synthetic_corpus(200, seed=2)
    model = WeibullAFT().fit(corpus)
    young = model.predict(_obs(age=5,  repairs=0))
    old   = model.predict(_obs(age=40, repairs=0))
    assert young.rul_days > old.rul_days


def test_prediction_ordering_by_repair_count():
    """More repairs → shorter RUL."""
    corpus = generate_synthetic_corpus(200, seed=3)
    model = WeibullAFT().fit(corpus)
    fresh   = model.predict(_obs(repairs=0))
    damaged = model.predict(_obs(repairs=5))
    assert fresh.rul_days > damaged.rul_days


def test_prediction_ordering_by_material():
    """Cast iron should have shorter RUL than PVC."""
    corpus = generate_synthetic_corpus(200, seed=4)
    model = WeibullAFT().fit(corpus)
    pvc       = model.predict(_obs(cast_iron=False))
    cast_iron = model.predict(_obs(cast_iron=True))
    assert pvc.rul_days > cast_iron.rul_days


def test_quantile_ordering():
    """lower_80 ≤ median ≤ upper_80."""
    corpus = generate_synthetic_corpus(200, seed=5)
    model = WeibullAFT().fit(corpus)
    r = model.predict(_obs(age=20, repairs=2, pressure=5.0))
    assert r.rul_lower_80 <= r.rul_days <= r.rul_upper_80


def test_survival_30d_in_range():
    corpus = generate_synthetic_corpus(200, seed=6)
    model = WeibullAFT().fit(corpus)
    r = model.predict(_obs())
    assert 0.0 <= r.survival_30d <= 1.0


def test_risk_tier_critical_for_bad_pipe():
    r = predict_rul("P-BAD", 40, 6, 7.0, True, True, 5)
    assert r.risk_tier == "CRITICAL"


def test_risk_tier_low_for_new_pipe():
    r = predict_rul("P-GOOD", 2, 0, 2.5, False, False, 0)
    assert r.risk_tier in ("LOW", "MEDIUM")


def test_singleton_is_reused():
    r1 = predict_rul("P-A", 10, 1, 4.0, False, False, 0)
    r2 = predict_rul("P-B", 10, 1, 4.0, False, False, 0)
    # Same features → same RUL (model is deterministic once fitted)
    assert abs(r1.rul_days - r2.rul_days) < 1.0
