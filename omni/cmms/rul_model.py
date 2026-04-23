"""Remaining Useful Life (RUL) Model — Weibull AFT with numpy.

Implements a parametric Accelerated Failure Time (AFT) model with a Weibull
baseline. Chosen over Cox PH because:
  - We care about the *magnitude* of RUL (days to failure), not just ordering.
  - Weibull AFT gives closed-form mean/median survival time per segment.
  - Pure numpy — no external survival library needed.

Model
-----
    log T_i = μ + β^T x_i + σ ε_i,   ε_i ~ Gumbel(0, 1)

This is equivalent to a Weibull proportional hazards model with:
    shape k = 1/σ,   scale λ_i = exp(μ + β^T x_i)

MLE via L-BFGS-B on the log-likelihood with right-censoring support.

Features (x_i)
---------------
    pipe_age_years      — years since installation
    repair_count        — cumulative prior repairs on this segment
    pressure_bar        — operating pressure at last reading
    is_cast_iron        — material indicator (cast iron ages faster)
    is_branched         — topology indicator (branched = higher stress)
    leak_detections_30d — number of acoustic detections in last 30 days

Output
------
    rul_days     — median days to next failure
    rul_lower_80 — 10th percentile (pessimistic)
    rul_upper_80 — 90th percentile (optimistic)
    survival_30d — probability of surviving the next 30 days
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np

log = logging.getLogger("rul")

# ─────────────────────────── Data structures ──────────────────────────────────

@dataclass
class PipeObservation:
    """One row in the training corpus — a pipe segment's history."""
    segment_id: str
    observed_days: float          # time from last repair / install to event or censor
    failed: bool                  # True = failure occurred; False = right-censored
    pipe_age_years: float
    repair_count: int
    pressure_bar: float
    is_cast_iron: bool
    is_branched: bool
    leak_detections_30d: int


@dataclass
class RULPrediction:
    segment_id: str
    predicted_at: datetime
    rul_days: float               # median survival time
    rul_lower_80: float           # 10th-percentile (pessimistic)
    rul_upper_80: float           # 90th-percentile (optimistic)
    survival_30d: float           # P(survive next 30 days)
    risk_tier: str                # "LOW" | "MEDIUM" | "HIGH" | "CRITICAL"
    recommendation: str


# ─────────────────────────── Feature engineering ──────────────────────────────

_FEATURE_NAMES = [
    "pipe_age_years",
    "repair_count",
    "pressure_bar",
    "is_cast_iron",
    "is_branched",
    "leak_detections_30d",
]

def _featurize(obs: PipeObservation) -> np.ndarray:
    return np.array([
        obs.pipe_age_years,
        float(obs.repair_count),
        obs.pressure_bar,
        float(obs.is_cast_iron),
        float(obs.is_branched),
        float(obs.leak_detections_30d),
    ], dtype=np.float64)


def _design_matrix(observations: list[PipeObservation]) -> np.ndarray:
    """Build (n, p+1) matrix with intercept column prepended."""
    X = np.stack([_featurize(o) for o in observations])
    ones = np.ones((len(X), 1))
    return np.hstack([ones, X])  # shape (n, 7)


# ─────────────────────────── Weibull AFT log-likelihood ───────────────────────

def _neg_log_likelihood(params: np.ndarray, X: np.ndarray, T: np.ndarray, E: np.ndarray) -> float:
    """
    Negative log-likelihood for the Weibull AFT model.

    params : [log_sigma, intercept, β_1 ... β_p]
    X      : (n, p+1) design matrix (with intercept column)
    T      : (n,) observed times (> 0)
    E      : (n,) event indicator (1 = failure, 0 = censored)
    """
    log_sigma = params[0]
    sigma = np.exp(log_sigma)            # shape parameter σ > 0
    beta = params[1:]                    # intercept + covariate weights

    mu = X @ beta                        # linear predictor (n,)
    # Standardised residuals: w_i = (log T_i - μ_i) / σ
    w = (np.log(T) - mu) / sigma

    # Log-likelihood contributions
    # Uncensored: log f(t) = log k - log t + (k-1) log t/λ - (t/λ)^k  →  Gumbel form
    ll_event = -log_sigma + w - np.exp(w)     # log f(w) for Gumbel extreme-value
    # Censored: log S(t) = -exp(w)
    ll_censor = -np.exp(w)

    ll = np.where(E == 1, ll_event, ll_censor)
    return -np.sum(ll)


def _neg_log_likelihood_grad(params: np.ndarray, X: np.ndarray, T: np.ndarray, E: np.ndarray) -> np.ndarray:
    """Analytic gradient of the negative log-likelihood."""
    log_sigma = params[0]
    sigma = np.exp(log_sigma)
    beta = params[1:]

    mu = X @ beta
    w = (np.log(T) - mu) / sigma
    exp_w = np.exp(w)

    # ∂ll/∂w_i for uncensored: 1 - exp(w_i);  for censored: -exp(w_i)
    dll_dw = np.where(E == 1, 1.0 - exp_w, -exp_w)

    # Chain through w = (log T - Xβ) / σ
    # ∂w/∂σ = -w/σ  →  ∂ll/∂log_σ = ∂ll/∂w * ∂w/∂log_σ + E * (-1)
    # (the -1 comes from the explicit -log σ term in event LL)
    dll_dlogsigma = np.sum(dll_dw * (-w) + np.where(E == 1, -1.0, 0.0))
    # ∂w/∂β = -X / σ
    dll_dbeta = (dll_dw / sigma) @ (-X)

    grad = np.concatenate([[dll_dlogsigma], dll_dbeta])
    return -grad  # negative because we minimise


# ─────────────────────────── Optimiser (L-BFGS-B via numpy) ───────────────────

def _lbfgsb(f, g, x0: np.ndarray, max_iter: int = 500, tol: float = 1e-7) -> np.ndarray:
    """Minimal L-BFGS-B using numpy only (no scipy needed).

    Pure Python implementation sufficient for p ≤ 20 parameters.
    Uses the two-loop recursion with m=10 pairs stored.
    """
    m = 10
    x = x0.copy()
    s_hist, y_hist = [], []

    for _ in range(max_iter):
        grad = g(x)
        if np.linalg.norm(grad) < tol:
            break

        q = grad.copy()
        alphas = []
        for s, y in zip(reversed(s_hist), reversed(y_hist), strict=False):
            rho = 1.0 / (y @ s + 1e-12)
            a = rho * (s @ q)
            q -= a * y
            alphas.append((rho, a, s, y))

        r = q.copy()
        if s_hist:
            s_last, y_last = s_hist[-1], y_hist[-1]
            gamma = (s_last @ y_last) / (y_last @ y_last + 1e-12)
            r *= gamma

        for (rho, a, s, y) in reversed(alphas):
            b = rho * (y @ r)
            r += s * (a - b)

        direction = -r

        # Armijo line-search
        step = 1.0
        f0 = f(x)
        slope = grad @ direction
        for _ in range(20):
            x_new = x + step * direction
            if f(x_new) <= f0 + 1e-4 * step * slope:
                break
            step *= 0.5
        else:
            step = 1e-5

        s_vec = step * direction
        x_new = x + s_vec
        y_vec = g(x_new) - grad

        if s_hist and len(s_hist) >= m:
            s_hist.pop(0)
            y_hist.pop(0)
        if y_vec @ s_vec > 1e-10:
            s_hist.append(s_vec)
            y_hist.append(y_vec)

        x = x_new

    return x


# ─────────────────────────── Model class ──────────────────────────────────────

class WeibullAFT:
    """Fitted Weibull AFT model. Fit once, predict many."""

    def __init__(self):
        self._params: np.ndarray | None = None
        self._sigma: float = 1.0
        self._beta: np.ndarray | None = None
        self._feature_mean: np.ndarray | None = None
        self._feature_std: np.ndarray | None = None
        self.n_obs: int = 0
        self.log_likelihood: float = float("nan")

    @property
    def is_fitted(self) -> bool:
        return self._params is not None

    def fit(self, observations: list[PipeObservation]) -> WeibullAFT:
        if len(observations) < 10:
            raise ValueError(f"Need ≥ 10 observations to fit; got {len(observations)}")

        X_raw = np.stack([_featurize(o) for o in observations])
        T = np.array([o.observed_days for o in observations], dtype=np.float64)
        E = np.array([float(o.failed) for o in observations], dtype=np.float64)

        # Standardise features (columns 1..p, not the intercept we'll add)
        self._feature_mean = X_raw.mean(axis=0)
        self._feature_std = X_raw.std(axis=0) + 1e-8
        X_std = (X_raw - self._feature_mean) / self._feature_std
        X = np.hstack([np.ones((len(X_std), 1)), X_std])

        # Clip T > 0
        T = np.clip(T, 1e-3, None)

        n, p = X.shape
        # Initialise: log_sigma=0, intercept=mean(log T), betas=0
        x0 = np.zeros(p + 1)
        x0[0] = 0.0                            # log_sigma → σ=1
        x0[1] = np.mean(np.log(T))             # intercept

        def f(p):
            return _neg_log_likelihood(p, X, T, E)
        def g(p):
            return _neg_log_likelihood_grad(p, X, T, E)

        self._params = _lbfgsb(f, g, x0)
        self._sigma = np.exp(self._params[0])
        self._beta = self._params[1:]
        self.n_obs = n
        self.log_likelihood = -f(self._params)

        log.info(
            "WeibullAFT fitted: n=%d σ=%.3f ll=%.2f",
            n, self._sigma, self.log_likelihood,
        )
        return self

    def _predict_scale(self, x_raw: np.ndarray) -> float:
        """λ_i = exp(μ_i) for a single raw feature vector."""
        x_std = (x_raw - self._feature_mean) / self._feature_std
        x = np.concatenate([[1.0], x_std])
        mu = float(x @ self._beta)
        return math.exp(mu)

    def predict(self, obs: PipeObservation) -> RULPrediction:
        if not self.is_fitted:
            raise RuntimeError("Model not fitted — call fit() first")

        x_raw = _featurize(obs)
        lam = self._predict_scale(x_raw)   # scale parameter λ
        k = 1.0 / self._sigma              # shape parameter k

        # Weibull quantile: S(t) = exp(-(t/λ)^k)  →  t_q = λ * (-log q)^(1/k)
        def quantile(q: float) -> float:
            return lam * ((-math.log(q + 1e-12)) ** (1.0 / k))

        rul_median   = quantile(0.50)
        rul_lower_80 = quantile(0.90)    # 10th percentile of survival
        rul_upper_80 = quantile(0.10)    # 90th percentile

        # P(T > 30) = exp(-(30/λ)^k)
        survival_30d = math.exp(-(30.0 / lam) ** k)

        # Risk tier
        if rul_median < 30:
            tier, rec = "CRITICAL", "Schedule immediate inspection and preventive repair."
        elif rul_median < 90:
            tier, rec = "HIGH", "Schedule repair within 2 weeks. Monitor daily."
        elif rul_median < 180:
            tier, rec = "MEDIUM", "Include in next quarterly maintenance window."
        else:
            tier, rec = "LOW", "No immediate action required. Re-evaluate in 90 days."

        return RULPrediction(
            segment_id=obs.segment_id,
            predicted_at=datetime.now(UTC),
            rul_days=round(max(0.0, rul_median), 1),
            rul_lower_80=round(max(0.0, rul_lower_80), 1),
            rul_upper_80=round(rul_upper_80, 1),
            survival_30d=round(survival_30d, 4),
            risk_tier=tier,
            recommendation=rec,
        )


# ─────────────────────────── Synthetic training corpus ────────────────────────

def generate_synthetic_corpus(n: int = 400, seed: int = 42) -> list[PipeObservation]:
    """Generate realistic pipe-failure history for Beirut's water network.

    Ground-truth Weibull parameters chosen to reflect:
      - Cast iron pipes fail ~2× faster than PVC/steel
      - Each prior repair halves expected lifetime (cumulative damage)
      - Higher operating pressure accelerates failure
      - ~25% right-censoring (pipes still active at observation window end)
    """
    rng = np.random.default_rng(seed)
    observations = []

    # True AFT parameters (log-scale).
    # Intercept chosen so a median "average" Beirut pipe lasts ~3 years
    # between failures: exp(7.9) ≈ 2700 days.  Covariates shift from there.
    TRUE_INTERCEPT  =  7.9    # exp(7.9) ≈ 2700 days baseline
    TRUE_AGE        = -0.055  # older → shorter lifetime
    TRUE_REPAIRS    = -0.55   # each repair → ~43% reduction in lifetime
    TRUE_PRESSURE   = -0.10   # higher pressure → faster failure
    TRUE_CAST_IRON  = -0.70   # cast iron ages significantly worse than PVC
    TRUE_BRANCHED   = -0.25   # branched topology → higher stress
    TRUE_DETECTIONS = -0.40   # more recent detections → already degraded
    TRUE_SIGMA      =  0.55   # Weibull shape k ≈ 1/0.55 ≈ 1.82

    PIPE_IDS = [f"P-{i:04d}" for i in range(n)]

    for _i, pid in enumerate(PIPE_IDS):
        age     = rng.uniform(2, 45)
        repairs = int(rng.poisson(1.5))
        pressure = rng.uniform(2.5, 8.0)
        cast_iron = bool(rng.random() < 0.35)
        branched  = bool(rng.random() < 0.50)
        detections = int(rng.poisson(1.0))

        # True log scale
        mu = (TRUE_INTERCEPT
              + TRUE_AGE        * age
              + TRUE_REPAIRS    * repairs
              + TRUE_PRESSURE   * pressure
              + TRUE_CAST_IRON  * float(cast_iron)
              + TRUE_BRANCHED   * float(branched)
              + TRUE_DETECTIONS * detections)

        # Sample from Weibull(shape=k, scale=exp(mu))
        lam_true = math.exp(mu)
        k_true = 1.0 / TRUE_SIGMA
        # Inverse CDF: t = λ * (-log U)^(1/k)
        u = rng.uniform(0, 1)
        t_failure = lam_true * ((-math.log(u + 1e-12)) ** (1.0 / k_true))

        # Right-censoring at observation window (730–1800 days) for ~25%
        censor_time = rng.uniform(730, 1800) if rng.random() < 0.25 else float("inf")
        observed = min(t_failure, censor_time)
        failed = t_failure <= censor_time

        observations.append(PipeObservation(
            segment_id=pid,
            observed_days=max(1.0, observed),
            failed=failed,
            pipe_age_years=age,
            repair_count=repairs,
            pressure_bar=pressure,
            is_cast_iron=cast_iron,
            is_branched=branched,
            leak_detections_30d=detections,
        ))

    return observations


# ─────────────────────────── Module-level singleton ───────────────────────────

_model: WeibullAFT | None = None


def get_model() -> WeibullAFT:
    """Lazy-load: train on synthetic corpus if not yet fitted."""
    global _model
    if _model is None or not _model.is_fitted:
        log.info("Training RUL model on synthetic corpus …")
        corpus = generate_synthetic_corpus(n=600)
        _model = WeibullAFT().fit(corpus)
    return _model


def predict_rul(
    segment_id: str,
    pipe_age_years: float,
    repair_count: int,
    pressure_bar: float,
    is_cast_iron: bool,
    is_branched: bool,
    leak_detections_30d: int,
) -> RULPrediction:
    """Convenience wrapper — the CMMS calls this after every work order."""
    obs = PipeObservation(
        segment_id=segment_id,
        observed_days=0.0,   # not used for prediction
        failed=False,
        pipe_age_years=pipe_age_years,
        repair_count=repair_count,
        pressure_bar=pressure_bar,
        is_cast_iron=is_cast_iron,
        is_branched=is_branched,
        leak_detections_30d=leak_detections_30d,
    )
    return get_model().predict(obs)
