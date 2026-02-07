
"""
data_models.py

Refactored data model classes for generating synthetic datasets (X, y) and
unlabeled feature distributions (X) for empirical risk minimization (ERM)
experiments.

Design (3 layers)
-----------------
Layer 1: Unlabeled, one-class feature models
  - BaseFeatureModel (X only; exposes mean/covariance)
  - GaussianModel
  - UnifModel
  - BetaModel          (supports moment-matching to a target mean/cov)
  - BernoulliModel     (supports moment-matching to a target mean/cov)
  - MultimodalModel / BimodalModel (mixture of feature models; optional moment-matching)

Layer 2: Labeled multi-class models (y is the class label)
  - MultiClassModel: mixture of K feature models, returns y in y_values.

Layer 3: Teacher/student models (y produced by a teacher, classes inherited from x_model)
  - TeacherStudentModel: wraps an x_model (feature model or a data model). The classes of the
    teacher model are exactly the classes of x_model (NOT the label cardinality of y_model).

Additional models
-----------------
  - LinearFactorMixedModel: user-provided BaseDataModel with a mixed signal/noise construction.
  - MNISTGroupedDataModel: grouped-digit MNIST sampler with a feature map.
      IMPORTANT: no "representation" switch. Raw pixels correspond to W=None + identity activation.

Pruning / super-label models (eps, y)
-------------------------------------
  - unit, make_aligned_vector
  - OracleMarginPruner
  - RealGeneratedPrunedModel: mixture of real and generated models returning u=(eps, y).

Notes on moment matching for constrained marginals
--------------------------------------------------
Some distributions (Beta, Bernoulli) cannot realize arbitrary (mu, C) while preserving their
native marginals and independence structure. The moment-matching implementations below follow a
pragmatic convention:

  - If you request a diagonal covariance and the requested moments are feasible, the model uses
    an i.i.d. construction that preserves the intended marginal family (e.g. true Beta on [0,1]).
  - If you request a non-diagonal covariance, the model uses an affine mixing step to match the
    full covariance exactly. This generally destroys the original marginal family (coordinates
    are no longer independent, and marginals are not exactly Beta/Bernoulli).

This mirrors the previous "uniform_iid vs uniform_affine" logic and keeps theoretical moment
inputs consistent across your experiments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional, Protocol, Sequence, Tuple, Union, cast
from abc import ABC, abstractmethod

import numpy as np

Array = np.ndarray
Rng = np.random.Generator

# ======================================================================================
# Small utilities (single-sourced)
# ======================================================================================

def rng_or_default(rng: Optional[Rng]) -> Rng:
    return np.random.default_rng() if rng is None else rng


def as_1d(x: Union[Array, Sequence[float]], p: int, name: str) -> Array:
    x = np.asarray(x, dtype=float).reshape(-1)
    if x.shape != (p,):
        raise ValueError(f"{name} must have shape (p,) with p={p}, got {x.shape}")
    return x


def as_2d(x: Union[Array, Sequence[Sequence[float]]], p: int, name: str) -> Array:
    x = np.asarray(x, dtype=float)
    if x.shape != (p, p):
        raise ValueError(f"{name} must have shape (p,p)=({p},{p}), got {x.shape}")
    return x


def symmetrize(M: Array) -> Array:
    M = np.asarray(M, dtype=float)
    return 0.5 * (M + M.T)


def is_diagonal(M: Array, tol: float = 1e-12) -> bool:
    M = np.asarray(M, dtype=float)
    return np.allclose(M, np.diag(np.diag(M)), atol=tol, rtol=0.0)


def safe_cholesky(C: Array, *, jitter: float = 1e-12, max_tries: int = 8) -> Array:
    """
    Cholesky factorization with progressive diagonal jitter for numerical stability.
    Raises if matrix is not numerically SPD.
    """
    C = symmetrize(C)
    p = C.shape[0]
    I = np.eye(p, dtype=float)
    for k in range(max_tries):
        try:
            return np.linalg.cholesky(C + (10.0 ** k) * jitter * I)
        except np.linalg.LinAlgError:
            continue
    raise np.linalg.LinAlgError("safe_cholesky failed: matrix not SPD (even with jitter).")


def normalize_probs(gamma: Union[Array, Sequence[float]], *, name: str = "gamma") -> Array:
    g = np.asarray(gamma, dtype=float).reshape(-1)
    if g.ndim != 1 or g.size == 0:
        raise ValueError(f"{name} must be a 1D array with length >= 1.")
    if np.any(g < 0):
        raise ValueError(f"{name} must be nonnegative.")
    s = float(g.sum())
    if not np.isfinite(s) or s <= 0:
        raise ValueError(f"{name} must sum to a positive finite number.")
    return g / s


def exponential_covariance(p: int, rho: float = 0.5) -> Array:
    """
    Exponential-decay Toeplitz covariance:
        C_ij = rho^{|i-j|}
    """
    if not (0.0 < rho < 1.0):
        raise ValueError("rho must be in (0,1).")
    idx = np.arange(p)
    return rho ** np.abs(idx[:, None] - idx[None, :])


def moment_estimates(
    X: Array,
    *,
    reg: float = 0.0,
    kind: Literal["full", "diag"] = "full",
) -> Tuple[Array, Array]:
    """
    Estimate mean and covariance of X (n,p), using population scaling (1/n):

        mu = mean(X)
        C  = E[xx^T] - mu mu^T

    reg adds reg*I for numerical stability.
    """
    X64 = np.asarray(X, dtype=np.float64)
    if X64.ndim != 2:
        raise ValueError(f"X must be 2D (n,p), got shape {X64.shape}")
    n, p = X64.shape
    if n == 0:
        raise ValueError("Cannot estimate moments from empty set.")

    mu = X64.mean(axis=0)
    if kind == "full":
        exx = (X64.T @ X64) / float(n)
        C = exx - np.outer(mu, mu)
        C = symmetrize(C)
        if reg > 0:
            C = C + reg * np.eye(p)
    elif kind == "diag":
        ex2 = (X64 * X64).mean(axis=0)
        var = ex2 - mu * mu
        var = np.maximum(var, 0.0)
        if reg > 0:
            var = var + reg
        C = np.diag(var)
    else:
        raise ValueError(f"Unknown kind={kind!r}")

    out_dtype = np.asarray(X).dtype
    return mu.astype(out_dtype, copy=False), C.astype(out_dtype, copy=False)


def sign_pm1(t: Array) -> Array:
    """
    Map real values to {+1, -1}, with 0 mapped to +1 by convention.
    """
    t = np.asarray(t)
    s = np.sign(t)
    return np.where(s == 0, 1.0, s).astype(float, copy=False)


def unit(v: Array) -> Array:
    v = np.asarray(v, dtype=float).reshape(-1)
    n = float(np.linalg.norm(v))
    return v / (n + 1e-12)


def make_aligned_vector(w_star: Array, rho: float, rng: Rng) -> Array:
    """
    Return a unit vector w_g with cosine alignment rho with w_star.
    """
    w_star = unit(w_star)
    p = w_star.size
    v = rng.standard_normal(size=p)
    v = v - (v @ w_star) * w_star
    v = unit(v)
    rho = float(np.clip(rho, -1.0, 1.0))
    w_g = rho * w_star + np.sqrt(max(1.0 - rho**2, 0.0)) * v
    return unit(w_g)


def _affine_matrix_from_covariances(C_base: Array, C_target: Array) -> Array:
    """
    Return A such that A C_base A^T = C_target (up to numerical precision).

    Uses Cholesky factors:
        C_base   = Lb Lb^T
        C_target = Lt Lt^T
        A = Lt Lb^{-1}

    Implemented without forming an explicit inverse.
    """
    C_base = symmetrize(np.asarray(C_base, dtype=float))
    C_target = symmetrize(np.asarray(C_target, dtype=float))
    Lb = safe_cholesky(C_base)
    Lt = safe_cholesky(C_target)
    # A^T solves: Lb (A^T) = Lt^T
    A_T = np.linalg.solve(Lb, Lt.T)
    return A_T.T


# --------------------------------------------------------------------------------------
# Compatibility aliases (kept minimal but useful)
# --------------------------------------------------------------------------------------
_as_1d = as_1d
_as_2d = as_2d
_symmetrize = symmetrize
_is_diagonal = is_diagonal
_safe_cholesky = safe_cholesky
_moment_estimates_full = lambda X, reg=0.0: moment_estimates(X, reg=reg, kind="full")
_moment_estimates_diag = lambda X, reg=0.0: moment_estimates(X, reg=reg, kind="diag")

# ======================================================================================
# Feature models (unlabeled, single class): X only
# ======================================================================================

class BaseFeatureModel(ABC):
    """
    Unlabeled one-class feature distribution model: produces X only.

    Contract:
      - p: feature dimension
      - sample(n, rng) -> X (n,p)
      - moments() -> (mu, C)
    """

    p: int

    @abstractmethod
    def sample(self, n: int, rng: Optional[Rng] = None) -> Array:
        ...

    @abstractmethod
    def moments(self) -> Tuple[Array, Array]:
        ...

    def params(self) -> Dict[str, Any]:
        return {"type": self.__class__.__name__, "p": int(self.p)}




# --------------------------------------------------------------------------------------
# Sampling helpers (functional API)
# --------------------------------------------------------------------------------------

def sample_gaussian(n: int, mu: Array, C: Array, rng: Optional[Rng] = None) -> Array:
    """Functional wrapper: draw N(mu, C)."""
    mu = np.asarray(mu, dtype=float).reshape(-1)
    p = int(mu.size)
    C = symmetrize(as_2d(C, p, "C"))
    rng = rng_or_default(rng)
    return rng.multivariate_normal(mean=mu, cov=C, size=int(n))


def sample_uniform_iid_from_mean_var(n: int, mu: Array, var: Array, rng: Optional[Rng] = None) -> Array:
    """
    Functional wrapper: independent Uniform[a_j,b_j] with given mean mu_j and variance var_j.
    """
    mu = np.asarray(mu, dtype=float).reshape(-1)
    var = np.asarray(var, dtype=float).reshape(-1)
    if mu.shape != var.shape:
        raise ValueError("mu and var must have the same shape.")
    if np.any(var < 0):
        raise ValueError("var must be nonnegative.")
    rng = rng_or_default(rng)
    half_width = np.sqrt(3.0 * var)
    low = mu - half_width
    high = mu + half_width
    return rng.uniform(low=low, high=high, size=(int(n), mu.size))


def sample_uniform_affine(n: int, mu: Array, C: Array, rng: Optional[Rng] = None) -> Array:
    """
    Functional wrapper: affine-uniform construction matching full covariance:
        X = mu + U L^T
    where U has iid Uniform(-sqrt(3),sqrt(3)) entries and L L^T = C.
    """
    mu = np.asarray(mu, dtype=float).reshape(-1)
    p = int(mu.size)
    C = symmetrize(as_2d(C, p, "C"))
    rng = rng_or_default(rng)
    U = rng.uniform(low=-np.sqrt(3.0), high=np.sqrt(3.0), size=(int(n), p))
    L = safe_cholesky(C)
    return mu + U @ L.T


def sample_iid_beta(n: int, alpha: Array, beta: Array, rng: Optional[Rng] = None) -> Array:
    """Functional wrapper: iid Beta(alpha_j, beta_j) per coordinate."""
    alpha = np.asarray(alpha, dtype=float).reshape(-1)
    beta = np.asarray(beta, dtype=float).reshape(-1)
    if alpha.shape != beta.shape:
        raise ValueError("alpha and beta must have the same shape.")
    if np.any(alpha <= 0) or np.any(beta <= 0):
        raise ValueError("alpha and beta must be strictly positive.")
    rng = rng_or_default(rng)
    return rng.beta(alpha, beta, size=(int(n), alpha.size))


def sample_iid_bernoulli(
    n: int,
    prob: Array,
    *,
    support: Literal["pm1", "01"] = "pm1",
    rng: Optional[Rng] = None,
) -> Array:
    """Functional wrapper: iid Bernoulli per coordinate (support {±1} or {0,1})."""
    prob = np.asarray(prob, dtype=float).reshape(-1)
    if np.any((prob < 0) | (prob > 1)):
        raise ValueError("prob must be in [0,1].")
    rng = rng_or_default(rng)
    U = rng.random(size=(int(n), prob.size))
    if support == "01":
        return (U < prob).astype(float)
    if support == "pm1":
        return np.where(U < prob, 1.0, -1.0).astype(float)
    raise ValueError("support must be 'pm1' or '01'.")
@dataclass
class GaussianModel(BaseFeatureModel):
    """
    X ~ N(mu, C)
    """
    p: int
    mu: Array
    C: Array

    def __post_init__(self) -> None:
        self.mu = as_1d(self.mu, self.p, "mu")
        self.C = symmetrize(as_2d(self.C, self.p, "C"))

    def sample(self, n: int, rng: Optional[Rng] = None) -> Array:
        rng = rng_or_default(rng)
        n = int(n)
        if n == 0:
            return np.zeros((0, self.p), dtype=float)
        return rng.multivariate_normal(mean=self.mu, cov=self.C, size=n)

    def moments(self) -> Tuple[Array, Array]:
        return self.mu.copy(), self.C.copy()

    def params(self) -> Dict[str, Any]:
        return {"type": "GaussianModel", "p": int(self.p), "mu": self.mu.copy(), "C": self.C.copy()}


@dataclass
class UnifModel(BaseFeatureModel):
    """
    Uniform-ish feature model with target mean mu and covariance C.

    mode="iid":
        independent coordinates Uniform[a_j, b_j] chosen to match mean and variance diag(C).
        Only reproduces diagonal covariance; if C is not diagonal, we automatically fall back to "affine".

    mode="affine":
        X = mu + L U where U has iid Uniform(-sqrt(3), sqrt(3)) entries (mean 0, cov I) and L L^T = C.
        Matches the full covariance C exactly, but coordinates are correlated and marginals are not uniform.
    """
    p: int
    mu: Array
    C: Array
    mode: Literal["iid", "affine"] = "affine"

    def __post_init__(self) -> None:
        self.mu = as_1d(self.mu, self.p, "mu")
        self.C = symmetrize(as_2d(self.C, self.p, "C"))
        if self.mode == "iid" and not is_diagonal(self.C):
            self.mode = "affine"

    def sample(self, n: int, rng: Optional[Rng] = None) -> Array:
        rng = rng_or_default(rng)
        n = int(n)
        if n == 0:
            return np.zeros((0, self.p), dtype=float)

        if self.mode == "iid":
            var = np.diag(self.C).reshape(-1)
            if np.any(var < 0):
                raise ValueError("Diagonal variances must be nonnegative for iid uniform sampling.")
            half_width = np.sqrt(3.0 * var)
            low = self.mu - half_width
            high = self.mu + half_width
            return rng.uniform(low=low, high=high, size=(n, self.p))

        if self.mode == "affine":
            U = rng.uniform(low=-np.sqrt(3.0), high=np.sqrt(3.0), size=(n, self.p))
            L = safe_cholesky(self.C)
            return self.mu + U @ L.T

        raise ValueError(f"Unknown mode={self.mode!r}")

    def moments(self) -> Tuple[Array, Array]:
        return self.mu.copy(), self.C.copy()

    def params(self) -> Dict[str, Any]:
        return {"type": "UnifModel", "p": int(self.p), "mu": self.mu.copy(), "C": self.C.copy(), "mode": self.mode}


@dataclass
class BetaModel(BaseFeatureModel):
    """
    Beta-based feature model with target mean/covariance.

    If mode=="iid":
        - independent coordinates, each true Beta on [0,1]
        - requires target mu in (0,1)^p
        - requires target C to be diagonal and feasible for a Beta variance
          (var_j < mu_j(1-mu_j))

    If mode=="affine":
        - samples iid Beta coordinates (base shape), then applies an affine mixing
          to match the *full* target covariance C exactly:
              X = mu + (X0 - E[X0]) A^T,  where  A C0 A^T = C
        - coordinates are generally NOT independent and NOT marginally Beta.

    You can optionally provide alpha/beta to control the *base* Beta shape in affine mode.
    """
    p: int
    mu: Optional[Array] = None
    C: Optional[Array] = None
    mode: Literal["iid", "affine"] = "affine"

    # Optional base Beta parameters (used when not uniquely determined from moments)
    alpha: Optional[Array] = None
    beta: Optional[Array] = None
    base_alpha: float = 2.0
    base_beta: float = 2.0

    cov_reg: float = 1e-12

    # ---- internal / derived ----
    _alpha: Array = field(init=False, repr=False)
    _beta: Array = field(init=False, repr=False)
    _base_mu: Array = field(init=False, repr=False)
    _A: Array = field(init=False, repr=False)  # mixing matrix
    _mu: Array = field(init=False, repr=False)  # target mean
    _C: Array = field(init=False, repr=False)   # target covariance

    def __post_init__(self) -> None:
        target_specified = (self.mu is not None) or (self.C is not None)
        if target_specified and (self.mu is None or self.C is None):
            raise ValueError("BetaModel: either provide both mu and C, or provide neither.")

        if target_specified:
            mu_t = as_1d(self.mu, self.p, "mu")  # type: ignore[arg-type]
            C_t = symmetrize(as_2d(self.C, self.p, "C"))  # type: ignore[arg-type]
            # Regularize slightly to help with Cholesky
            if self.cov_reg > 0:
                C_t = C_t + float(self.cov_reg) * np.eye(self.p)
        else:
            mu_t = None
            C_t = None

        # If iid requested but C is not diagonal, fall back to affine.
        mode = self.mode
        if target_specified and mode == "iid" and not is_diagonal(C_t):  # type: ignore[arg-type]
            mode = "affine"
        self.mode = mode

        # Choose base alpha/beta
        if self.alpha is not None or self.beta is not None:
            if self.alpha is None or self.beta is None:
                raise ValueError("BetaModel: provide both alpha and beta (or neither).")
            a0 = as_1d(self.alpha, self.p, "alpha")
            b0 = as_1d(self.beta, self.p, "beta")
            if np.any(a0 <= 0) or np.any(b0 <= 0):
                raise ValueError("BetaModel: alpha and beta must be strictly positive.")
        else:
            # If we have target moments and they're feasible for a true Beta, use them as base.
            # This keeps affine distortions minimal when C has off-diagonal entries.
            a0 = None
            b0 = None
            if target_specified:
                # try to infer per-coordinate alpha/beta from mean+variance
                var_t = np.diag(C_t)  # type: ignore[arg-type]
                # Feasibility checks (only for deriving base params; affine mode can still work without)
                if np.all((mu_t > 0) & (mu_t < 1)) and np.all(var_t >= 0):
                    # For Beta: var = mu(1-mu)/(s+1)  =>  s = mu(1-mu)/var - 1
                    denom = var_t
                    # Guard against zeros: if var=0, use large concentration
                    s = np.empty_like(mu_t)
                    tiny = 1e-15
                    ok = denom > tiny
                    s[~ok] = 1e12
                    s[ok] = (mu_t[ok] * (1.0 - mu_t[ok])) / denom[ok] - 1.0
                    if np.all(s > 0):
                        a0 = mu_t * s
                        b0 = (1.0 - mu_t) * s

            if a0 is None or b0 is None:
                # Fallback base parameters
                a0 = np.full(self.p, float(self.base_alpha), dtype=float)
                b0 = np.full(self.p, float(self.base_beta), dtype=float)

        self._alpha = a0
        self._beta = b0

        # Base moments
        s0 = self._alpha + self._beta
        base_mu = self._alpha / s0
        base_var = (self._alpha * self._beta) / (s0 * s0 * (s0 + 1.0))
        base_C = np.diag(base_var) + float(self.cov_reg) * np.eye(self.p)

        self._base_mu = base_mu

        if target_specified:
            self._mu = mu_t  # type: ignore[assignment]
            self._C = C_t    # type: ignore[assignment]
        else:
            self._mu = base_mu.copy()
            self._C = base_C.copy()

        if self.mode == "iid":
            # In iid mode, we want TRUE Beta marginals. Therefore, the requested target must be feasible
            # and diagonal. If target was not specified, iid just means base (already iid).
            if target_specified:
                if not np.all((mu_t > 0) & (mu_t < 1)):
                    raise ValueError("BetaModel(iid): requires target mu entries in (0,1).")
                if not is_diagonal(C_t):
                    raise ValueError("BetaModel(iid): requires diagonal target covariance C.")
                var_t = np.diag(C_t)
                if np.any(var_t < 0):
                    raise ValueError("BetaModel(iid): variances must be nonnegative.")
                # Feasibility: var < mu(1-mu)
                if np.any(var_t >= mu_t * (1.0 - mu_t) - 1e-15):
                    raise ValueError(
                        "BetaModel(iid): infeasible variance for Beta. Need var_j < mu_j(1-mu_j)."
                    )
                # Override base params to match the target moments exactly (no affine step needed).
                s = (mu_t * (1.0 - mu_t)) / var_t - 1.0
                self._alpha = mu_t * s
                self._beta = (1.0 - mu_t) * s
                self._base_mu = mu_t
                self._mu = mu_t
                self._C = np.diag(var_t) + float(self.cov_reg) * np.eye(self.p)
                self._A = np.eye(self.p, dtype=float)
            else:
                self._A = np.eye(self.p, dtype=float)

        elif self.mode == "affine":
            # Compute A such that A base_C A^T = target_C (or base_C if target unspecified).
            self._A = _affine_matrix_from_covariances(base_C, self._C)
        else:
            raise ValueError(f"Unknown mode={self.mode!r}")

        # Expose (mu,C) as public-ish attributes for convenience
        self.mu = self._mu
        self.C = self._C

    def sample(self, n: int, rng: Optional[Rng] = None) -> Array:
        rng = rng_or_default(rng)
        n = int(n)
        if n == 0:
            return np.zeros((0, self.p), dtype=float)

        X0 = rng.beta(self._alpha, self._beta, size=(n, self.p))
        X = self._mu + (X0 - self._base_mu) @ self._A.T
        return X.astype(float, copy=False)

    def moments(self) -> Tuple[Array, Array]:
        return np.asarray(self._mu, dtype=float).copy(), np.asarray(self._C, dtype=float).copy()

    def params(self) -> Dict[str, Any]:
        return {
            "type": "BetaModel",
            "p": int(self.p),
            "mode": self.mode,
            "mu": np.asarray(self._mu, dtype=float).copy(),
            "C": np.asarray(self._C, dtype=float).copy(),
            "alpha": np.asarray(self._alpha, dtype=float).copy(),
            "beta": np.asarray(self._beta, dtype=float).copy(),
        }


@dataclass
class BernoulliModel(BaseFeatureModel):
    """
    Bernoulli-based feature model with target mean/covariance.

    support="pm1":
        base takes values in {-a, +a} (a>=0), with P(+a)=prob.
        In iid mode with target specified, prob and a are chosen to match the requested
        mean and variance *exactly* for each coordinate.

    support="01":
        base takes values in {0,1}, with P(1)=prob.
        In iid mode with target specified, the requested mean must lie in [0,1] and
        the requested variance must match prob(1-prob); otherwise this mode is infeasible.

    As for BetaModel:
      - mode="iid" preserves the intended coordinate-wise two-point distribution (requires diagonal C).
      - mode="affine" matches full covariance via an affine mixing step, destroying the exact
        Bernoulli marginals.
    """
    p: int
    mu: Optional[Array] = None
    C: Optional[Array] = None
    mode: Literal["iid", "affine"] = "affine"

    # If target is not specified, prob drives the base distribution.
    prob: Optional[Array] = None
    support: Literal["pm1", "01"] = "pm1"

    cov_reg: float = 1e-12

    # ---- internal / derived ----
    _prob: Array = field(init=False, repr=False)
    _values_neg: Array = field(init=False, repr=False)
    _values_pos: Array = field(init=False, repr=False)
    _base_mu: Array = field(init=False, repr=False)
    _A: Array = field(init=False, repr=False)
    _mu: Array = field(init=False, repr=False)
    _C: Array = field(init=False, repr=False)

    def __post_init__(self) -> None:
        target_specified = (self.mu is not None) or (self.C is not None)
        if target_specified and (self.mu is None or self.C is None):
            raise ValueError("BernoulliModel: either provide both mu and C, or provide neither.")

        if target_specified:
            mu_t = as_1d(self.mu, self.p, "mu")  # type: ignore[arg-type]
            C_t = symmetrize(as_2d(self.C, self.p, "C"))  # type: ignore[arg-type]
            if self.cov_reg > 0:
                C_t = C_t + float(self.cov_reg) * np.eye(self.p)
        else:
            mu_t = None
            C_t = None

        mode = self.mode
        if target_specified and mode == "iid" and not is_diagonal(C_t):  # type: ignore[arg-type]
            mode = "affine"
        self.mode = mode

        if self.support not in ("pm1", "01"):
            raise ValueError("support must be 'pm1' or '01'.")

        if target_specified:
            var_t = np.diag(C_t)  # type: ignore[arg-type]
            if np.any(var_t < 0):
                raise ValueError("BernoulliModel: target variances must be nonnegative.")

            if self.support == "pm1":
                # Choose a and prob to match mean/var per coordinate:
                #   X ∈ {-a,+a}, P(+a)=p
                #   E[X]=a(2p-1)=mu, Var[X]=a^2 - mu^2 = var  =>  a = sqrt(var+mu^2)
                a = np.sqrt(np.maximum(var_t + mu_t * mu_t, 0.0))  # type: ignore[operator]
                # Handle a=0 (degenerate)
                m = np.zeros_like(a)
                mask = a > 1e-15
                m[mask] = mu_t[mask] / a[mask]  # type: ignore[index]
                m = np.clip(m, -1.0, 1.0)
                p = 0.5 * (m + 1.0)
                p = np.clip(p, 0.0, 1.0)
                vneg = -a
                vpos = +a
                base_mu = mu_t  # already matched
                base_C = np.diag(var_t) + float(self.cov_reg) * np.eye(self.p)

            else:  # support=="01"
                if np.any((mu_t < 0) | (mu_t > 1)):  # type: ignore[operator]
                    raise ValueError("BernoulliModel(01): target mu must be in [0,1].")
                p = np.clip(mu_t, 0.0, 1.0)  # type: ignore[arg-type]
                expected_var = p * (1.0 - p)
                if np.any(np.abs(expected_var - var_t) > 1e-8 * (1.0 + expected_var)):
                    raise ValueError(
                        "BernoulliModel(01): target variance must match p(1-p) for p=mu."
                    )
                vneg = np.zeros(self.p)
                vpos = np.ones(self.p)
                base_mu = p
                base_C = np.diag(expected_var) + float(self.cov_reg) * np.eye(self.p)

            self._prob = p.astype(float, copy=False)
            self._values_neg = vneg.astype(float, copy=False)
            self._values_pos = vpos.astype(float, copy=False)
            self._base_mu = base_mu.astype(float, copy=False)

            self._mu = mu_t  # type: ignore[assignment]
            self._C = C_t    # type: ignore[assignment]

            if self.mode == "iid":
                self._A = np.eye(self.p, dtype=float)
            else:
                self._A = _affine_matrix_from_covariances(base_C, self._C)

        else:
            # No target moments specified: use prob directly.
            if self.prob is None:
                raise ValueError("BernoulliModel: if mu/C are not provided, you must provide prob.")
            p = as_1d(self.prob, self.p, "prob")
            if np.any((p < 0) | (p > 1)):
                raise ValueError("BernoulliModel: prob entries must be in [0,1].")

            if self.support == "01":
                vneg = np.zeros(self.p)
                vpos = np.ones(self.p)
                mu0 = p
                var0 = p * (1.0 - p)
            else:  # pm1 (unscaled ±1)
                vneg = -np.ones(self.p)
                vpos = +np.ones(self.p)
                mu0 = 2.0 * p - 1.0
                var0 = 4.0 * p * (1.0 - p)

            self._prob = p
            self._values_neg = vneg
            self._values_pos = vpos
            self._base_mu = mu0

            self._mu = mu0.copy()
            self._C = np.diag(var0) + float(self.cov_reg) * np.eye(self.p)
            self._A = np.eye(self.p, dtype=float)

        self.mu = self._mu
        self.C = self._C

    def sample(self, n: int, rng: Optional[Rng] = None) -> Array:
        rng = rng_or_default(rng)
        n = int(n)
        if n == 0:
            return np.zeros((0, self.p), dtype=float)

        U = rng.random(size=(n, self.p))
        # base samples in {vneg, vpos}
        X0 = np.where(U < self._prob, self._values_pos, self._values_neg).astype(float, copy=False)
        X = self._mu + (X0 - self._base_mu) @ self._A.T
        return X.astype(float, copy=False)

    def moments(self) -> Tuple[Array, Array]:
        return np.asarray(self._mu, dtype=float).copy(), np.asarray(self._C, dtype=float).copy()

    def params(self) -> Dict[str, Any]:
        return {
            "type": "BernoulliModel",
            "p": int(self.p),
            "mode": self.mode,
            "support": self.support,
            "mu": np.asarray(self._mu, dtype=float).copy(),
            "C": np.asarray(self._C, dtype=float).copy(),
            "prob": np.asarray(self._prob, dtype=float).copy(),
            "values_neg": np.asarray(self._values_neg, dtype=float).copy(),
            "values_pos": np.asarray(self._values_pos, dtype=float).copy(),
        }


@dataclass
class MultimodalModel(BaseFeatureModel):
    """
    Mixture of M feature models (unlabeled), optionally affinely matched to a target (mu, C).

    If mu/C are provided, the output is
        X = mu + (X0 - E[X0]) A^T
    where X0 is the raw mixture sample and A is chosen to match the target covariance exactly.
    """
    components: Sequence[BaseFeatureModel]
    weights: Optional[Array] = None

    # Optional target moments
    mu: Optional[Array] = None
    C: Optional[Array] = None

    cov_reg: float = 1e-12

    p: int = field(init=False)

    # internal
    _weights: Array = field(init=False, repr=False)
    _base_mu: Array = field(init=False, repr=False)
    _A: Array = field(init=False, repr=False)
    _mu: Array = field(init=False, repr=False)
    _C: Array = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if len(self.components) == 0:
            raise ValueError("components must be a non-empty sequence.")
        p0 = int(self.components[0].p)
        if any(int(c.p) != p0 for c in self.components):
            raise ValueError("All components must have the same dimension p.")
        self.p = p0

        if self.weights is None:
            w = np.ones(len(self.components), dtype=float)
        else:
            w = np.asarray(self.weights, dtype=float).reshape(-1)
        self._weights = normalize_probs(w, name="weights")
        if self._weights.size != len(self.components):
            raise ValueError("weights must have length len(components).")

        # mixture base moments
        mus: List[Array] = []
        covs: List[Array] = []
        for c in self.components:
            mu_c, C_c = c.moments()
            mus.append(np.asarray(mu_c, dtype=float))
            covs.append(np.asarray(C_c, dtype=float))

        base_mu = np.sum([wk * mk for wk, mk in zip(self._weights, mus)], axis=0)
        base_C = np.zeros((self.p, self.p), dtype=float)
        for wk, mk, Sk in zip(self._weights, mus, covs):
            dm = (mk - base_mu).reshape(-1, 1)
            base_C += wk * (Sk + dm @ dm.T)
        base_C = symmetrize(base_C) + float(self.cov_reg) * np.eye(self.p)

        self._base_mu = base_mu

        target_specified = (self.mu is not None) or (self.C is not None)
        if target_specified and (self.mu is None or self.C is None):
            raise ValueError("MultimodalModel: either provide both mu and C, or provide neither.")

        if target_specified:
            mu_t = as_1d(self.mu, self.p, "mu")  # type: ignore[arg-type]
            C_t = symmetrize(as_2d(self.C, self.p, "C"))  # type: ignore[arg-type]
            if self.cov_reg > 0:
                C_t = C_t + float(self.cov_reg) * np.eye(self.p)
            self._mu = mu_t
            self._C = C_t
            self._A = _affine_matrix_from_covariances(base_C, C_t)
        else:
            self._mu = base_mu
            self._C = base_C
            self._A = np.eye(self.p, dtype=float)

        self.mu = self._mu
        self.C = self._C

    def sample(self, n: int, rng: Optional[Rng] = None) -> Array:
        rng = rng_or_default(rng)
        n = int(n)
        if n == 0:
            return np.zeros((0, self.p), dtype=float)

        m = len(self.components)
        idx = rng.choice(m, size=n, p=self._weights)
        X0 = np.empty((n, self.p), dtype=float)
        for j in range(m):
            mask = idx == j
            nj = int(mask.sum())
            if nj == 0:
                continue
            X0[mask] = self.components[j].sample(nj, rng=rng)

        X = self._mu + (X0 - self._base_mu) @ self._A.T
        return X.astype(float, copy=False)

    def moments(self) -> Tuple[Array, Array]:
        return np.asarray(self._mu, dtype=float).copy(), np.asarray(self._C, dtype=float).copy()

    def params(self) -> Dict[str, Any]:
        return {
            "type": "MultimodalModel",
            "p": int(self.p),
            "weights": self._weights.copy(),
            "mu": np.asarray(self._mu, dtype=float).copy(),
            "C": np.asarray(self._C, dtype=float).copy(),
            "components": [c.params() for c in self.components],
        }


@dataclass
class BimodalModel(MultimodalModel):
    """
    Convenience wrapper for a 2-component mixture.
    """
    left: BaseFeatureModel = field(default=None)   # type: ignore[assignment]
    right: BaseFeatureModel = field(default=None)  # type: ignore[assignment]
    pi_right: float = 0.5

    # override parent init-fields
    components: Sequence[BaseFeatureModel] = field(init=False, repr=False)
    weights: Optional[Array] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.left is None or self.right is None:
            raise ValueError("BimodalModel requires left and right components.")
        self.components = [self.left, self.right]
        self.weights = np.asarray([1.0 - float(self.pi_right), float(self.pi_right)], dtype=float)
        super().__post_init__()

    def params(self) -> Dict[str, Any]:
        d = super().params()
        d["type"] = "BimodalModel"
        d["pi_right"] = float(self.pi_right)
        return d


# ======================================================================================
# Data models (labeled): (X, y)
# ======================================================================================

class BaseDataModel(ABC):
    """
    Base interface for labeled data models producing (X, y).

    IMPORTANT: `num_classes` refers to the number of *feature-classes* / mixture components
    exposed by `sample_class`, not necessarily the cardinality of the label space y.
    """

    p: int

    @property
    def num_classes(self) -> int:
        return 1

    @abstractmethod
    def sample(self, n: int, rng: Optional[Rng] = None) -> Tuple[Array, Array]:
        ...

    def sample_class(self, class_index: int, n: int, rng: Optional[Rng] = None) -> Tuple[Array, Array]:
        raise NotImplementedError

    @abstractmethod
    def class_params(self) -> Dict[str, Any]:
        """
        Should return (at minimum):
            dict(p=..., num_classes=..., gamma=..., mus=..., covs=..., y_values=...)
        where mus/covs are class-conditional moments of X.
        """
        ...

    # ------------------------------------------------------------------
    # Utility: Monte-Carlo validation of (mu_k, C_k, gamma_k)
    # ------------------------------------------------------------------
    def validate_model_moments(self, n_samples_per_class: int = 20000) -> None:
        """
        Monte-Carlo check: compares sampled class means/covs with class_params().

        Works even when the model's returned labels are NOT class labels (e.g. TeacherStudentModel),
        because it uses sample_class(k, ...) rather than filtering by y.
        """
        params = self.class_params()
        mus = params.get("mus")
        covs = params.get("covs")
        gamma = np.asarray(params.get("gamma"), dtype=float).reshape(-1)
        if mus is None or covs is None or gamma.size != self.num_classes:
            raise ValueError("class_params() must provide mus/covs and gamma of length num_classes.")

        print(f"--- Validating {self.__class__.__name__} ---")
        print(f"{'k':<4} | {'gamma':<10} | {'||mu-mu_hat||':<14} | {'||C-C_hat||_F':<14}")
        print("-" * 52)

        for k in range(self.num_classes):
            Xk, _ = self.sample_class(k, int(n_samples_per_class))
            mu_hat = Xk.mean(axis=0)
            C_hat = np.cov(Xk, rowvar=False, ddof=1)
            if C_hat.ndim == 0:  # p=1
                C_hat = C_hat.reshape(1, 1)

            mu_err = float(np.linalg.norm(mu_hat - mus[k]))
            C_err = float(np.linalg.norm(C_hat - covs[k], ord="fro"))
            print(f"{k:<4} | {gamma[k]:<10.4f} | {mu_err:<14.4e} | {C_err:<14.4e}")

    # ------------------------------------------------------------------
    # Classification error helpers (binary labels only)
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_y_binary(y: Array) -> Array:
        """
        Accept y as:
          - (n,) with values in {-1,+1} or {0,1}
          - (n,2) super-label u=(eps,y) where we take y=u[:,1]
        Returns y in {-1,+1} as float.
        """
        y = np.asarray(y)
        if y.ndim == 2 and y.shape[1] >= 2:
            y = y[:, 1]  # (eps, y)
        y = y.reshape(-1).astype(float, copy=False)

        # Map {0,1} -> {-1,+1}
        uniq = np.unique(y)
        if np.all(np.isin(uniq, [-1.0, 1.0])):
            return y
        if np.all(np.isin(uniq, [0.0, 1.0])):
            return 2.0 * y - 1.0

        raise ValueError("Binary classification error requires y in {-1,+1} or {0,1} (or u=(eps,y)).")

    def error_classif_th(
        self,
        mu: Array,
        alpha: Union[float, Array] = 0.0,
        num_trials: int = 100000,
        rng: Optional[Rng] = None,
        select_class: Optional[Sequence[int]] = None,
    ) -> float:
        """
        Monte-Carlo estimate of binary misclassification error for the stochastic score model:
            score = X @ mu + alpha_k * Z,   Z ~ N(0,1)

        The prediction is sign_pm1(score). The true label is the y returned by the model.

        alpha can be:
          - a scalar (same alpha for all classes), or
          - an array of shape (num_classes,) giving alpha per class.

        select_class optionally restricts which feature-classes are included in the weighted error.
        (Classes not in select_class are ignored, i.e. contribute 0 weight.)
        """
        rng = rng_or_default(rng)
        mu = as_1d(mu, self.p, "mu")
        num_trials = int(num_trials)

        params = self.class_params()
        gamma = np.asarray(params.get("gamma"), dtype=float).reshape(-1)
        if gamma.shape[0] != self.num_classes:
            raise ValueError("class_params()['gamma'] must have length num_classes.")

        if select_class is None or len(select_class) == 0:
            classes = list(range(self.num_classes))
        else:
            classes = [int(k) for k in select_class]
            if any(k < 0 or k >= self.num_classes for k in classes):
                raise ValueError("select_class contains an out-of-range class index.")

        alpha_arr: Optional[Array]
        if np.isscalar(alpha):
            alpha_arr = None
            alpha_scalar = float(alpha)
        else:
            alpha_arr = np.asarray(alpha, dtype=float).reshape(-1)
            if alpha_arr.shape[0] != self.num_classes:
                raise ValueError("alpha must be a scalar or an array of shape (num_classes,).")
            alpha_scalar = 0.0

        gamma_sel_sum = float(np.sum(gamma[classes]))
        if gamma_sel_sum <= 0:
            return 0.0

        # Allocate trials across classes proportional to gamma (conditional on selected classes)
        nks = np.floor(num_trials * gamma[classes] / gamma_sel_sum).astype(int)
        # Make sure total equals num_trials
        remainder = num_trials - int(np.sum(nks))
        if remainder > 0:
            # distribute remaining counts to classes with largest fractional parts
            frac = (num_trials * gamma[classes] / gamma_sel_sum) - nks
            order = np.argsort(-frac)
            for i in range(remainder):
                nks[order[i % len(order)]] += 1

        err = 0.0
        for k, nk in zip(classes, nks):
            nk = int(nk)
            if nk <= 0:
                continue
            Xk, yk = self.sample_class(k, nk, rng=rng)
            yk = self._extract_y_binary(yk)
            a = float(alpha_arr[k]) if alpha_arr is not None else alpha_scalar
            z = rng.standard_normal(size=nk)
            score = Xk @ mu + a * z
            pred = sign_pm1(score)
            err_k = float(np.mean(pred != yk))
            err += float(gamma[k]) * err_k

        return float(err)

    def error_classif_emp(
        self,
        thetas_rcrd: Sequence[Array],
        n_test: int = 5000,
        rng: Optional[Rng] = None,
        select_class: Optional[Sequence[int]] = None,
    ) -> float:
        """
        Empirical binary misclassification error averaged over a list of classifier weights thetas_rcrd.
        For each theta, we draw fresh test samples from the data model.

        Prediction: sign_pm1(X @ theta)
        True labels: y returned by the model (must be binary ±1 or 0/1, or u=(eps,y)).
        """
        rng = rng_or_default(rng)
        n_test = int(n_test)

        params = self.class_params()
        gamma = np.asarray(params.get("gamma"), dtype=float).reshape(-1)
        if gamma.shape[0] != self.num_classes:
            raise ValueError("class_params()['gamma'] must have length num_classes.")

        if select_class is None or len(select_class) == 0:
            classes = list(range(self.num_classes))
        else:
            classes = [int(k) for k in select_class]
            if any(k < 0 or k >= self.num_classes for k in classes):
                raise ValueError("select_class contains an out-of-range class index.")

        gamma_sel_sum = float(np.sum(gamma[classes]))
        if gamma_sel_sum <= 0:
            return 0.0

        # Allocate test points across classes proportional to gamma (conditional on selected classes)
        nks = np.floor(n_test * gamma[classes] / gamma_sel_sum).astype(int)
        remainder = n_test - int(np.sum(nks))
        if remainder > 0:
            frac = (n_test * gamma[classes] / gamma_sel_sum) - nks
            order = np.argsort(-frac)
            for i in range(remainder):
                nks[order[i % len(order)]] += 1

        thetas = [as_1d(theta, self.p, "theta") for theta in thetas_rcrd]
        if len(thetas) == 0:
            raise ValueError("thetas_rcrd must be a non-empty sequence.")

        err_total = 0.0
        for theta in thetas:
            err_theta = 0.0
            for k, nk in zip(classes, nks):
                nk = int(nk)
                if nk <= 0:
                    continue
                Xk, yk = self.sample_class(k, nk, rng=rng)
                yk = self._extract_y_binary(yk)
                pred = sign_pm1(Xk @ theta)
                err_k = float(np.mean(pred != yk))
                err_theta += float(gamma[k]) * err_k
            err_total += err_theta

        return float(err_total / len(thetas))


@dataclass
class MultiClassModel(BaseDataModel):
    """
    K-class model built from K feature models.

        y ~ Categorical(gamma)
        X | (y=k) ~ components[k]
        returned y is y_values[k] (default: 0..K-1)

    Here, y is the class label (supervised setting).
    """
    components: Sequence[BaseFeatureModel]
    gamma: Union[Array, Sequence[float]]
    y_values: Optional[Sequence[float]] = None

    p: int = field(init=False)
    _gamma: Array = field(init=False, repr=False)
    _y_values: List[float] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if len(self.components) == 0:
            raise ValueError("components must have length >= 1.")
        p0 = int(self.components[0].p)
        if any(int(c.p) != p0 for c in self.components):
            raise ValueError("All components must share the same dimension p.")
        self.p = p0

        self._gamma = normalize_probs(self.gamma, name="gamma")
        if self._gamma.size != len(self.components):
            raise ValueError("gamma must have length len(components).")

        K = len(self.components)
        if self.y_values is None:
            self._y_values = [float(k) for k in range(K)]
        else:
            if len(self.y_values) != K:
                raise ValueError("y_values must have length K.")
            self._y_values = [float(v) for v in self.y_values]

    @property
    def num_classes(self) -> int:
        return len(self.components)

    def sample_class(self, class_index: int, n: int, rng: Optional[Rng] = None) -> Tuple[Array, Array]:
        rng = rng_or_default(rng)
        k = int(class_index)
        n = int(n)
        if not (0 <= k < self.num_classes):
            raise ValueError(f"class_index must be in [0,{self.num_classes-1}], got {k}.")
        X = self.components[k].sample(n, rng=rng)
        y = np.full(n, self._y_values[k], dtype=float)
        return X, y

    def sample(self, n: int, rng: Optional[Rng] = None) -> Tuple[Array, Array]:
        rng = rng_or_default(rng)
        n = int(n)
        idx = rng.choice(self.num_classes, size=n, p=self._gamma)
        X = np.empty((n, self.p), dtype=float)
        y = np.empty(n, dtype=float)
        for k in range(self.num_classes):
            mask = idx == k
            nk = int(mask.sum())
            if nk == 0:
                continue
            X[mask] = self.components[k].sample(nk, rng=rng)
            y[mask] = self._y_values[k]
        return X, y

    def class_params(self) -> Dict[str, Any]:
        mus: List[Array] = []
        covs: List[Array] = []
        for c in self.components:
            mu_c, C_c = c.moments()
            mus.append(np.asarray(mu_c, dtype=float))
            covs.append(np.asarray(C_c, dtype=float))
        return {
            "p": int(self.p),
            "num_classes": int(self.num_classes),
            "gamma": self._gamma.copy(),
            "mus": [m.copy() for m in mus],
            "covs": [C.copy() for C in covs],
            "y_values": list(self._y_values),
            "components": [c.params() for c in self.components],
        }

    # Explicitly re-expose the error_* methods here (for discoverability)
    def error_classif_th(self, *args, **kwargs) -> float:  # type: ignore[override]
        return super().error_classif_th(*args, **kwargs)

    def error_classif_emp(self, *args, **kwargs) -> float:  # type: ignore[override]
        return super().error_classif_emp(*args, **kwargs)


TeacherModel = Literal["linear_regression", "sign", "logistic"]


def sigmoid(t: Array) -> Array:
    t = np.asarray(t, dtype=float)
    return 1.0 / (1.0 + np.exp(-t))


@dataclass
class TeacherStudentModel(BaseDataModel):
    """
    Teacher/student model:

        X ~ x_model
        y = teacher(X)     (linear_regression, sign, logistic)

    IMPORTANT:
      - The teacher model's *classes* (for sample_class / class_params) are inherited from x_model.
      - Therefore, num_classes == x_model.num_classes if x_model is a BaseDataModel, else 1.
      - In particular, y_model="sign" does NOT automatically imply num_classes=2.

    Labels:
      - y_model="linear_regression": y ∈ R
      - y_model="sign" or "logistic": y ∈ {-1,+1}
    """
    x_model: Union[BaseFeatureModel, BaseDataModel]
    theta_teacher: Array
    y_model: TeacherModel = "linear_regression"
    noise_std: float = 1.0
    temperature: float = 1.0

    p: int = field(init=False)

    def __post_init__(self) -> None:
        self.p = int(getattr(self.x_model, "p"))
        self.theta_teacher = as_1d(self.theta_teacher, self.p, "theta_teacher")
        if self.temperature <= 0:
            raise ValueError("temperature must be > 0.")

    @property
    def num_classes(self) -> int:
        if isinstance(self.x_model, BaseDataModel):
            return int(self.x_model.num_classes)
        return 1

    def _sample_X_from_class(self, class_index: int, n: int, rng: Rng) -> Array:
        n = int(n)
        if isinstance(self.x_model, BaseDataModel):
            X, _ = self.x_model.sample_class(int(class_index), n, rng=rng)
            return np.asarray(X, dtype=float)
        # feature model: ignore class_index
        return np.asarray(self.x_model.sample(n, rng=rng), dtype=float)

    def _sample_X_unconditional(self, n: int, rng: Rng) -> Array:
        n = int(n)
        if isinstance(self.x_model, BaseDataModel):
            X, _ = self.x_model.sample(n, rng=rng)
            return np.asarray(X, dtype=float)
        return np.asarray(self.x_model.sample(n, rng=rng), dtype=float)

    def _sample_y_given_X(self, X: Array, rng: Rng) -> Array:
        s = X @ self.theta_teacher
        if self.y_model == "linear_regression":
            y = s + float(self.noise_std) * rng.standard_normal(size=s.shape[0])
            return y.astype(float, copy=False)
        if self.y_model == "sign":
            y = s + float(self.noise_std) * rng.standard_normal(size=s.shape[0])
            return np.where(y >= 0.0, 1.0, -1.0).astype(float, copy=False)
        if self.y_model == "logistic":
            t = s / float(self.temperature)
            p_pos = sigmoid(t)
            u = rng.random(size=s.shape[0])
            return np.where(u < p_pos, 1.0, -1.0).astype(float, copy=False)
        raise ValueError(f"Unknown y_model={self.y_model!r}")

    def sample(self, n: int, rng: Optional[Rng] = None) -> Tuple[Array, Array]:
        rng = rng_or_default(rng)
        X = self._sample_X_unconditional(int(n), rng)
        y = self._sample_y_given_X(X, rng)
        return X, y

    def sample_class(self, class_index: int, n: int, rng: Optional[Rng] = None) -> Tuple[Array, Array]:
        rng = rng_or_default(rng)
        X = self._sample_X_from_class(int(class_index), int(n), rng)
        y = self._sample_y_given_X(X, rng)
        return X, y

    def class_params(self) -> Dict[str, Any]:
        # Get class-conditional moments of X from x_model
        if isinstance(self.x_model, BaseDataModel):
            xparams = self.x_model.class_params()
            gamma = np.asarray(xparams.get("gamma"), dtype=float).reshape(-1)
            mus = [np.asarray(m, dtype=float) for m in xparams.get("mus", [])]
            covs = [np.asarray(C, dtype=float) for C in xparams.get("covs", [])]
            if gamma.shape[0] != self.num_classes or len(mus) != self.num_classes or len(covs) != self.num_classes:
                raise ValueError("x_model.class_params() is inconsistent with x_model.num_classes.")
        else:
            mu_x, C_x = self.x_model.moments()
            gamma = np.array([1.0], dtype=float)
            mus = [np.asarray(mu_x, dtype=float)]
            covs = [np.asarray(C_x, dtype=float)]

        return dict(
            p=int(self.p),
            num_classes=int(self.num_classes),
            gamma=gamma.copy(),
            mus=[m.copy() for m in mus],
            covs=[symmetrize(C).copy() for C in covs],
            y_values=None,  # teacher labels are not "class labels"
            theta_teacher=self.theta_teacher.copy(),
            y_model=self.y_model,
            noise_std=float(self.noise_std),
            temperature=float(self.temperature),
            # Often useful for pruning / analysis
            w_values=[self.theta_teacher.copy() for _ in range(self.num_classes)],
        )


# ======================================================================================
# User-provided LinearFactorMixedModel (BaseDataModel)
# ======================================================================================

@dataclass
class LinearFactorMixedModel(BaseDataModel):
    """
    User-provided model (reproduced with minor fixes: rng usage + style):

        y ~ Bernoulli(P) in {-1,+1}  OR  y ~ N(0,1)
        X = E + signal(y) in first q coordinates, then rotated by basis.

    signal_type="modded" adds a 1D Gaussian mixture on coordinate q.
    """
    p: int
    q: int
    P: Optional[float] = None        # Only relevant if y_type='bernoulli'
    s: Array = None                  # shape (q,)
    noise_std: float = 1.0
    basis: Optional[Array] = None    # shape (p, p), columns = v_i
    y_type: str = "bernoulli"        # 'bernoulli' or 'gaussian'
    signal_type: str = "default"     # 'modded' or 'default'

    def __post_init__(self) -> None:
        if self.y_type not in ["bernoulli", "gaussian"]:
            raise ValueError("y_type must be 'bernoulli' or 'gaussian'")
        if self.signal_type not in ["modded", "default"]:
            raise ValueError("signal_type must be 'modded' or 'default'")

        if self.y_type == "bernoulli":
            if self.P is None or not (0.0 < float(self.P) < 1.0):
                raise ValueError("P must be in (0, 1) for bernoulli y_type")

        if self.q > self.p:
            raise ValueError("q must be <= p")

        self.s = np.asarray(self.s, dtype=float).reshape(-1)
        if self.s.shape != (self.q,):
            raise ValueError(f"s must have shape ({self.q},)")

        if self.basis is None:
            self.basis = np.eye(self.p, dtype=float)
        else:
            self.basis = np.asarray(self.basis, dtype=float)
            if self.basis.shape != (self.p, self.p):
                raise ValueError("basis must have shape (p, p)")

    @property
    def num_classes(self) -> int:
        return 2 if self.y_type == "bernoulli" else 1

    def sample(self, n: int, rng: Optional[Rng] = None) -> Tuple[Array, Array]:
        rng = rng_or_default(rng)
        n = int(n)

        # --- Sample y ---
        if self.y_type == "bernoulli":
            y = rng.uniform(size=n) < float(self.P)
            y = np.where(y, 1.0, -1.0)
        else:  # gaussian
            y = rng.standard_normal(size=n)

        # --- Sample noise ---
        X = float(self.noise_std) * rng.standard_normal(size=(n, self.p))

        # --- Signal contribution ---
        X[:, : self.q] += y[:, None] * self.s[None, :]

        if self.signal_type == "modded":
            pi = 0.5
            mus = np.array([-1.0, 1.0])
            sigmas = np.array([0.2, 0.2])
            components = rng.choice([0, 1], size=n, p=[pi, 1 - pi])
            X[:, self.q] = rng.normal(loc=mus[components], scale=sigmas[components])

        # --- Rotate into basis ---
        X = X @ self.basis.T
        return X, y.astype(float, copy=False)

    def sample_class(self, class_index: int, n: int, rng: Optional[Rng] = None) -> Tuple[Array, Array]:
        # For gaussian y_type, there is no meaningful class conditioning.
        if self.y_type == "gaussian":
            return self.sample(n, rng=rng)

        rng = rng_or_default(rng)
        n = int(n)
        k = int(class_index)
        if k not in (0, 1):
            raise ValueError("class_index must be 0 or 1 for bernoulli y_type")

        y_val = 1.0 if k == 1 else -1.0
        y = np.full(n, y_val)

        # --- Sample noise ---
        X = float(self.noise_std) * rng.standard_normal(size=(n, self.p))

        # --- Signal contribution ---
        X[:, : self.q] += y[:, None] * self.s[None, :]

        if self.signal_type == "modded":
            pi = 0.5
            mus = np.array([-1.0, 1.0])
            sigmas = np.array([0.2, 0.2])
            components = rng.choice([0, 1], size=n, p=[pi, 1 - pi])
            X[:, self.q] = rng.normal(loc=mus[components], scale=sigmas[components])

        X = X @ self.basis.T
        return X, y.astype(float, copy=False)

    def class_params(self) -> Dict[str, Any]:
        if self.y_type == "bernoulli":
            mu_pos = np.zeros(self.p)
            mu_neg = np.zeros(self.p)
            mu_pos[: self.q] = self.s
            mu_neg[: self.q] = -self.s
            mu_pos = mu_pos @ self.basis.T
            mu_neg = mu_neg @ self.basis.T
            cov = (float(self.noise_std) ** 2) * np.eye(self.p)
            return dict(
                p=self.p,
                num_classes=2,
                gamma=np.array([1.0 - float(self.P), float(self.P)], dtype=float),
                mus=[mu_neg, mu_pos],
                covs=[cov, cov],
                y_values=[-1.0, 1.0],
                w_values=[None, None],
            )
        # Gaussian y: unconditional mean/cov for X only
        cov = (float(self.noise_std) ** 2) * np.eye(self.p)
        return dict(
            p=self.p,
            num_classes=1,
            gamma=np.array([1.0], dtype=float),
            mus=[np.zeros(self.p)],
            covs=[cov],
            y_values=None,
            w_values=[None],
        )


# Backward-compat alias some users might have relied on
LinearFactorModel = LinearFactorMixedModel


# ======================================================================================
# MNIST grouped data model (labeled): y is class index
# ======================================================================================

ActivationName = Literal["identity", "relu", "tanh", "cos", "sign_pm1"]


def get_activation(name: ActivationName) -> Callable[[Array], Array]:
    if name == "identity":
        return lambda z: z
    if name == "relu":
        return lambda z: np.maximum(z, 0.0)
    if name == "tanh":
        return np.tanh
    if name == "cos":
        return np.cos
    if name == "sign_pm1":
        return sign_pm1
    raise ValueError(f"Unknown activation={name!r}")


def flatten_images_uint8(images: Array) -> Array:
    """
    images: (N, 28, 28) or already flattened (N, 784).
    returns: (N, 784)
    """
    images = np.asarray(images)
    if images.ndim == 3:
        n, h, w = images.shape
        return images.reshape(n, h * w)
    if images.ndim == 2:
        return images
    raise ValueError(f"Expected images with ndim 2 or 3, got shape {images.shape}")


def normalize_pixels(
    X: Array,
    pixel_scaling: Literal["uint8", "unit_interval"],
    dtype: np.dtype,
) -> Array:
    """
    pixel_scaling:
      - "uint8": keep in [0,255] but cast to dtype
      - "unit_interval": scale to [0,1] assuming 8-bit images
    """
    X = np.asarray(X).astype(dtype, copy=False)
    if pixel_scaling == "uint8":
        return X
    if pixel_scaling == "unit_interval":
        maxv = float(np.max(X)) if X.size else 0.0
        if maxv > 1.5:
            return X / dtype.type(255.0)
        return X
    raise ValueError(f"Unknown pixel_scaling={pixel_scaling!r}")


def load_mnist_npz(path: str) -> Tuple[Array, Array, Array, Array]:
    """
    Load MNIST arrays from a local .npz file.

    Supports keys:
      - x_train, y_train, x_test, y_test (Keras convention)
      - X_train, y_train, X_test, y_test
      - images_train, labels_train, images_test, labels_test
    """
    with np.load(path) as f:
        keys = set(f.files)

        if {"x_train", "y_train", "x_test", "y_test"} <= keys:
            return f["x_train"], f["y_train"], f["x_test"], f["y_test"]

        if {"X_train", "y_train", "X_test", "y_test"} <= keys:
            return f["X_train"], f["y_train"], f["X_test"], f["y_test"]

        if {"images_train", "labels_train", "images_test", "labels_test"} <= keys:
            return f["images_train"], f["labels_train"], f["images_test"], f["labels_test"]

        raise ValueError(f"Unrecognized MNIST .npz format at {path!r}. Available keys: {sorted(keys)}")


def create_downsampling_matrix(target_m: int, original_dim: int = 28) -> Array:
    """
    Create a matrix W of shape (target_m^2, original_dim^2) that extracts pixels on a centered grid.
    If x is flattened (original_dim^2,), then (W @ x) is flattened (target_m^2,).
    """
    if target_m <= 0 or target_m > original_dim:
        raise ValueError("target_m must be in {1,...,original_dim}.")
    stride = original_dim // target_m
    grid_span = (target_m - 1) * stride + 1
    start_offset = (original_dim - grid_span) // 2

    W = np.zeros((target_m * target_m, original_dim * original_dim), dtype=float)
    row_idx = 0
    for i in range(target_m):
        for j in range(target_m):
            orig_row = start_offset + i * stride
            orig_col = start_offset + j * stride
            flat_idx_orig = orig_row * original_dim + orig_col
            W[row_idx, flat_idx_orig] = 1.0
            row_idx += 1
    return W


@dataclass
class MNISTGroupedDataModel(BaseDataModel):
    """
    MNIST model where each "class" is a user-defined group of digits.

    Feature map (always used):
        Phi(X) = activation(X W^T + b) * feature_scale
    Convention:
        - W=None means identity (raw pixels). This is equivalent to W=I and identity activation.

    y returned by sample()/sample_class() is the class index 0..K-1.
    """
    data_path: str
    split: Literal["train", "test", "full"] = "train"
    stats_split: Literal["train", "test", "full"] = "test"

    class_groups: Optional[Sequence[Sequence[int]]] = None  # list of digit lists

    # Feature map
    W: Optional[Array] = None                   # (m,p_raw) or None for identity
    bias: Optional[Array] = None                # (m,)
    activation: ActivationName = "identity"
    feature_scale: float = 1.0

    # Noise added to final representation returned by sample()/sample_class()
    noise_std: float = 0.0

    # Data formatting
    pixel_scaling: Literal["uint8", "unit_interval"] = "unit_interval"
    dtype: Any = np.float32
    replace: bool = True

    # Moments / numerics
    cov_reg: float = 1e-6

    # Derived
    p_raw: int = field(init=False)
    p: int = field(init=False)

    # Internal storage
    _X_train: Array = field(init=False, repr=False)
    _y_train: Array = field(init=False, repr=False)
    _X_test: Array = field(init=False, repr=False)
    _y_test: Array = field(init=False, repr=False)
    _groups: List[np.ndarray] = field(init=False, repr=False)

    _pools_cache: Dict[str, List[np.ndarray]] = field(default_factory=dict, init=False, repr=False)
    _params_cache: Dict[Tuple[bool, str], Dict[str, Any]] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        x_train, y_train, x_test, y_test = load_mnist_npz(self.data_path)
        dt = np.dtype(self.dtype)

        self._X_train = normalize_pixels(flatten_images_uint8(x_train), self.pixel_scaling, dt)
        self._y_train = np.asarray(y_train, dtype=int)
        self._X_test = normalize_pixels(flatten_images_uint8(x_test), self.pixel_scaling, dt)
        self._y_test = np.asarray(y_test, dtype=int)

        self.p_raw = int(self._X_train.shape[1])

        # Build / validate groups
        if self.class_groups is None:
            groups = [[d] for d in range(10)]
        else:
            groups = [list(g) for g in self.class_groups]
            if len(groups) == 0:
                raise ValueError("class_groups must contain at least one class.")

        cleaned: List[np.ndarray] = []
        for g in groups:
            if len(g) == 0:
                raise ValueError("Found an empty class in class_groups.")
            gg = np.unique(np.asarray(g, dtype=int))
            if np.any((gg < 0) | (gg > 9)):
                raise ValueError(f"Digits must be in 0..9; got {gg.tolist()}")
            cleaned.append(gg)
        self._groups = cleaned

        # Determine output dimension p and validate feature map params
        if self.W is None:
            self.p = self.p_raw
            m = self.p_raw
        else:
            W = np.asarray(self.W, dtype=float)
            if W.ndim != 2 or W.shape[1] != self.p_raw:
                raise ValueError(f"W must have shape (m,{self.p_raw}); got {W.shape}")
            m = int(W.shape[0])
            self.p = m

        if self.bias is not None:
            b = np.asarray(self.bias, dtype=float).reshape(-1)
            if b.shape != (m,):
                raise ValueError(f"bias must have shape ({m},), got {b.shape}")

    @property
    def num_classes(self) -> int:
        return len(self._groups)

    # ---------- internal helpers ----------
    def _get_split(self, split: Literal["train", "test", "full"]) -> Tuple[Array, Array]:
        if split == "train":
            return self._X_train, self._y_train
        if split == "test":
            return self._X_test, self._y_test
        if split == "full":
            X = np.concatenate([self._X_train, self._X_test], axis=0)
            y = np.concatenate([self._y_train, self._y_test], axis=0)
            return X, y
        raise ValueError(f"Unknown split={split!r}")

    def _class_pools(self, split: Literal["train", "test", "full"]) -> List[np.ndarray]:
        if split in self._pools_cache:
            return self._pools_cache[split]
        _, y = self._get_split(split)
        pools = [np.flatnonzero(np.isin(y, digits)) for digits in self._groups]
        if all(len(p) == 0 for p in pools):
            raise ValueError("All classes are empty on this split (no matching digits).")
        self._pools_cache[split] = pools
        return pools

    def transform(self, X_raw: Array) -> Array:
        dt = np.dtype(self.dtype)
        X_raw = np.asarray(X_raw, dtype=dt)

        if self.W is None:
            Z = X_raw
        else:
            W = np.asarray(self.W, dtype=dt)
            Z = X_raw @ W.T

        if self.bias is not None:
            b = np.asarray(self.bias, dtype=dt).reshape(-1)
            Z = Z + b

        Phi = get_activation(self.activation)(Z)
        if self.feature_scale != 1.0:
            Phi = Phi * dt.type(self.feature_scale)
        return Phi.astype(dt, copy=False)

    def _add_noise(self, X: Array, rng: Rng) -> Array:
        if self.noise_std <= 0:
            return X
        dt = np.dtype(self.dtype)
        return X + rng.normal(scale=float(self.noise_std), size=X.shape).astype(dt, copy=False)

    # ---------- public API ----------
    def sample(self, n: int, rng: Optional[Rng] = None) -> Tuple[Array, Array]:
        rng = rng_or_default(rng)
        n = int(n)

        X_raw, _ = self._get_split(self.split)
        pools = self._class_pools(self.split)

        sizes = np.array([len(p) for p in pools], dtype=float)
        if sizes.sum() <= 0:
            raise ValueError("No samples available after applying class_groups.")
        gamma = sizes / sizes.sum()

        y = rng.choice(self.num_classes, size=n, p=gamma)
        idx = np.empty(n, dtype=int)
        for k in range(self.num_classes):
            nk = int(np.sum(y == k))
            if nk == 0:
                continue
            pool = pools[k]
            if pool.size == 0:
                raise ValueError(f"Class {k} has no samples on split={self.split!r}.")
            idx[y == k] = rng.choice(pool, size=nk, replace=bool(self.replace))

        X = self.transform(X_raw[idx])
        X = self._add_noise(X, rng)
        return X.astype(float, copy=False), y.astype(float, copy=False)

    def sample_class(self, class_index: int, n: int, rng: Optional[Rng] = None) -> Tuple[Array, Array]:
        rng = rng_or_default(rng)
        n = int(n)
        k = int(class_index)
        if not (0 <= k < self.num_classes):
            raise ValueError(f"class_index must be in [0,{self.num_classes-1}], got {k}.")

        X_raw, _ = self._get_split(self.split)
        pool = self._class_pools(self.split)[k]
        if pool.size == 0:
            raise ValueError(f"No samples for class {k} on split={self.split!r}.")

        idx = rng.choice(pool, size=n, replace=bool(self.replace))
        X = self.transform(X_raw[idx])
        X = self._add_noise(X, rng)
        y = np.full(n, float(k))
        return X.astype(float, copy=False), y

    def class_params(self, *, include_noise: bool = True) -> Dict[str, Any]:
        cache_key = (bool(include_noise), self.stats_split)
        if cache_key in self._params_cache:
            return self._params_cache[cache_key]

        X_raw_all, _ = self._get_split(self.stats_split)
        pools = self._class_pools(self.stats_split)

        sizes = np.array([len(p) for p in pools], dtype=float)
        if sizes.sum() <= 0:
            raise ValueError("No stats samples available after applying class_groups.")
        gamma = sizes / sizes.sum()

        dt = np.dtype(self.dtype)
        mus: List[Array] = []
        covs: List[Array] = []

        for pool in pools:
            if pool.size == 0:
                mu = np.zeros(self.p, dtype=dt)
                C = float(self.cov_reg) * np.eye(self.p, dtype=dt)
            else:
                X_rep = self.transform(X_raw_all[pool])
                mu, C = moment_estimates(X_rep, reg=float(self.cov_reg), kind="full")
                mu = mu.astype(dt, copy=False)
                C = C.astype(dt, copy=False)

            if include_noise and self.noise_std > 0:
                C = C + dt.type(float(self.noise_std) ** 2) * np.eye(self.p, dtype=dt)

            mus.append(mu)
            covs.append(C)

        params = {
            "p": int(self.p),
            "num_classes": int(self.num_classes),
            "gamma": gamma.astype(dt, copy=False),
            "mus": mus,
            "covs": covs,
            "y_values": list(range(self.num_classes)),
            "class_groups": [g.tolist() for g in self._groups],
            "stats_split": self.stats_split,
            "include_noise": bool(include_noise),
            "noise_std": float(self.noise_std if include_noise else 0.0),
        }
        self._params_cache[cache_key] = params
        return params


# Backward-compatible aliases for MNIST helper names in older scripts
_flatten_uint8 = flatten_images_uint8
_as_2d_flattened_uint8 = flatten_images_uint8
_normalize_pixels = normalize_pixels
_load_mnist_npz = load_mnist_npz
_get_activation = get_activation


# ======================================================================================
# Pruning / super-label models
# ======================================================================================

class Pruner(Protocol):
    """
    Minimal protocol for pruners that produce per-sample weights eps ∈ [0,1].
    """
    def epsilon(self, X: Array, y: Array, rng: Optional[Rng] = None) -> Array:
        ...


@dataclass
class OracleMarginPruner:
    """
    Oracle / curation rule based on an oracle direction w_oracle and a margin threshold delta.

    Hard, label-aware version:
        eps(x, y) = 1[ y == sign(x^T w_oracle) ] · 1[ |x^T w_oracle| >= delta ].

    Soft relaxation (optional):
        eps(x, y) = 1[ y == sign(x^T w_oracle) ] · sigmoid( beta (|x^T w_oracle| - delta) ).
    """
    w_oracle: Array
    delta: float = 0.0
    label_aware: bool = True
    soft: bool = False
    beta: float = 10.0

    def __post_init__(self) -> None:
        self.w_oracle = np.asarray(self.w_oracle, dtype=float).reshape(-1)
        if float(self.delta) < 0:
            raise ValueError("delta must be >= 0")

    def epsilon(self, X: Array, y: Array, rng: Optional[Rng] = None) -> Array:  # pylint: disable=unused-argument
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        s = X @ self.w_oracle
        margin = np.abs(s)

        if self.soft:
            t = float(self.beta) * (margin - float(self.delta))
            gate = 1.0 / (1.0 + np.exp(-t))
        else:
            gate = (margin >= float(self.delta)).astype(float)

        if self.label_aware:
            y_oracle = sign_pm1(s)
            agree = (y == y_oracle).astype(float)
            gate = gate * agree

        return np.clip(gate, 0.0, 1.0)


@dataclass
class RealGeneratedPrunedModel(BaseDataModel):
    """
    Combined model for "real + generated + pruning" with super-label u=(eps, y).

    If the underlying task has K feature-classes (i.e. underlying model's num_classes), this combined
    model exposes 2K classes:
      - class 0..K-1      : real data from real_model, eps=1
      - class K..2K-1     : generated data from gen_model, eps=lambda_gen * pruner.epsilon(x,y)

    The returned label is u=(eps,y) with shape (n,2). `y` is the underlying model's label.
    """
    real_model: BaseDataModel
    gen_model: BaseDataModel
    pruner: Optional[Pruner] = None
    gamma_r: float = 0.5
    lambda_gen: float = 1.0

    p: int = field(init=False)
    K_base: int = field(init=False)

    _gamma: Array = field(init=False, repr=False)
    _mus: List[Array] = field(init=False, repr=False)
    _covs: List[Array] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        params_r = self.real_model.class_params()
        params_g = self.gen_model.class_params()

        self.p = int(params_r["p"])
        if int(params_g["p"]) != self.p:
            raise ValueError("real_model and gen_model must have the same feature dimension p")

        self.K_base = int(params_r["num_classes"])
        if int(params_g["num_classes"]) != self.K_base:
            raise ValueError("real_model and gen_model must have the same num_classes")

        gr = float(self.gamma_r)
        if not (0.0 < gr < 1.0):
            raise ValueError("gamma_r must be in (0,1)")
        gg = 1.0 - gr

        gamma_r_classes = np.asarray(params_r["gamma"], dtype=float).reshape(-1)
        gamma_g_classes = np.asarray(params_g["gamma"], dtype=float).reshape(-1)
        if gamma_r_classes.shape[0] != self.K_base or gamma_g_classes.shape[0] != self.K_base:
            raise ValueError("Underlying models returned inconsistent gamma sizes")

        self._gamma = np.concatenate([gr * gamma_r_classes, gg * gamma_g_classes], axis=0)
        self._gamma = self._gamma / float(np.sum(self._gamma))

        # Mus / covariances for X, per class (pruning does not change X distribution)
        self._mus = [np.asarray(m, dtype=float).reshape(-1) for m in params_r.get("mus", [])] + \
                    [np.asarray(m, dtype=float).reshape(-1) for m in params_g.get("mus", [])]
        self._covs = [np.asarray(C, dtype=float) for C in params_r.get("covs", [])] + \
                     [np.asarray(C, dtype=float) for C in params_g.get("covs", [])]

        if len(self._covs) != 2 * self.K_base:
            raise ValueError("Expected both submodels to provide per-class covariances in class_params()['covs'].")

    @property
    def num_classes(self) -> int:
        return 2 * self.K_base

    def class_params(self) -> Dict[str, Any]:
        # w_values: attempt to expose per-class "teacher direction" if it exists
        w_r = None
        w_g = None
        if hasattr(self.real_model, "theta_teacher"):
            w_r = cast(Array, getattr(self.real_model, "theta_teacher"))
        if hasattr(self.gen_model, "theta_teacher"):
            w_g = cast(Array, getattr(self.gen_model, "theta_teacher"))
        w_values = None
        if w_r is not None or w_g is not None:
            w_values = [w_r for _ in range(self.K_base)] + [w_g for _ in range(self.K_base)]

        return dict(
            p=int(self.p),
            num_classes=int(self.num_classes),
            gamma=self._gamma.copy(),
            mus=[m.copy() for m in self._mus],
            covs=[C.copy() for C in self._covs],
            y_values=None,  # super-label
            w_values=w_values,
        )

    @staticmethod
    def _make_u(eps: Array, y: Array) -> Array:
        eps = np.asarray(eps, dtype=float).reshape(-1)
        y = np.asarray(y, dtype=float).reshape(-1)
        if eps.shape[0] != y.shape[0]:
            raise ValueError("eps and y must have the same length")
        return np.stack([eps, y], axis=1)  # shape (n,2)

    def sample_class(self, class_index: int, n: int, rng: Optional[Rng] = None) -> Tuple[Array, Array]:
        rng = rng_or_default(rng)
        n = int(n)
        k = int(class_index)
        if not (0 <= k < self.num_classes):
            raise ValueError("class_index out of range")

        if k < self.K_base:
            # Real
            X, y = self.real_model.sample_class(k, n, rng=rng)
            eps = np.ones(n, dtype=float)
            u = self._make_u(eps, y)
            return np.asarray(X, dtype=float), u

        # Generated
        X, y = self.gen_model.sample_class(k - self.K_base, n, rng=rng)
        if self.pruner is None:
            eps = np.ones(n, dtype=float)
        else:
            eps = np.asarray(self.pruner.epsilon(X, y, rng=rng), dtype=float).reshape(-1)
            if eps.shape[0] != n:
                raise ValueError("pruner.epsilon must return an array of shape (n,)")
        eps = float(self.lambda_gen) * eps
        u = self._make_u(eps, y)
        return np.asarray(X, dtype=float), u

    def sample(self, n: int, rng: Optional[Rng] = None) -> Tuple[Array, Array]:
        rng = rng_or_default(rng)
        n = int(n)
        class_idx = rng.choice(self.num_classes, size=n, replace=True, p=self._gamma)

        X_out = np.zeros((n, self.p), dtype=float)
        u_out = np.zeros((n, 2), dtype=float)

        for k in range(self.num_classes):
            mask = (class_idx == k)
            nk = int(np.sum(mask))
            if nk == 0:
                continue
            Xk, uk = self.sample_class(k, nk, rng=rng)
            X_out[mask, :] = Xk
            u_out[mask, :] = uk

        return X_out, u_out

    # Keep the user-facing wrapper names from the original snippet (delegating to BaseDataModel impl)
    def error_classif_th(
        self,
        mu: Array,
        alpha: Union[float, Array],
        select_class: Sequence[int] = (),
        num_trials: int = 100000,
        rng: Optional[Rng] = None,
    ) -> float:
        classes = list(select_class) if len(select_class) else list(range(self.num_classes))
        return super().error_classif_th(mu=mu, alpha=alpha, num_trials=num_trials, rng=rng, select_class=classes)

    def error_classif_emp(
        self,
        thetas_rcrd: Sequence[Array],
        select_class: Sequence[int] = (),
        n_test: int = 5000,
        rng: Optional[Rng] = None,
    ) -> float:
        classes = list(select_class) if len(select_class) else list(range(self.num_classes))
        return super().error_classif_emp(thetas_rcrd=thetas_rcrd, n_test=n_test, rng=rng, select_class=classes)
