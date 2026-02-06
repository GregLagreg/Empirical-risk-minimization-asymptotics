

"""
data_models_superlabel.py

Extensions to `data_models.py` for experiments mixing real data and generated data
with pruning, using a *super-label* u = (eps, y).

The core idea:
  - A single ERM objective can handle both real and generated samples by using a
    loss family L_u(z) = eps * ℓ_y(z), where eps is a per-sample pruning/weight.
  - Real samples use eps = 1.
  - Generated samples use eps = ε(x, y) ∈ [0,1] produced by a pruner/oracle.

If the underlying task has K classes (in the sense of the underlying data model's
`num_classes`), the combined model exposes 2K classes:
  - class 0..K-1      : real data
  - class K..2K-1     : generated data
Each class k has its own feature covariance C_k inherited from the corresponding
real / generator model. The *label* returned by this combined model is u=(eps,y),
so y may still depend on x (teacher/student) and eps may depend on (x,y) (pruning).

This file is meant to plug into the existing `erm_theory.py` workflow with the
super-label-aware loss wrapper defined in `losses_regularizers_superlabel.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, Protocol

import numpy as np

from data_models import BaseDataModel, Array


def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(-1)
    n = np.linalg.norm(v)
    return v / (n + 1e-12)

def make_aligned_vector(w_star: np.ndarray, rho: float, rng: np.random.Generator) -> np.ndarray:
    """Return a unit vector w_g with cosine alignment rho with w_star."""
    w_star = unit(w_star)
    p = w_star.size
    # Sample random vector orthogonal to w_star
    v = rng.standard_normal(size=p)
    v = v - (v @ w_star) * w_star
    v = unit(v)
    rho = float(rho)
    rho = np.clip(rho, -1.0, 1.0)
    w_g = rho * w_star + np.sqrt(max(1.0 - rho**2, 0.0)) * v
    return unit(w_g)



class Pruner(Protocol):
    """
    Minimal protocol for pruners that produce per-sample weights eps ∈ [0,1].
    """

    def epsilon(self, X: Array, y: Array, rng: Optional[np.random.Generator] = None) -> Array:
        ...


def _sign_pm1(t: Array) -> Array:
    """
    Map real values to {+1, -1}, with 0 mapped to +1 by convention.
    """
    s = np.sign(t)
    return np.where(s == 0, 1.0, s)


@dataclass
class OracleMarginPruner:
    """
    Oracle / curation rule based on an oracle direction w_o and a margin threshold δ.

    Hard, label-aware version:
        eps(x, y) = 1[ y == sign(x^T w_o) ] · 1[ |x^T w_o| >= δ ].

    Soft relaxation (optional):
        eps(x, y) = 1[ y == sign(x^T w_o) ] · σ( β (|x^T w_o| - δ) ).

    Parameters
    ----------
    w_oracle : array, shape (p,)
        Oracle direction.
    delta : float
        Margin threshold δ >= 0.
    label_aware : bool
        If True, enforce label agreement with the oracle. If False, use only
        the margin condition.
    soft : bool
        If True, use the sigmoid relaxation of the margin threshold.
    beta : float
        Slope for the sigmoid in the soft rule.
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

    def epsilon(self, X: Array, y: Array, rng: Optional[np.random.Generator] = None) -> Array:  # pylint: disable=unused-argument
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        s = X @ self.w_oracle
        margin = np.abs(s)

        if self.soft:
            # sigmoid(beta * (margin - delta))
            t = float(self.beta) * (margin - float(self.delta))
            # stable sigmoid
            soft_gate = 1.0 / (1.0 + np.exp(-t))
            gate = soft_gate
        else:
            gate = (margin >= float(self.delta)).astype(float)

        if self.label_aware:
            y_oracle = _sign_pm1(s)
            agree = (y == y_oracle).astype(float)
            gate = gate * agree

        # clip to [0,1]
        return np.clip(gate, 0.0, 1.0)


@dataclass
class RealGeneratedPrunedModel(BaseDataModel):
    """
    Combined model for "real + generated + pruning" with super-label u=(eps, y).

    The combined distribution is a mixture:
      - with probability gamma_r: draw from real_model
      - with probability gamma_g: draw from gen_model

    The combined model exposes 2K classes if both submodels have K classes:
      - class k in [0, K-1]      : real class k, eps=1
      - class k in [K, 2K-1]     : generated class (k-K), eps = lambda_gen * pruner.epsilon(x,y)

    Notes
    -----
    - We do NOT *drop* samples: pruning is represented as a weight eps ∈ [0,1]
      stored in u. If eps=0, that sample contributes zero to the ERM loss.
    - The feature covariances per class are inherited from the underlying models
      (pruning does not change x's distribution; it only reweights the loss).
    """
    real_model: BaseDataModel
    gen_model: BaseDataModel
    pruner: Optional[Pruner] = None
    gamma_r: float = 0.5
    lambda_gen: float = 1.0

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
        # Normalize defensively (should already sum to 1)
        self._gamma = self._gamma / float(np.sum(self._gamma))

        # Mus / covariances for x, per class
        self._mus = [np.asarray(m, dtype=float).reshape(-1) for m in params_r.get("mus", [])] + \
                    [np.asarray(m, dtype=float).reshape(-1) for m in params_g.get("mus", [])]
        self._covs = [np.asarray(C, dtype=float) for C in params_r.get("covs", [])] + \
                     [np.asarray(C, dtype=float) for C in params_g.get("covs", [])]


        if len(self._covs) != 2 * self.K_base:
            # As a fallback, if submodels do not provide per-class covariances,
            # we approximate with the unconditional covariance via sampling.
            # But the standard models in `data_models.py` do provide them.
            raise ValueError("Expected both submodels to provide per-class covariances in class_params()['covs'].")

    @property
    def num_classes(self) -> int:
        return 2 * self.K_base

    def class_params(self) -> Dict[str, Any]:
        return dict(
            p=self.p,
            num_classes=self.num_classes,
            gamma=self._gamma,
            mus=self._mus,
            covs=self._covs,
            # Super-label u=(eps,y) is not a scalar label; indicate non-deterministic labels.
            y_values=None,
            w_values = self.K_base*[self.real_model.theta_teacher] + self.K_base*[self.gen_model.theta_teacher]
        )

    def _make_u(self, eps: Array, y: Array) -> Array:
        eps = np.asarray(eps, dtype=float).reshape(-1)
        y = np.asarray(y, dtype=float).reshape(-1)
        if eps.shape[0] != y.shape[0]:
            raise ValueError("eps and y must have the same length")
        return np.stack([eps, y], axis=1)  # shape (n,2)

    def sample_class(
        self,
        class_index: int,
        n: int,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[Array, Array]:
        rng = np.random.default_rng() if rng is None else rng
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

    def sample(self, n: int, rng: Optional[np.random.Generator] = None) -> Tuple[Array, Array]:
        rng = np.random.default_rng() if rng is None else rng
        n = int(n)
        # Sample class indices according to gamma
        class_idx = rng.choice(self.num_classes, size=n, replace=True, p=self._gamma)

        X_out = np.zeros((n, self.p), dtype=float)
        u_out = np.zeros((n, 2), dtype=float)

        # Sample class-wise for efficiency
        for k in range(self.num_classes):
            mask = (class_idx == k)
            nk = int(np.sum(mask))
            if nk == 0:
                continue
            Xk, uk = self.sample_class(k, nk, rng=rng)
            X_out[mask, :] = Xk
            u_out[mask, :] = uk

        return X_out, u_out
    
    def error_classif_th(self,mu: Array, alpha: Array, select_class = [], num_trials=100000, rng: Optional[np.random.Generator] = None) -> float:
        if len(select_class)==0:
            select_class = list(range(self.num_classes))
        err=0
        for k in select_class:
            if k < self.K_base:
                model = self.real_model
            else:
                model = self.gen_model
            err += self._gamma[k] * model.error_classif_th(mu=mu, alpha=alpha[k], rng=rng)
        return float(err)
    def error_classif_emp(self,thetas_rcrd: Sequence[Array], select_class = [], n_test=5000, rng: Optional[np.random.Generator] = None) -> float:
        if len(select_class)==0:
            select_class = list(range(self.num_classes))
        err=0
        for k in select_class:
            if k < self.K_base:
                model = self.real_model
            else:
                model = self.gen_model
            err += self._gamma[k] * model.error_classif_emp(thetas_rcrd, n_test=n_test, rng=rng)
        return float(err)

        
