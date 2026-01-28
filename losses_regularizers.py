
"""
losses_regularizers.py

Loss and regularization classes for ERM.

This module provides:
  - Convex losses: squared (L2), logistic, hinge (max-margin).
  - Convex regularizers: quadratic (affine + quadratic), pseudo-Huber.

Each Loss provides:
  - value(z, y): compute L_y(z) elementwise for arrays
  - grad(z, y): derivative d/dz L_y(z)
  - prox(z, kappa, y): prox_{kappa L_y}(z) elementwise (vectorized)
  - r(z, kappa, y): z - prox_{kappa L_y}(z)
  - Monte Carlo estimators for E[r^2], E[r x], E[r z] for use in theory.

Notes / design choices (important):
  1) For numerical stability, logistic loss uses scipy.special.expit.
  2) Logistic prox is computed by Newton iterations; this is fast and robust
     because the 1D objective is strongly convex (quadratic term + convex loss).
  3) Hinge loss is non-smooth; we provide a subgradient in grad(). This is
     sufficient for many first-order solvers but will not make the objective
     everywhere differentiable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union, Sequence, Dict

import numpy as np
from scipy.special import expit

Array = np.ndarray


# ----------------------------
# Regularizers
# ----------------------------

class Regularizer:
    def value(self, theta: Array) -> float:
        raise NotImplementedError

    def grad(self, theta: Array) -> Array:
        raise NotImplementedError

    def hessian(self, theta: Array) -> Array:
        """
        Return a (p,p) Hessian matrix at theta.

        For non-quadratic regularizers, this is used by the theory solver
        in Q(ν) = (Σ γ_k ν_k C_k + ∇^2 ρ(μ_*))^{-1}.
        """
        raise NotImplementedError

    def hessian_at_zero(self, p: int) -> Array:
        return self.hessian(np.zeros(p, dtype=float))

    def solve_grad_eq(self, b: Array) -> Array:
        """
        Solve ∇ρ(theta) = b for theta.

        This is used as a fixed-point update in the theory solver.

        Not all regularizers have a closed form inverse of the gradient map.
        In those cases, you can override this method with a numerical solver.
        """
        raise NotImplementedError


@dataclass
class QuadraticRegularizer(Regularizer):
    """
    ρ(θ) = a^T θ + (1/2) θ^T H θ

    - H must be symmetric positive semidefinite for convexity.
    - ∇ρ(θ) = a + H θ
    - ∇^2ρ(θ) = H

    IMPORTANT: we use the (1/2) convention so that Hessian equals H.
    If you intended ρ(θ)=a^T θ + θ^T H θ, then your Hessian is 2H.
    """
    a: Array
    H: Array

    def __post_init__(self) -> None:
        self.a = np.asarray(self.a, dtype=float).reshape(-1)
        self.H = np.asarray(self.H, dtype=float)
        if self.H.shape != (self.a.shape[0], self.a.shape[0]):
            raise ValueError("H must have shape (p,p) consistent with a")
        # Symmetrize for safety
        self.H = 0.5 * (self.H + self.H.T)

    def value(self, theta: Array) -> float:
        theta = np.asarray(theta, dtype=float).reshape(-1)
        return float(self.a @ theta + 0.5 * theta @ (self.H @ theta))

    def grad(self, theta: Array) -> Array:
        theta = np.asarray(theta, dtype=float).reshape(-1)
        return self.a + self.H @ theta

    def hessian(self, theta: Array) -> Array:  # pylint: disable=unused-argument
        return self.H

    def solve_grad_eq(self, b: Array) -> Array:
        b = np.asarray(b, dtype=float).reshape(-1)
        # Solve a + H θ = b  =>  H θ = (b - a)
        return np.linalg.solve(self.H, b - self.a)


@dataclass
class PseudoHuberRegularizer(Regularizer):
    """
    ρ(θ) = λ δ^2 Σ_i ( sqrt(1 + (θ_i/δ)^2 ) - 1 )

    This is a smooth convex approximation of the ℓ1 norm:
      for |θ_i| >> δ, gradient saturates at ±λ δ.

    Scalar derivatives:
      d/dt  ρ_i(t) = λ * t / sqrt(1 + (t/δ)^2)
      d^2/dt^2 ρ_i(t) = λ / (1 + (t/δ)^2)^{3/2}

    Hessian is diagonal.
    """
    lam: float = 1.0
    delta: float = 1.0

    def value(self, theta: Array) -> float:
        theta = np.asarray(theta, dtype=float).reshape(-1)
        d = float(self.delta)
        return float(self.lam * d * d * np.sum(np.sqrt(1.0 + (theta / d) ** 2) - 1.0))

    def grad(self, theta: Array) -> Array:
        theta = np.asarray(theta, dtype=float).reshape(-1)
        d = float(self.delta)
        return self.lam * theta / np.sqrt(1.0 + (theta / d) ** 2)

    def hessian(self, theta: Array) -> Array:
        theta = np.asarray(theta, dtype=float).reshape(-1)
        d = float(self.delta)
        diag = self.lam / (1.0 + (theta / d) ** 2) ** (1.5)
        return np.diag(diag)

    def solve_grad_eq(self, b: Array) -> Array:
        """
        Solve λ t / sqrt(1 + (t/δ)^2) = b elementwise.

        This has a closed-form solution when |b| < λ δ:
          t = (b/λ) / sqrt(1 - (b/(λ δ))^2)

        If |b| approaches λ δ, t diverges; we clip b slightly to keep finite.
        """
        b = np.asarray(b, dtype=float).reshape(-1)
        lam = float(self.lam)
        d = float(self.delta)
        if lam <= 0 or d <= 0:
            raise ValueError("lam and delta must be positive for pseudo-huber")
        # Clip to ensure |b| < lam*d
        bound = lam * d
        eps = 1e-12
        b_clipped = np.clip(b, -bound * (1.0 - eps), bound * (1.0 - eps))
        u = b_clipped / lam
        denom = np.sqrt(1.0 - (b_clipped / bound) ** 2)
        return u / denom


# ----------------------------
# Losses
# ----------------------------

class Loss:
    """
    Base class for scalar losses L_y(z), where z = x^T θ is scalar.

    All methods should support numpy arrays for z and y.
    """

    def value(self, z: Array, y: Array) -> Array:
        raise NotImplementedError

    def grad(self, z: Array, y: Array) -> Array:
        """
        Subgradient / gradient with respect to z.
        """
        raise NotImplementedError

    def prox(self, z: Array, kappa: float, y: Array) -> Array:
        """
        prox_{kappa L_y}(z) elementwise.
        """
        raise NotImplementedError

    def r(self, z: Array, kappa: float, y: Array) -> Array:
        """
        r(z,kappa,y) = z - prox_{kappa L_y}(z).
        """
        return np.asarray(z, dtype=float) - self.prox(z, kappa, y)

    # --- Monte Carlo estimators useful for theory ---

    def estimate_r_moments(
        self,
        *,
        X: Array,
        y: Array,
        mu: Array,
        alpha: float,
        kappa: float,
        z_samples: Optional[Array] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> Dict[str, Array]:
        """
        Estimate E[r^2], E[r x], E[r z] using provided samples (X, y) and either
        provided z_samples or fresh Gaussian z.

        Inputs
        ------
        X : array, shape (T,p)
        y : array, shape (T,)
        mu : array, shape (p,)
        alpha : float
        kappa : float
        z_samples : array, shape (T,), optional
        rng : numpy Generator, optional

        Returns
        -------
        dict with:
          - E_r2 : float
          - E_rx : array shape (p,)
          - E_rz : float
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        mu = np.asarray(mu, dtype=float).reshape(-1)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have same number of samples")
        T, p = X.shape
        if mu.shape[0] != p:
            raise ValueError("mu must have shape (p,)")
        if rng is None:
            rng = np.random.default_rng()
        if z_samples is None:
            z_samples = rng.standard_normal(size=T)
        else:
            z_samples = np.asarray(z_samples, dtype=float).reshape(-1)
            if z_samples.shape[0] != T:
                raise ValueError("z_samples must have length T")

        u = X @ mu + float(alpha) * z_samples
        r_vals = self.r(u, float(kappa), y)

        E_r2 = float(np.mean(r_vals ** 2))
        E_rx = np.mean(r_vals[:, None] * X, axis=0)
        E_rz = float(np.mean(r_vals * z_samples))
        return dict(E_r2=E_r2, E_rx=E_rx, E_rz=E_rz)

    def estimate_expected_loss(
        self,
        *,
        X: Array,
        y: Array,
        mu: Array,
        alpha: float,
        z_samples: Optional[Array] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> float:
        """
        Estimate E[L_y( mu^T x + alpha z )] using provided samples (X, y) and z.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        mu = np.asarray(mu, dtype=float).reshape(-1)
        T, p = X.shape
        if mu.shape[0] != p:
            raise ValueError("mu must have shape (p,)")
        if rng is None:
            rng = np.random.default_rng()
        if z_samples is None:
            z_samples = rng.standard_normal(size=T)
        else:
            z_samples = np.asarray(z_samples, dtype=float).reshape(-1)
            if z_samples.shape[0] != T:
                raise ValueError("z_samples must have length T")
        u = X @ mu + float(alpha) * z_samples
        return float(np.mean(self.value(u, y)))


    def estimate_r_moments_under_model(
        self,
        model: object,
        *,
        mu: Array,
        alpha: float,
        kappa: float,
        class_index: Optional[int] = None,
        T: int = 10000,
        rng: Optional[np.random.Generator] = None,
    ) -> Dict[str, Array]:
        """
        Convenience wrapper: sample (X,y) from a model and estimate
        E[r^2], E[r x], E[r z].

        The model is expected to implement:
          - sample(n, rng=...) -> (X,y)
          - and optionally sample_class(k, n, rng=...) -> (X,y)
        """
        rng = np.random.default_rng() if rng is None else rng
        T = int(T)
        if class_index is None:
            X, y = model.sample(T, rng=rng)
        else:
            X, y = model.sample_class(int(class_index), T, rng=rng)
        z_samples = rng.standard_normal(size=T)
        return self.estimate_r_moments(
            X=X, y=y, mu=mu, alpha=alpha, kappa=kappa, z_samples=z_samples, rng=rng
        )

    def estimate_expected_loss_under_model(
        self,
        model: object,
        *,
        mu: Array,
        alpha: float,
        class_index: Optional[int] = None,
        T: int = 10000,
        rng: Optional[np.random.Generator] = None,
    ) -> float:
        """
        Convenience wrapper: sample (X,y) from a model and estimate
        E[L_y(mu^T x + alpha z)].
        """
        rng = np.random.default_rng() if rng is None else rng
        T = int(T)
        if class_index is None:
            X, y = model.sample(T, rng=rng)
        else:
            X, y = model.sample_class(int(class_index), T, rng=rng)
        z_samples = rng.standard_normal(size=T)
        return self.estimate_expected_loss(
            X=X, y=y, mu=mu, alpha=alpha, z_samples=z_samples, rng=rng
        )


@dataclass
class SquaredLoss(Loss):
    """
    L_y(z) = 0.5 (z - y)^2
    """
    def value(self, z: Array, y: Array) -> Array:
        z = np.asarray(z, dtype=float)
        y = np.asarray(y, dtype=float)
        return 0.5 * (z - y) ** 2

    def grad(self, z: Array, y: Array) -> Array:
        z = np.asarray(z, dtype=float)
        y = np.asarray(y, dtype=float)
        return z - y

    def prox(self, z: Array, kappa: float, y: Array) -> Array:
        z = np.asarray(z, dtype=float)
        y = np.asarray(y, dtype=float)
        k = float(kappa)
        return (z + k * y) / (1.0 + k)


@dataclass
class LogisticLoss(Loss):
    """
    Binary logistic loss with labels y ∈ {+1, -1}:

      L_y(z) = log(1 + exp(-y z))

    Prox is computed by Newton iterations.
    """
    max_iter: int = 60
    tol: float = 1e-10

    def value(self, z: Array, y: Array) -> Array:
        z = np.asarray(z, dtype=float)
        y = np.asarray(y, dtype=float)
        t = y * z
        # log(1+exp(-t)) in a stable way:
        # Use softplus: log(1+exp(u)) = logaddexp(0,u)
        return np.logaddexp(0.0, -t)

    def grad(self, z: Array, y: Array) -> Array:
        z = np.asarray(z, dtype=float)
        y = np.asarray(y, dtype=float)
        t = y * z
        # d/dz log(1+exp(-t)) = -y * sigmoid(-t)
        return -y * expit(-t)

    def prox(self, z: Array, kappa: float, y: Array) -> Array:
        """
        Compute prox_{kappa * logistic_y}(z) elementwise using Newton.

        Solve for w:
          0 = (w - z) - kappa * y / (1 + exp(y w))
        """
        z = np.asarray(z, dtype=float)
        y = np.asarray(y, dtype=float)
        k = float(kappa)

        # Initial guess: w0 = z
        w = z.copy()

        # Vectorized Newton iterations
        for _ in range(self.max_iter):
            t = y * w
            sig_t = expit(t)          # sigmoid(t)
            h = expit(-t)             # 1/(1+exp(t))
            g = (w - z) - k * y * h
            gprime = 1.0 + k * sig_t * (1.0 - sig_t)

            step = g / gprime
            w_new = w - step

            if np.max(np.abs(w_new - w)) < self.tol:
                w = w_new
                break
            w = w_new

        return w


@dataclass
class HingeLoss(Loss):
    """
    Hinge (max-margin) loss with labels y ∈ {+1, -1}:

      L_y(z) = max(0, 1 - y z)

    prox is available in closed form.
    """
    def value(self, z: Array, y: Array) -> Array:
        z = np.asarray(z, dtype=float)
        y = np.asarray(y, dtype=float)
        return np.maximum(0.0, 1.0 - y * z)

    def grad(self, z: Array, y: Array) -> Array:
        """
        Subgradient w.r.t z.
          -y if y z < 1
           0 if y z > 1
           0 at the kink (y z == 1) by convention.
        """
        z = np.asarray(z, dtype=float)
        y = np.asarray(y, dtype=float)
        yz = y * z
        return np.where(yz < 1.0, -y, 0.0)

    def prox(self, z: Array, kappa: float, y: Array) -> Array:
        """
        prox for hinge can be derived by reducing to u = y w, s = y z.

        For h(u)=max(0,1-u), prox_{kappa h}(s):
          if s < 1 - kappa: u = s + kappa
          elif 1 - kappa <= s <= 1: u = 1
          else: u = s

        Then w = y u.
        """
        z = np.asarray(z, dtype=float)
        y = np.asarray(y, dtype=float)
        k = float(kappa)

        s = y * z
        u = np.empty_like(s)

        u = np.where(s < 1.0 - k, s + k, u)
        u = np.where((s >= 1.0 - k) & (s <= 1.0), 1.0, u)
        u = np.where(s > 1.0, s, u)

        return y * u
