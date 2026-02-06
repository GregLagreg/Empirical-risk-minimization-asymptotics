
"""
erm_theory.py

Empirical ERM training and a simple fixed-point Monte Carlo solver for the
theoretical quantities (mu_*, alpha, kappa, nu, Q) described in the prompt.

This module depends on:
  - data_models.py
  - losses_regularizers.py

Core classes
------------
1) ERMTrainer:
   - Samples (X,y) from a model.
   - Solves the convex ERM:
        minimize  (1/n) Σ_i L_{y_i}(x_i^T θ) + ρ(θ)
   - Evaluates generalization performance on fresh test samples.

2) TheoryFixedPointSolver:
   - Implements a damped fixed-point iteration for the system:
        Q(ν) = (Σ_k γ_k ν_k C_k + ∇^2ρ(μ_*))^{-1}
        κ_k  = (1/n) tr(C_k Q)
        ν_k  = (1/κ_k) E[ z r(μ_*^T x_k + α_k z, κ_k, y_k ) ]
        α_k^2 = Σ_l γ_l/(n κ_l^2) tr(C_k Q C_l Q) E[ r(...)^2 ]
        ∇ρ(μ_*) = - Σ_k γ_k/κ_k E[ x_k r(...) ]

   - All expectations are approximated by Monte Carlo with fixed samples
     (to reduce variance across iterations).

IMPORTANT practical notes:
  - This is an experimental/numerical solver, not a proof-grade implementation.
  - Convergence is not guaranteed for arbitrary settings; damping helps.
  - For pseudo-Huber regularization, the Hessian depends on μ_* (diagonal).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any

import numpy as np
from numpy.linalg import LinAlgError
from scipy.optimize import minimize

from data_models import BaseDataModel
from losses_regularizers import Loss, Regularizer

Array = np.ndarray


def _sym(M: Array) -> Array:
    return 0.5 * (M + M.T)


def spd_inverse(M: Array, jitter: float = 1e-12) -> Array:
    """
    Invert a symmetric positive (semi-)definite matrix using Cholesky with jitter.
    """
    M = _sym(np.asarray(M, dtype=float))
    p = M.shape[0]
    I = np.eye(p)
    for k in range(7):
        try:
            L = np.linalg.cholesky(M + (10.0 ** k) * jitter * I)
            # Solve L L^T X = I
            X = np.linalg.solve(L.T, np.linalg.solve(L, I))
            return _sym(X)
        except LinAlgError:
            continue
    # Fall back to generic inverse
    return np.linalg.inv(M + 1e-6 * I)


@dataclass
class ERMTrainer:
    """
    Empirical ERM solver and evaluator.
    """
    model: BaseDataModel
    loss: Loss
    regularizer: Regularizer

    def solve_theta_hat(
        self,
        X: Array,
        y: Array,
        theta0: Optional[Array] = None,
        solver_maxiter: int = 500,
        tol: float = 1e-9,
        method: str = "L-BFGS-B",
        verbose: bool = False,
    ) -> Array:
        """
        Solve the ERM given a training set (X,y).

        Uses scipy.optimize.minimize.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, p = X.shape
        if y.shape[0] != n:
            raise ValueError("X and y must have compatible shapes")
        if theta0 is None:
            theta0 = np.zeros(p, dtype=float)
        else:
            theta0 = np.asarray(theta0, dtype=float).reshape(-1)
            if theta0.shape[0] != p:
                raise ValueError("theta0 must have shape (p,)")

        def obj(theta: Array) -> float:
            z = X @ theta
            return float(np.mean(self.loss.value(z, y)) + self.regularizer.value(theta))

        def grad(theta: Array) -> Array:
            z = X @ theta
            g_z = self.loss.grad(z, y)  # shape (n,)
            g = (X.T @ g_z) / n + self.regularizer.grad(theta)
            return np.asarray(g, dtype=float)

        res = minimize(
            fun=obj,
            x0=theta0,
            jac=grad,
            method=method,
            options=dict(maxiter=solver_maxiter, ftol=tol, disp=verbose),
        )
        if not res.success and verbose:
            print("Optimization did not fully converge:", res.message)
        return np.asarray(res.x, dtype=float)

    def sample_theta_hat(
        self,
        n: int,
        rng: Optional[np.random.Generator] = None,
        **solver_kwargs: Any,
    ) -> Tuple[Array, Array, Array]:
        """
        Sample a dataset (X,y) and return (theta_hat, X, y).
        """
        rng = np.random.default_rng() if rng is None else rng
        X, y = self.model.sample(n, rng=rng)
        theta_hat = self.solve_theta_hat(X, y, **solver_kwargs)
        return theta_hat, X, y

    def generalization_error(
        self,
        theta: Array,
        n_test: int = 1,
        rng: Optional[np.random.Generator] = None,
    ) -> float:
        """
        Estimate E[L_y(x^T theta)] by Monte Carlo on a fresh test set.
        """
        rng = np.random.default_rng() if rng is None else rng
        Xte, yte = self.model.sample(n_test, rng=rng)
        z = Xte @ np.asarray(theta, dtype=float)
        return float(np.mean(self.loss.value(z, yte)))

    def run_trials(
        self,
        n_train: int,
        n_test: int,
        num_trials: int,
        rng: Optional[np.random.Generator] = None,
        **solver_kwargs: Any,
    ) -> Dict[str, Array]:
        """
        Run multiple independent ERM trainings to estimate:
          - mean(theta_hat)
          - cov(theta_hat)
          - mean generalization loss
        """
        # n_test=1
        rng = np.random.default_rng() if rng is None else rng
        self.thetas = []
        losses = []
        for _ in range(int(num_trials)):
            theta_hat, _, _ = self.sample_theta_hat(n_train, rng=rng, **solver_kwargs)
            gen = self.generalization_error(theta_hat, n_test=n_test, rng=rng)
            self.thetas.append(theta_hat)
            losses.append(gen)
        self.thetas = np.stack(self.thetas, axis=0)  # (T,p)
        losses = np.asarray(losses, dtype=float)
        mean_theta = np.mean(self.thetas, axis=0)
        cov_theta = np.cov(self.thetas, rowvar=False, bias=False)
        return dict(
            mean_theta=mean_theta,
            cov_theta=cov_theta,
            gen_losses=losses,
            gen_loss_mean=float(np.mean(losses)),
            gen_loss_std=float(np.std(losses, ddof=1)) if num_trials > 1 else 0.0,
        )



@dataclass
class TheoryFixedPointSolver:
    """
    Damped fixed-point solver for (mu_*, alpha, kappa, nu, Q) with Monte Carlo
    expectations.

    Parameters
    ----------
    model : BaseDataModel
    loss : Loss
    regularizer : Regularizer
    n_train : int
        The n used in the theoretical equations (training sample size).
    mc_samples : int
        Monte Carlo samples used to approximate expectations per class.
    rng : numpy Generator
        Random generator used to pre-sample Monte Carlo (X,z) per class.
    """
    model: BaseDataModel
    loss: Loss
    regularizer: Regularizer
    n_train: int
    mc_samples: int = 5000
    rng: Optional[np.random.Generator] = None

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng() if self.rng is None else self.rng
        params = self.model.class_params()
        self.p = int(params["p"])
        self.K = int(params["num_classes"])
        self.gamma = np.asarray(params["gamma"], dtype=float).reshape(-1)
        self.mus = [np.asarray(C, dtype=float) for C in params["mus"]]
        self.covs = [np.asarray(C, dtype=float) for C in params["covs"]]
        y_values = params.get("y_values", None)
        if y_values is None:
            # Teacher/student: y varies with x; for the theory we treat it as single-class
            # and will *resample y* each time by sampling from the model (slower).
            # For mixture classification models we expect y_values to exist.
            self.y_values = None
        else:
            self.y_values = [float(v) for v in y_values]

        # Pre-sample Monte Carlo features per class to stabilize iterations.
        # For mixture classification: y is deterministic given class, so we store it.
        # For single-class teacher/student: we still pre-sample X and z, but y will be
        # resampled when needed (because y depends on x with teacher noise).
        self.X_mc = []
        self.y_mc = []
        self.z_mc = []
        for k in range(self.K):
            Xk, yk = self.model.sample_class(k, self.mc_samples, rng=self.rng)
            zk = self.rng.standard_normal(size=self.mc_samples)
            self.X_mc.append(np.asarray(Xk, dtype=float))
            self.y_mc.append(np.asarray(yk, dtype=float))
            self.z_mc.append(np.asarray(zk, dtype=float))

    def _moments_for_class(
        self, k: int, mu: Array, alpha_k: float, kappa_k: float
    ) -> Dict[str, Array]:
        """
        Return E[r^2], E[r x], E[r z] for class k using fixed MC samples.
        """
        Xk = self.X_mc[k]
        yk = self.y_mc[k]
        zk = self.z_mc[k]
        return self.loss.estimate_r_moments(
            X=Xk, y=yk, mu=mu, alpha=alpha_k, kappa=kappa_k, z_samples=zk, rng=self.rng
        )

    def _expected_loss_for_class(
        self, k: int, mu: Array, alpha_k: float
    ) -> float:
        Xk = self.X_mc[k]
        yk = self.y_mc[k]
        zk = self.z_mc[k]
        return self.loss.estimate_expected_loss(
            X=Xk, y=yk, mu=mu, alpha=alpha_k, z_samples=zk, rng=self.rng
        )
    def _expected_loss_for_class_gaussian_score(
        self, k: int, mu: Array, alpha_k: float
    ) -> float:
        return self.loss.estimate_expected_loss_gaussian_score(
            muk=self.mus[k], Ck=self.covs[k], y=self.y_mc[k], mu=mu, alpha=alpha_k, z_samples=self.z_mc[k], rng=self.rng
        )



    def solve(
        self,
        max_iter: int = 100,
        tol: float = 1e-6,
        damping: float = 0.1,
        verbose: bool = False,
        mu0: Optional[Array] = None,
        alpha0: Optional[Array] = None,
        nu0: Optional[Array] = None,
    ) -> Dict[str, Any]:
        """
        Run the fixed-point iterations.

        Returns a dictionary with keys:
          - mu_star: array (p,)
          - alpha: array (K,)
          - kappa: array (K,)
          - nu: array (K,)
          - Q: array (p,p)
          - predicted_loss: float
          - converged: bool
          - num_iter: int
        """
        p, K = self.p, self.K
        n = float(self.n_train)

        mu = np.zeros(p, dtype=float) if mu0 is None else np.asarray(mu0, dtype=float).reshape(-1).copy()
        alpha = np.ones(K, dtype=float) if alpha0 is None else np.asarray(alpha0, dtype=float).reshape(-1).copy()
        nu = np.ones(K, dtype=float) if nu0 is None else np.asarray(nu0, dtype=float).reshape(-1).copy()

        converged = False
        Q = np.eye(p)
        kappa = np.ones(K, dtype=float)

        for it in range(int(max_iter)):
            if it%10 == 0 and verbose:
                print(f"[FP] Starting iter {it:03d}")
            # Q(ν) = (Σ γ_k ν_k C_k + ∇^2ρ(μ))^{-1}
            H_mu = self.regularizer.hessian(mu)
            S = H_mu.copy()
            for k in range(K):
                S = S + self.gamma[k] * nu[k] * self.covs[k]
            Q_new = spd_inverse(S)

            # κ_k = (1/n) tr(C_k Q)
            kappa_new = np.zeros(K, dtype=float)
            B = []
            for k in range(K):
                Bk = self.covs[k] @ Q_new
                B.append(Bk)
                kappa_new[k] = float(np.trace(Bk) / n)

            # Moments per class
            E_r2 = np.zeros(K, dtype=float)
            E_rz = np.zeros(K, dtype=float)
            E_rx = np.zeros((K, p), dtype=float)
            for k in range(K):
                moms = self._moments_for_class(k, mu=mu, alpha_k=alpha[k], kappa_k=kappa_new[k])
                E_r2[k] = float(moms["E_r2"])
                E_rz[k] = float(moms["E_rz"])
                E_rx[k, :] = np.asarray(moms["E_rx"], dtype=float)

            # ν_k update
            nu_new = E_rz / (kappa_new*alpha)

            # trace terms tr(C_k Q C_l Q) = tr((C_k Q)(C_l Q)) = <B_k, B_l^T>_F
            trace_terms = np.zeros((K, K), dtype=float)
            for k in range(K):
                for l in range(K):
                    trace_terms[k, l] = float(np.sum(B[k] * B[l].T))

            # α_k^2 update
            alpha_sq_new = np.zeros(K, dtype=float)
            for k in range(K):
                s = 0.0
                for l in range(K):
                    s += (
                        self.gamma[l]
                        * trace_terms[k, l]
                        * E_r2[l]
                        / (n * (kappa_new[l] ** 2))
                    )
                alpha_sq_new[k] = max(s, 0.0)
            alpha_new = np.sqrt(alpha_sq_new)

            # μ update via ∇ρ(μ) = - Σ γ_k/κ_k E[x r]
            b = np.zeros(p, dtype=float)
            for k in range(K):
                b += (self.gamma[k] / kappa_new[k]) * E_rx[k, :]
            b = -b
            try:
                mu_new = self.regularizer.solve_grad_eq(b)
            except Exception as e:  # pragma: no cover
                raise RuntimeError(
                    "regularizer.solve_grad_eq failed. Provide a regularizer with an "
                    "invertible gradient map or implement a numerical solver."
                ) from e

            # Damping
            d = float(damping)
            # d_it = d
            d_it = d/((float(it + 1))**0.5)
            mu_upd = (1.0 - d_it) * mu + d_it * mu_new
            alpha_upd = (1.0 - d_it) * alpha + d_it * alpha_new
            nu_upd = (1.0 - d_it) * nu + d_it * nu_new

            # Convergence check
            delta = max(
                float(np.linalg.norm(mu_upd - mu)),
                float(np.max(np.abs(alpha_upd - alpha))),
                float(np.max(np.abs(nu_upd - nu))),
            )
            mu, alpha, nu = mu_upd, alpha_upd, nu_upd
            Q, kappa = Q_new, kappa_new

            if verbose:
                print(f"[FP] iter={it:03d}  delta={delta:.3e}  "
                      f"||mu||={np.linalg.norm(mu):.3e}  "
                      f"alpha={alpha}  kappa={kappa}")

            if delta < d_it*tol and it >= 9:
                converged = True
                break

        # Predicted generalization loss
        pred_loss = 0.0
        pred_loss_gauss_score = 0.0
        for k in range(K):
            pred_loss += self.gamma[k] * self._expected_loss_for_class(k, mu=mu, alpha_k=alpha[k])
            pred_loss_gauss_score += self.gamma[k] * self._expected_loss_for_class_gaussian_score(k, mu=mu, alpha_k=alpha[k])

        return dict(
            mu=mu,
            alpha=alpha,
            kappa=kappa,
            nu=nu,
            Q=Q,
            A = trace_terms,
            predicted_loss=float(pred_loss),
            pred_loss_gauss_score = float(pred_loss_gauss_score),
            converged=converged,
            num_iter=it + 1,
            damping_final = d_it,
        )
