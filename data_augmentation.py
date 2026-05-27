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
class DataAugmTrainer:

    model: BaseDataModel
    loss: Loss
    regularizer: Regularizer
    transform: function # La transformation aléatoire
    K_augm: int = 40

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

        X_augm_list = []
        for i in range(self.K_augm):
            X_augm = self.transform(X)
            X_augm_list.append(X_augm)

        def obj(theta):
            z_augm = np.zeros(n)
            for X_augm in X_augm_list:
                z_augm += self.loss.value(X_augm @ theta, y)
            z_augm /= self.K_augm
            return float(np.mean(z_augm) + self.regularizer.value(theta))

        def grad(theta):
            g = np.zeros(p)
            for X_augm in X_augm_list:
                z = X_augm @ theta
                g_z = self.loss.grad(z, y)
                g += (X_augm.T @ g_z) / n
            g /= self.K_augm
            return np.asarray(g + self.regularizer.grad(theta), dtype=float)

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
            thetas = self.thetas,
            mean_theta=mean_theta,
            cov_theta=cov_theta,
            gen_losses=losses,
            gen_loss_mean=float(np.mean(losses)),
            gen_loss_std=float(np.std(losses, ddof=1)) if num_trials > 1 else 0.0,
        )
