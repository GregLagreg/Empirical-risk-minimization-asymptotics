
"""
data_models.py

Data model classes for generating synthetic datasets (X, y) for empirical
risk minimization (ERM) experiments.

The design supports:
  - Teacher/student models with a single feature distribution and a teacher
    mechanism producing y.
  - Mixture/class-conditional classification models with K classes, each having
    its own (mu_k, C_k) and class proportion gamma_k.

Notes / design choices (important):
  1) "Uniform with independent entries" can only match a *diagonal* covariance
     matrix. If you pass a non-diagonal covariance, the model will default to
     an affine-uniform construction that matches the full covariance but does
     not yield independent coordinates. This is explained in the docstrings.
  2) For binary classification losses (logistic, hinge), labels should be
     y ∈ {+1, -1}. The MixtureClassificationModel allows any y_values, but your
     chosen Loss must be compatible with them.
"""

from __future__ import annotations

from dataclasses import dataclass,field
from typing import Callable, Literal, Optional, Sequence, Tuple, Union, List

import numpy as np

Array = np.ndarray


def exponential_covariance(n, rho=0.5):
    """
    Create covariance matrix with exponential decay:
    Σ_ij = σ_i * σ_j * ρ^|i-j|
    
    Parameters:
    n: dimension of the matrix
    rho: correlation parameter (0 < rho < 1)
    """
    # Create the first row/column of the covariance matrix
    indices = np.arange(n)
    cov = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            cov[i, j] = rho ** abs(i - j)
    
    return cov
def _as_1d(x: Union[Array, Sequence[float]], p: int, name: str) -> Array:
    x = np.asarray(x, dtype=float).reshape(-1)
    if x.shape[0] != p:
        raise ValueError(f"{name} must have shape (p,), got {x.shape} with p={p}")
    return x


def _as_2d(x: Union[Array, Sequence[Sequence[float]]], p: int, name: str) -> Array:
    x = np.asarray(x, dtype=float)
    if x.shape != (p, p):
        raise ValueError(f"{name} must have shape (p,p)=({p},{p}), got {x.shape}")
    return x


def _symmetrize(M: Array) -> Array:
    return 0.5 * (M + M.T)


def _is_diagonal(M: Array, tol: float = 1e-12) -> bool:
    return np.allclose(M, np.diag(np.diag(M)), atol=tol, rtol=0.0)


def _safe_cholesky(C: Array, jitter: float = 1e-12) -> Array:
    """
    Cholesky factorization with a tiny diagonal jitter for numerical stability.
    Raises if matrix is not numerically SPD.
    """
    C = _symmetrize(C)
    try:
        return np.linalg.cholesky(C)
    except np.linalg.LinAlgError:
        # Add jitter progressively
        for k in range(6):
            try:
                return np.linalg.cholesky(C + (10.0 ** k) * jitter * np.eye(C.shape[0]))
            except np.linalg.LinAlgError:
                continue
        raise


def sample_gaussian(
    n: int, mu: Array, C: Array, rng: np.random.Generator
) -> Array:
    """Sample n vectors from N(mu, C)."""
    mu = np.asarray(mu, dtype=float).reshape(-1)
    C = _symmetrize(np.asarray(C, dtype=float))
    return rng.multivariate_normal(mean=mu, cov=C, size=n)


def sample_uniform_iid_from_mean_var(
    n: int, mu: Array, var: Array, rng: np.random.Generator
) -> Array:
    """
    Sample with independent coordinates, each Uniform[a_j, b_j], matching
    mean mu_j and variance var_j.

    For Uniform[a,b], mean=(a+b)/2, var=(b-a)^2/12.
    => choose a_j = mu_j - sqrt(3 var_j), b_j = mu_j + sqrt(3 var_j).
    """
    mu = np.asarray(mu, dtype=float).reshape(-1)
    var = np.asarray(var, dtype=float).reshape(-1)
    if np.any(var < 0):
        raise ValueError("variances must be nonnegative")
    half_width = np.sqrt(3.0 * var)
    low = mu - half_width
    high = mu + half_width
    return rng.uniform(low=low, high=high, size=(n, mu.shape[0]))


def sample_uniform_affine(
    n: int, mu: Array, C: Array, rng: np.random.Generator
) -> Array:
    """
    Sample x = mu + L u, where u has iid Uniform(-sqrt(3), sqrt(3)) entries.

    This construction matches mean mu and covariance C exactly (up to numerical
    precision), but coordinates of x are generally correlated and marginals are
    not uniform.

    Use this when you want "uniform-ish" features with a full (possibly
    non-diagonal) covariance matrix.
    """
    p = mu.shape[0]
    # u has mean 0 and covariance I_p
    u = rng.uniform(low=-np.sqrt(3.0), high=np.sqrt(3.0), size=(n, p))
    L = _safe_cholesky(C)
    return mu + u @ L.T


FeatureDistribution = Literal["gaussian", "uniform_iid", "uniform_affine"]


class BaseDataModel:
    """
    Base interface for data models producing (X, y).
    """

    p: int

    def sample(self, n: int, rng: Optional[np.random.Generator] = None) -> Tuple[Array, Array]:
        raise NotImplementedError

    @property
    def num_classes(self) -> int:
        return 1

    def sample_class(
        self, class_index: int, n: int, rng: Optional[np.random.Generator] = None
    ) -> Tuple[Array, Array]:
        """
        Conditional sampling for class k. Only meaningful for mixture models.
        """
        raise NotImplementedError

    def class_params(self) -> dict:
        """
        Return a dictionary of parameters useful for theoretical computations.
        """
        raise NotImplementedError
    def validate_model_moments(self, n_samples=20000, tol=1e-2):
        """
        Verifies that the model's sampled data matches its internal parameters.
        
        Args:
            model: Instance of your DataModel.
            n_samples: Number of samples to draw for computing empirical stats.
            tol: Tolerance for warning (just for display purposes).
        """
        print(f"--- Validating Model: {self.__class__.__name__} ---")
        print(f"Sampling {n_samples} points...")
        
        # 1. Get Theoretical Parameters
        params = self.class_params()
        theo_mus = params['mus']
        theo_covs = params['covs']
        theo_gamma = params['gamma']
        
        # 2. Get Empirical Data
        X_sample, y_sample = self.sample(n_samples)
        
        # Handle label mapping (align sampled labels to the index in params)
        # If the model has explicit y_values (like [-1, 1]), we map them to indices 0, 1...
        unique_labels = params.get('y_values', list(range(self.num_classes)))
        
        print(f"{'Class':<6} | {'Prop (Gamma)':<15} | {'Mean Diff (Norm)':<20} | {'Cov Diff (Frobenius)':<20}")
        print("-" * 75)

        for k, label_val in enumerate(unique_labels):
            # Filter samples for this class
            # (Using close for float comparison if labels are floats)
            mask = np.isclose(y_sample, label_val)
            X_k = X_sample[mask]
            n_k = len(X_k)
            
            # --- A. Check Proportions (Gamma) ---
            emp_gamma = n_k / n_samples
            theo_gamma_k = theo_gamma[k]
            
            # --- B. Check Means ---
            if n_k > 1:
                emp_mu = np.mean(X_k, axis=0)
                # L2 Norm of the difference vector
                mu_diff = np.linalg.norm(emp_mu - theo_mus[k])
            else:
                mu_diff = 0.0

            # --- C. Check Covariances ---
            if n_k > 1:
                # rowvar=False because X is (n_samples, n_features)
                emp_cov = np.cov(X_k, rowvar=False) 
                
                # If dimension is 1, np.cov returns a scalar array, fix shape
                if emp_cov.ndim == 0: emp_cov = emp_cov.reshape(1, 1)
                
                # Frobenius Norm of the difference matrix
                cov_diff = np.linalg.norm(emp_cov - theo_covs[k], ord='fro')
            else:
                cov_diff = 0.0

            # Print row
            print(f"{str(label_val):<6} | {emp_gamma:.3f} vs {theo_gamma_k:.3f} | {mu_diff:.5f}             | {cov_diff:.5f}")


@dataclass
class LinearFactorMixedModel(BaseDataModel):
    """
    Linear factor mixed model.

    x = sum_{i=1}^q (s_i * y + e_i) v_i
        + sum_{i=q+1}^p e_i v_i

    where:
      y ~ Bernoulli(P) mapped to {-1, +1}
      e_i ~ N(0, noise_std^2)
    """

    p: int
    q: int
    P: float                     # P(y = +1)
    s: Array                     # shape (q,)
    noise_std: float = 1.0
    basis: Optional[Array] = None  # shape (p, p), columns = v_i

    def __post_init__(self):
        if not (0 < self.P < 1):
            raise ValueError("P must be in (0, 1)")

        if self.q > self.p:
            raise ValueError("q must be <= p")

        self.s = np.asarray(self.s, dtype=float)
        if self.s.shape != (self.q,):
            raise ValueError(f"s must have shape ({self.q},)")

        # Default basis: canonical basis
        if self.basis is None:
            self.basis = np.eye(self.p)
        else:
            self.basis = np.asarray(self.basis, dtype=float)
            if self.basis.shape != (self.p, self.p):
                raise ValueError("basis must have shape (p, p)")

    @property
    def num_classes(self) -> int:
        return 2

    def sample(self, n: int, rng: Optional[np.random.Generator] = None) -> Tuple[Array, Array]:
        rng = np.random.default_rng() if rng is None else rng

        # --- Sample y in {-1, +1} ---
        y = rng.uniform(size=n) < self.P
        y = np.where(y, 1.0, -1.0)

        # --- Sample noise ---
        E = self.noise_std * rng.standard_normal(size=(n, self.p))

        # --- Signal contribution ---
        X = E.copy()
        X[:, :self.q] += y[:, None] * self.s[None, :]

        # --- Rotate into basis ---
        X = X @ self.basis.T

        return X, y

    def sample_class(self, class_index: int, n: int, rng: Optional[np.random.Generator] = None):
        rng = np.random.default_rng() if rng is None else rng

        y_val = 1.0 if class_index == 1 else -1.0
        y = np.full(n, y_val)

        E = self.noise_std * rng.standard_normal(size=(n, self.p))
        X = E.copy()
        X[:, :self.q] += y[:, None] * self.s[None, :]
        X = X @ self.basis.T

        return X, y

    def class_params(self) -> dict:
        """
        Theoretical mean and covariance for validation.
        """
        # Means
        mu_pos = np.zeros(self.p)
        mu_neg = np.zeros(self.p)
        mu_pos[:self.q] = self.s
        mu_neg[:self.q] = -self.s

        mu_pos = mu_pos @ self.basis.T
        mu_neg = mu_neg @ self.basis.T

        # Covariance (same for both classes)
        cov = self.noise_std ** 2 * np.eye(self.p)

        return dict(
            p=self.p,
            num_classes=2,
            gamma=np.array([1 - self.P, self.P]),
            mus=[mu_neg, mu_pos],
            covs=[cov, cov],
            y_values=[-1.0, 1.0],
        )
from dataclasses import dataclass
from typing import Literal, Optional, Sequence, Tuple
import numpy as np

Array = np.ndarray


@dataclass
# class TeacherStudentModel(BaseDataModel):
#     """
#     Teacher/student model with a *bimodal* (two-mode) feature distribution.

#     Generative model:
#       mode m ~ Categorical(gamma) over {0,1}
#       x | (m=k) ~ D_k with mean mu_k and covariance C_k
#       y generated from x via a teacher mechanism (same teacher for both modes)

#     Feature sampling options (for each mode):
#       - "gaussian": x_k ~ N(mu_k, C_k)
#       - "uniform_iid": independent Uniform coords matching mu_k and diag(C_k)
#       - "uniform_affine": affine-uniform matching full C_k (not independent)

#     Teacher mechanisms:
#       - "linear_regression": y = x^T theta_teacher + noise_std * eps
#       - "sign": y = sign(x^T theta_teacher + noise_std * eps) in {+1, -1}
#       - "logistic": P(y=+1|x)=sigmoid(x^T theta_teacher / temperature)

#     Notes
#     -----
#     * This is "teacher-student" in the sense that y is generated from x through
#       a teacher parameter theta_teacher, but the covariates x are a mixture of
#       two distributions (two modes).
#     * For compatibility with your theory code (which expects "classes"), we
#       treat the two modes as class_index=0 or 1 in sample_class().
#     """

#     p: int
#     gamma: Array                    # shape (2,)
#     mus: Sequence[Array]            # length 2, each (p,)
#     covs: Sequence[Array]           # length 2, each (p,p)
#     feature_dist: FeatureDistribution = "gaussian"

#     theta_teacher: Optional[Array] = None
#     y_model: Literal["linear_regression", "sign", "logistic"] = "linear_regression"
#     noise_std: float = 1.0
#     temperature: float = 1.0

#     def __post_init__(self) -> None:
#         # --- gamma ---
#         self.gamma = np.asarray(self.gamma, dtype=float).reshape(-1)
#         if self.gamma.shape != (2,):
#             raise ValueError(f"gamma must have shape (2,), got {self.gamma.shape}")
#         if np.any(self.gamma < 0):
#             raise ValueError("gamma must be nonnegative")
#         s = float(np.sum(self.gamma))
#         if not np.isfinite(s) or s <= 0:
#             raise ValueError("gamma must sum to a positive finite number")
#         self.gamma = self.gamma / s

#         # --- mus/covs ---
#         if len(self.mus) != 2 or len(self.covs) != 2:
#             raise ValueError("mus and covs must each have length 2")
#         self.mus = [_as_1d(self.mus[k], self.p, f"mu[{k}]") for k in range(2)]
#         self.covs = [_as_2d(self.covs[k], self.p, f"C[{k}]") for k in range(2)]

#         # --- teacher ---
#         if self.theta_teacher is None:
#             self.theta_teacher = np.zeros(self.p, dtype=float)
#         self.theta_teacher = _as_1d(self.theta_teacher, self.p, "theta_teacher")

#         self.noise_std = float(self.noise_std)
#         self.temperature = float(self.temperature)
#         if self.temperature <= 0:
#             raise ValueError("temperature must be > 0")

#         # uniform_iid only matches diagonal covariance; if any mode has non-diagonal cov, fallback
#         if self.feature_dist == "uniform_iid":
#             if any(not _is_diagonal(C) for C in self.covs):
#                 self.feature_dist = "uniform_affine"

#     @property
#     def num_classes(self) -> int:
#         # treat modes as "classes" for conditional sampling / theory expectations
#         return 2

#     def _sample_x_mode(self, k: int, n: int, rng: np.random.Generator) -> Array:
#         mu = self.mus[k]
#         C = self.covs[k]
#         if self.feature_dist == "gaussian":
#             return sample_gaussian(n, mu, C, rng)
#         elif self.feature_dist == "uniform_iid":
#             var = np.diag(C)
#             return sample_uniform_iid_from_mean_var(n, mu, var, rng)
#         elif self.feature_dist == "uniform_affine":
#             return sample_uniform_affine(n, mu, C, rng)
#         else:
#             raise ValueError(f"Unknown feature_dist={self.feature_dist}")

#     def _sample_y(self, X: Array, rng: np.random.Generator) -> Array:
#         s = X @ self.theta_teacher
#         if self.y_model == "linear_regression":
#             y = s + self.noise_std * rng.standard_normal(size=s.shape[0])
#             return y.astype(float)
#         elif self.y_model == "sign":
#             y = s + self.noise_std * rng.standard_normal(size=s.shape[0])
#             y = np.where(y >= 0, 1.0, -1.0)
#             return y.astype(float)
#         elif self.y_model == "logistic":
#             t = s / self.temperature
#             p_pos = 1.0 / (1.0 + np.exp(-t))
#             u = rng.uniform(size=t.shape[0])
#             y = np.where(u < p_pos, 1.0, -1.0)
#             return y.astype(float)
#         else:
#             raise ValueError(f"Unknown y_model={self.y_model}")

#     def sample_class(
#         self, class_index: int, n: int, rng: Optional[np.random.Generator] = None
#     ) -> Tuple[Array, Array]:
#         """
#         Sample conditionally on the mode (class_index = 0 or 1).
#         """
#         rng = np.random.default_rng() if rng is None else rng
#         k = int(class_index)
#         if k not in (0, 1):
#             raise ValueError("class_index must be 0 or 1")
#         X = self._sample_x_mode(k, n, rng)
#         y = self._sample_y(X, rng)
#         return X, y

#     def sample(self, n: int, rng: Optional[np.random.Generator] = None) -> Tuple[Array, Array]:
#         """
#         Sample from the bimodal mixture.
#         """
#         rng = np.random.default_rng() if rng is None else rng
#         mode = rng.choice(2, size=n, p=self.gamma)

#         X = np.zeros((n, self.p), dtype=float)
#         for k in (0, 1):
#             mask = (mode == k)
#             nk = int(np.sum(mask))
#             if nk > 0:
#                 X[mask] = self._sample_x_mode(k, nk, rng)

#         y = self._sample_y(X, rng)
#         return X, y

#     def overall_mean_cov(self) -> Tuple[Array, Array]:
#         """
#         Return the unconditional mean and covariance of x under the mixture.

#         mean = sum_k gamma_k mu_k
#         cov  = sum_k gamma_k (C_k + (mu_k-mean)(mu_k-mean)^T)
#         """
#         mu0, mu1 = self.mus
#         C0, C1 = self.covs
#         g0, g1 = float(self.gamma[0]), float(self.gamma[1])

#         mean = g0 * mu0 + g1 * mu1
#         d0 = (mu0 - mean).reshape(-1, 1)
#         d1 = (mu1 - mean).reshape(-1, 1)
#         cov = g0 * (C0 + d0 @ d0.T) + g1 * (C1 + d1 @ d1.T)
#         return mean, cov

#     def class_params(self) -> dict:
#         """
#         Provide mode parameters to the theory code.

#         Here "classes" correspond to the two modes of x, not to label values.
#         """
#         return dict(
#             p=self.p,
#             num_classes=2,
#             gamma=self.gamma.copy(),
#             mus=[m.copy() for m in self.mus],
#             covs=[C.copy() for C in self.covs],
#             y_values=None,  # labels are generated by teacher, not fixed per class
        # )
@dataclass
class TeacherStudentModel(BaseDataModel):
    """
    General teacher/student model.

    Features x are sampled from either:
      - multivariate Gaussian N(mu_x, C_x), or
      - independent uniform coordinates (matching mean + diagonal covariance),
      - affine-uniform (matching full covariance, not independent coordinates).

    Then y is produced via a "teacher" mechanism.

    Parameters
    ----------
    p : int
        Dimension of x.
    mu_x : array-like, shape (p,)
        Mean of x.
    C_x : array-like, shape (p,p)
        Covariance of x. If feature_dist="uniform_iid" and C_x is not diagonal,
        the model will fall back to "uniform_affine" (with a warning in the
        docstring) because independent uniform coordinates cannot reproduce a
        non-diagonal covariance.
    feature_dist : {"gaussian", "uniform_iid", "uniform_affine"}
    theta_teacher : array-like, shape (p,)
        Teacher parameter.
    y_model : {"linear_regression", "sign", "logistic"}
        Mechanism to generate y.
        - "linear_regression": y = x^T theta_teacher + noise_std * eps
        - "sign": y = sign(x^T theta_teacher + noise_std * eps) in {+1, -1}
        - "logistic": P(y=+1|x)=sigmoid(x^T theta_teacher / temperature)
    noise_std : float
        Standard deviation of additive noise used for "linear_regression" and
        "sign".
    temperature : float
        Used only for y_model="logistic". Larger => softer probabilities.
    """
    p: int
    mu_x: Array
    C_x: Array
    feature_dist: FeatureDistribution = "gaussian"
    theta_teacher: Optional[Array] = None
    y_model: Literal["linear_regression", "sign", "logistic"] = "linear_regression"
    noise_std: float = 1.0
    temperature: float = 1.0

    def __post_init__(self) -> None:
        self.mu_x = _as_1d(self.mu_x, self.p, "mu_x")
        self.C_x = _as_2d(self.C_x, self.p, "C_x")
        if self.theta_teacher is None:
            self.theta_teacher = np.zeros(self.p)
        self.theta_teacher = _as_1d(self.theta_teacher, self.p, "theta_teacher")

        if self.feature_dist == "uniform_iid" and not _is_diagonal(self.C_x):
            # Fallback for correctness; we can't match non-diagonal covariance with iid uniforms.
            self.feature_dist = "uniform_affine"

    def _sample_x(self, n: int, rng: np.random.Generator) -> Array:
        if self.feature_dist == "gaussian":
            return sample_gaussian(n, self.mu_x, self.C_x, rng)
        elif self.feature_dist == "uniform_iid":
            var = np.diag(self.C_x)
            return sample_uniform_iid_from_mean_var(n, self.mu_x, var, rng)
        elif self.feature_dist == "uniform_affine":
            return sample_uniform_affine(n, self.mu_x, self.C_x, rng)
        else:
            raise ValueError(f"Unknown feature_dist={self.feature_dist}")

    def _sample_y(self, X: Array, rng: np.random.Generator) -> Array:
        s = X @ self.theta_teacher
        if self.y_model == "linear_regression":
            y = s + self.noise_std * rng.standard_normal(size=s.shape[0])
            return y.astype(float)
        elif self.y_model == "sign":
            y = s + self.noise_std * rng.standard_normal(size=s.shape[0])
            y = np.where(y >= 0, 1.0, -1.0)
            return y.astype(float)
        elif self.y_model == "logistic":
            t = s / float(self.temperature)
            # sigmoid(t)
            p_pos = 1.0 / (1.0 + np.exp(-t))
            u = rng.uniform(size=t.shape[0])
            y = np.where(u < p_pos, 1.0, -1.0)
            return y.astype(float)
        else:
            raise ValueError(f"Unknown y_model={self.y_model}")

    def sample(self, n: int, rng: Optional[np.random.Generator] = None) -> Tuple[Array, Array]:
        rng = np.random.default_rng() if rng is None else rng
        X = self._sample_x(n, rng)
        y = self._sample_y(X, rng)
        return X, y

    def sample_class(self, class_index: int, n: int, rng: Optional[np.random.Generator] = None):
        # Single-class model: ignore class_index
        return self.sample(n, rng=rng)


    def error_classif_th(self,mu: Array, alpha: Array, num_trials_th=100000, rng: Optional[np.random.Generator] = None) -> float:
        rng = np.random.default_rng() if rng is None else rng
        if self.y_model != "sign":
            raise ValueError("for classification y_model must be sign")
        z_samples = rng.standard_normal(size=num_trials_th) 
        Xtst, utst = self.sample_class(0,num_trials_th, rng=rng)
        scores_th = Xtst@mu + alpha*z_samples
        return float(sum(np.array(scores_th)>0)/num_trials_th)

    def error_classif_emp(self, thetas_rcrd: Sequence[Array], n_test=5000, rng: Optional[np.random.Generator] = None) -> float:
        rng = np.random.default_rng() if rng is None else rng
        if self.y_model != "sign":
            raise ValueError("for classification y_model must be sign")
        num_trials = len(thetas_rcrd)
        sc_emp=[]
        for i in range(num_trials):
            Xte, ute = self.sample_class(0, n_test, rng=rng)
            # eps, yte = self._split_u(ute)
            sc_emp += [Xte[j]@thetas_rcrd[i] for j in range(n_test)]
        return float(sum(np.array(sc_emp)>0)/(num_trials*n_test))

    def class_params(self) -> dict:
        return dict(
            p=self.p,
            num_classes=1,
            gamma=np.array([1.0]),
            mus=[self.mu_x.copy()],
            covs=[self.C_x.copy()],
            y_values=None,
        )



@dataclass
class TwoModeTeacherStudentModel(BaseDataModel):
    """
    Teacher/student model where x has a *two-mode* (bimodal) distribution,
    but the task is still a standard teacher/student regression or classification
    (y is generated from x through a teacher).

    Construction:
      - You specify the *overall* mean mu_x and covariance C_x.
      - You specify a mode-separation vector delta (Delta) and a mixing weight pi.
      - We build a two-component mixture for x:
            mode m ~ Bernoulli(pi)
            x | (m=0) ~ D(mu0, C_base)
            x | (m=1) ~ D(mu1, C_base)
        where
            mu0 = mu_x - pi * delta
            mu1 = mu_x + (1-pi) * delta
            C_base = C_x - pi*(1-pi) * delta delta^T
        so that the *unconditional* mean/covariance of x are exactly (mu_x, C_x)
        (as long as C_base is PSD).

    Feature sampling options (for each mode):
      - "gaussian": N(mu_mode, C_base)
      - "uniform_iid": independent Uniform coords matching mu_mode and diag(C_base)
      - "uniform_affine": affine-uniform matching full C_base

    Teacher mechanisms (same as TeacherStudentModel):
      - "linear_regression": y = x^T theta_teacher + noise_std * eps
      - "sign": y = sign(x^T theta_teacher + noise_std * eps) in {+1, -1}
      - "logistic": P(y=+1|x)=sigmoid(x^T theta_teacher / temperature)

    Notes
    -----
    * This model has num_classes=1 for theory code compatibility; the two modes
      are *not* exposed as classes.
    * If C_base is not PSD, the requested (mu_x, C_x, pi, delta) combination is
      infeasible. Reduce ||delta|| or adjust pi.
    """

    p: int
    mu_x: Array
    C_x: Array
    feature_dist: FeatureDistribution = "gaussian"

    # Bimodality parameters
    pi: float = 0.5                    # P(mode=1), must be in (0,1)
    delta: Optional[Array] = None      # mode separation vector (mu1 - mu0)

    # Teacher parameters
    theta_teacher: Optional[Array] = None
    y_model: Literal["linear_regression", "sign", "logistic"] = "linear_regression"
    noise_std: float = 1.0
    temperature: float = 1.0

    def __post_init__(self) -> None:
        self.mu_x = _as_1d(self.mu_x, self.p, "mu_x")
        self.C_x = _as_2d(self.C_x, self.p, "C_x")

        self.pi = float(self.pi)
        if not (0.0 < self.pi < 1.0):
            raise ValueError("pi must be in (0,1)")

        if self.delta is None:
            self.delta = np.zeros(self.p, dtype=float)
        self.delta = _as_1d(self.delta, self.p, "delta")

        if self.theta_teacher is None:
            self.theta_teacher = np.zeros(self.p, dtype=float)
        self.theta_teacher = _as_1d(self.theta_teacher, self.p, "theta_teacher")

        self.noise_std = float(self.noise_std)
        self.temperature = float(self.temperature)
        if self.temperature <= 0.0:
            raise ValueError("temperature must be > 0")

        # Build component means and shared within-mode covariance
        pi = self.pi
        delta = self.delta

        self._mu0 = self.mu_x - pi * delta
        self._mu1 = self.mu_x + (1.0 - pi) * delta

        C_base = self.C_x - (pi * (1.0 - pi)) * np.outer(delta, delta)
        self._C_base = _symmetrize(C_base)

        # PSD check (allow tiny numerical negatives)
        w = np.linalg.eigvalsh(self._C_base)
        min_w = float(np.min(w))
        if min_w < -1e-10:
            raise ValueError(
                "Infeasible bimodal parameters: C_base = C_x - pi(1-pi) delta delta^T "
                "is not PSD. Reduce ||delta|| or adjust pi."
            )
        # If slightly negative due to numerical noise, add a tiny jitter
        if min_w < 0.0:
            self._C_base = self._C_base + (-min_w + 1e-12) * np.eye(self.p)

        # For uniform_iid, fall back to uniform_affine if covariance not diagonal.
        if self.feature_dist == "uniform_iid" and not _is_diagonal(self._C_base):
            self.feature_dist = "uniform_affine"

    def _sample_x_mode(self, mu_mode: Array, n: int, rng: np.random.Generator) -> Array:
        if self.feature_dist == "gaussian":
            return sample_gaussian(n, mu_mode, self._C_base, rng)
        elif self.feature_dist == "uniform_iid":
            var = np.diag(self._C_base)
            return sample_uniform_iid_from_mean_var(n, mu_mode, var, rng)
        elif self.feature_dist == "uniform_affine":
            return sample_uniform_affine(n, mu_mode, self._C_base, rng)
        else:
            raise ValueError(f"Unknown feature_dist={self.feature_dist}")

    def _sample_x(self, n: int, rng: np.random.Generator) -> Array:
        # One mode per sample
        mode1 = rng.uniform(size=n) < self.pi
        X = np.zeros((n, self.p), dtype=float)

        n1 = int(np.sum(mode1))
        n0 = n - n1
        if n0 > 0:
            X[~mode1] = self._sample_x_mode(self._mu0, n0, rng)
        if n1 > 0:
            X[mode1] = self._sample_x_mode(self._mu1, n1, rng)
        return X

    def _sample_y(self, X: Array, rng: np.random.Generator) -> Array:
        s = X @ self.theta_teacher
        if self.y_model == "linear_regression":
            y = s + self.noise_std * rng.standard_normal(size=s.shape[0])
            return y.astype(float)
        elif self.y_model == "sign":
            y = s + self.noise_std * rng.standard_normal(size=s.shape[0])
            return np.where(y >= 0, 1.0, -1.0).astype(float)
        elif self.y_model == "logistic":
            t = s / float(self.temperature)
            p_pos = 1.0 / (1.0 + np.exp(-t))
            u = rng.uniform(size=t.shape[0])
            return np.where(u < p_pos, 1.0, -1.0).astype(float)
        else:
            raise ValueError(f"Unknown y_model={self.y_model}")

    def sample(self, n: int, rng: Optional[np.random.Generator] = None) -> Tuple[Array, Array]:
        rng = np.random.default_rng() if rng is None else rng
        X = self._sample_x(n, rng)
        y = self._sample_y(X, rng)
        return X, y

    def sample_class(self, class_index: int, n: int, rng: Optional[np.random.Generator] = None):
        # Single-population interface: ignore class_index
        return self.sample(n, rng=rng)

    def class_params(self) -> dict:
        # For theory code: expose only the unconditional mean/covariance.
        return dict(
            p=self.p,
            num_classes=1,
            gamma=np.array([1.0]),
            mus=[self.mu_x.copy()],
            covs=[self.C_x.copy()],
            y_values=None,
        )
from dataclasses import dataclass
from typing import Sequence, Optional, Tuple, Literal
import numpy as np

# Assuming BaseDataModel, Array, and helper functions (_as_1d, _as_2d, _is_diagonal, etc.) 
# are defined in your environment as per your snippet. 
# If not, simpler versions would need to be added.

FeatureDistribution = Literal["gaussian", "uniform_iid", "uniform_affine", "beta"]
Array = np.ndarray

def sample_gaussian(n, mu, C, rng):
    return rng.multivariate_normal(mu, C, n)

def sample_uniform_iid_from_mean_var(n, mu, var, rng):
    # Uniform[a, b] has mean (a+b)/2 and var (b-a)^2/12
    # width = sqrt(12 * var)
    width = np.sqrt(12 * var)
    a = mu - width / 2
    b = mu + width / 2
    return rng.uniform(a, b, (n, len(mu)))

def sample_uniform_affine(n, mu, C, rng):
    # X = U * A + mu, where U ~ Uniform[-0.5, 0.5]
    # Cov(X) = A^T Cov(U) A = A^T (I/12) A = (1/12) A^T A
    # We need C = (1/12) A^T A  =>  12*C = A^T A
    # We can use Cholesky: A = cholesky(12*C).T
    # However, standard implementations might vary. 
    # Simplified placeholder logic:
    p = len(mu)
    L = np.linalg.cholesky(12 * C)
    # U is usually defined on [0,1], centered is [-0.5, 0.5]
    U = rng.uniform(-0.5, 0.5, (n, p)) 
    return U @ L.T + mu

@dataclass
class MixtureClassificationModel(BaseDataModel):
    """
    K-class mixture model for classification-like settings with optional bimodal features.

    Supports Gaussian, Uniform, and Beta distributions.
    
    For Beta distribution:
        - Set feature_dist="beta"
        - Provide `alphas` and `betas` (lists of length K).
        - `mus` and `covs` are ignored for sampling but required by the class structure 
          (you can pass dummies or use them to store theoretical means/vars).
        - `sub_mode_deltas` splits the ALPHA parameter: 
             Left Mode Alpha = alpha - delta/2
             Right Mode Alpha = alpha + delta/2
    """
    p: int
    gamma: Array
    mus: Sequence[Array]
    covs: Sequence[Array]
    sub_mode_deltas: Optional[Sequence[Array]] = None
    y_values: Optional[Sequence[float]] = None
    feature_dist: FeatureDistribution = "gaussian"
    
    # --- NEW: Beta parameters ---
    alphas: Optional[Sequence[Array]] = None
    betas: Optional[Sequence[Array]] = None

    def __post_init__(self) -> None:
        self.gamma = np.asarray(self.gamma, dtype=float).reshape(-1)
        if self.gamma.ndim != 1 or self.gamma.shape[0] < 1:
            raise ValueError("gamma must be a 1D array with length K>=1")
        K = self.gamma.shape[0]
        if np.any(self.gamma < 0):
            raise ValueError("gamma must be nonnegative")
        s = float(np.sum(self.gamma))
        if not np.isfinite(s) or s <= 0:
            raise ValueError("gamma must sum to a positive finite number")
        # Normalize for safety
        self.gamma = self.gamma / s

        if len(self.mus) != K or len(self.covs) != K:
            raise ValueError("mus and covs must have length K=len(gamma)")
        
        self.mus = [_as_1d(mu, self.p, f"mu[{k}]") for k, mu in enumerate(self.mus)]
        self.covs = [_as_2d(C, self.p, f"C[{k}]") for k, C in enumerate(self.covs)]

        # --- Handle sub_mode_deltas ---
        if self.sub_mode_deltas is None:
            self.sub_mode_deltas = [np.zeros(self.p) for _ in range(K)]
        else:
            if len(self.sub_mode_deltas) != K:
                raise ValueError("sub_mode_deltas must have length K=len(gamma)")
            self.sub_mode_deltas = [_as_1d(d, self.p, f"delta[{k}]") for k, d in enumerate(self.sub_mode_deltas)]

        if self.y_values is None:
            self.y_values = list(range(K))
        if len(self.y_values) != K:
            raise ValueError("y_values must have length K=len(gamma)")
        self.y_values = [float(v) for v in self.y_values]

        # --- NEW: Handle Beta Params ---
        if self.feature_dist == "beta":
            if self.alphas is None or self.betas is None:
                raise ValueError("If feature_dist='beta', `alphas` and `betas` must be provided.")
            if len(self.alphas) != K or len(self.betas) != K:
                raise ValueError("alphas and betas must have length K=len(gamma)")
            
            self.alphas = [_as_1d(a, self.p, f"alpha[{k}]") for k, a in enumerate(self.alphas)]
            self.betas = [_as_1d(b, self.p, f"beta[{k}]") for k, b in enumerate(self.betas)]
            
            # Check for valid params (>0)
            if any(np.any(a <= 0) for a in self.alphas) or any(np.any(b <= 0) for b in self.betas):
                raise ValueError("Alpha and Beta parameters must be strictly positive.")

        # For uniform_iid, fall back to uniform_affine if any covariance is not diagonal.
        if self.feature_dist == "uniform_iid":
            if any(not _is_diagonal(C) for C in self.covs):
                self.feature_dist = "uniform_affine"

    @property
    def num_classes(self) -> int:
        return int(self.gamma.shape[0])

    def _sample_from_dist(self, n: int, param1: Array, param2: Array, rng: np.random.Generator) -> Array:
        """
        Helper to sample from the specific base distribution.
        
        Args:
            param1: Primary parameter (Mean `mu` for Gaussian/Uniform, Shape `alpha` for Beta)
            param2: Secondary parameter (Covariance `C` for Gaussian/Uniform, Shape `beta` for Beta)
        """
        if n == 0:
            return np.zeros((0, self.p))
            
        if self.feature_dist == "gaussian":
            return sample_gaussian(n, param1, param2, rng)
        elif self.feature_dist == "uniform_iid":
            var = np.diag(param2)
            return sample_uniform_iid_from_mean_var(n, param1, var, rng)
        elif self.feature_dist == "uniform_affine":
            return sample_uniform_affine(n, param1, param2, rng)
        elif self.feature_dist == "beta":
            # param1 is alpha (vector), param2 is beta (vector)
            # numpy.random.beta takes `a` and `b`. 
            # If `p` > 1, passing arrays (p,) works and returns (n, p) if size=(n, p)
            return rng.beta(param1, param2, size=(n, self.p))
        else:
            raise ValueError(f"Unknown feature_dist={self.feature_dist}")

    def _sample_x_k(self, k: int, n: int, rng: np.random.Generator) -> Array:
        # 1. Select Parameters based on distribution
        if self.feature_dist == "beta":
            # For Beta, we use alphas and betas
            # Primary = Alpha, Secondary = Beta
            p1_center = self.alphas[k]
            p2_center = self.betas[k]
        else:
            # For Gaussian/Uniform, we use mus and covs
            # Primary = Mu, Secondary = Covariance
            p1_center = self.mus[k]
            p2_center = self.covs[k]

        delta = self.sub_mode_deltas[k]

        # If delta is essentially zero, use standard unimodal sampling
        if np.allclose(delta, 0):
            return self._sample_from_dist(n, p1_center, p2_center, rng)

        # --- Bimodal Logic ---
        # We split the n samples into two sub-modes with 50/50 probability
        mode_choices = rng.integers(0, 2, size=n) 
        n_left = np.sum(mode_choices == 0)
        n_right = n - n_left

        # 2. Define centers for sub-modes
        # Shift the PRIMARY parameter by +/- delta/2
        # For Gaussian: Shift Mean. For Beta: Shift Alpha.
        p1_left = p1_center - 0.5 * delta
        p1_right = p1_center + 0.5 * delta

        # Note: For Beta, this shift must keep alpha > 0.
        if self.feature_dist == "beta":
            if np.any(p1_left <= 0) or np.any(p1_right <= 0):
                raise ValueError(f"Splitting Alpha with delta results in non-positive alpha for class {k}")

        # 3. Sample
        X_left = self._sample_from_dist(n_left, p1_left, p2_center, rng)
        X_right = self._sample_from_dist(n_right, p1_right, p2_center, rng)

        # 4. Combine preserving random order
        X = np.empty((n, self.p), dtype=float)
        X[mode_choices == 0] = X_left
        X[mode_choices == 1] = X_right
        
        return X

    def sample_class(
        self, class_index: int, n: int, rng: Optional[np.random.Generator] = None
    ) -> Tuple[Array, Array]:
        rng = np.random.default_rng() if rng is None else rng
        k = int(class_index)
        if k < 0 or k >= self.num_classes:
            raise ValueError(f"class_index must be in [0, K-1], got {k}")
        X = self._sample_x_k(k, n, rng)
        y = np.full(shape=(n,), fill_value=self.y_values[k], dtype=float)
        return X, y

    def sample(self, n: int, rng: Optional[np.random.Generator] = None) -> Tuple[Array, Array]:
        rng = np.random.default_rng() if rng is None else rng
        K = self.num_classes
        class_idx = rng.choice(K, size=n, p=self.gamma)
        X = np.zeros((n, self.p), dtype=float)
        y = np.zeros(n, dtype=float)
        for k in range(K):
            mask = class_idx == k
            nk = int(np.sum(mask))
            if nk == 0:
                continue
            X[mask] = self._sample_x_k(k, nk, rng)
            y[mask] = self.y_values[k]
        return X, y

    def class_params(self) -> dict:
        base = dict(
            p=self.p,
            num_classes=self.num_classes,
            gamma=self.gamma.copy(),
            mus=[m.copy() for m in self.mus],
            covs=[C.copy() for C in self.covs],
            sub_mode_deltas=[d.copy() for d in self.sub_mode_deltas],
            y_values=list(self.y_values),
        )
        if self.alphas is not None:
            base['alphas'] = [a.copy() for a in self.alphas]
        if self.betas is not None:
            base['betas'] = [b.copy() for b in self.betas]
        return base

@dataclass
class MixtureClassificationModelGaussian(BaseDataModel): # Inherit from BaseDataModel if available in your env
    """
    K-class mixture model for classification-like settings with optional bimodal features.

    A class label y takes values y_values[k] with probability gamma[k].
    Then x|y=y_values[k] is distributed as a mixture of two sub-modes:
       0.5 * P(x | center = mu_k - delta_k/2) + 0.5 * P(x | center = mu_k + delta_k/2)

    Parameters
    ----------
    p : int
        Dimension of feature vector.
    gamma : array-like shape (K,)
        Class proportions.
    mus : sequence of length K, each shape (p,)
        The global center of each class.
    covs : sequence of length K, each shape (p,p)
        Covariance matrix for the sub-modes.
    sub_mode_deltas : Optional[Sequence[Array]]
        Sequence of length K, each shape (p,). 
        Represents the vector difference between the two sub-modes for class k.
        If None, defaults to zeros (unimodal).
    y_values : sequence of length K
        The actual label values (e.g., [0, 1] or [-1, 1]).
    feature_dist : {"gaussian", "uniform_iid", "uniform_affine"}
    """
    p: int
    gamma: Array
    mus: Sequence[Array]
    covs: Sequence[Array]
    sub_mode_deltas: Optional[Sequence[Array]] = None
    y_values: Optional[Sequence[float]] = None
    feature_dist: FeatureDistribution = "gaussian"

    def __post_init__(self) -> None:
        self.gamma = np.asarray(self.gamma, dtype=float).reshape(-1)
        if self.gamma.ndim != 1 or self.gamma.shape[0] < 1:
            raise ValueError("gamma must be a 1D array with length K>=1")
        K = self.gamma.shape[0]
        if np.any(self.gamma < 0):
            raise ValueError("gamma must be nonnegative")
        s = float(np.sum(self.gamma))
        if not np.isfinite(s) or s <= 0:
            raise ValueError("gamma must sum to a positive finite number")
        # Normalize for safety
        self.gamma = self.gamma / s

        if len(self.mus) != K or len(self.covs) != K:
            raise ValueError("mus and covs must have length K=len(gamma)")
        
        self.mus = [_as_1d(mu, self.p, f"mu[{k}]") for k, mu in enumerate(self.mus)]
        self.covs = [_as_2d(C, self.p, f"C[{k}]") for k, C in enumerate(self.covs)]

        # --- NEW: Handle sub_mode_deltas ---
        if self.sub_mode_deltas is None:
            # Default to no separation (unimodal)
            self.sub_mode_deltas = [np.zeros(self.p) for _ in range(K)]
        else:
            if len(self.sub_mode_deltas) != K:
                raise ValueError("sub_mode_deltas must have length K=len(gamma)")
            self.sub_mode_deltas = [_as_1d(d, self.p, f"delta[{k}]") for k, d in enumerate(self.sub_mode_deltas)]
        # -----------------------------------

        if self.y_values is None:
            # Default: integer labels 0..K-1
            self.y_values = list(range(K))
        if len(self.y_values) != K:
            raise ValueError("y_values must have length K=len(gamma)")
        self.y_values = [float(v) for v in self.y_values]

        # For uniform_iid, fall back to uniform_affine if any covariance is not diagonal.
        if self.feature_dist == "uniform_iid":
            if any(not _is_diagonal(C) for C in self.covs):
                self.feature_dist = "uniform_affine"

    @property
    def num_classes(self) -> int:
        return int(self.gamma.shape[0])

    def _sample_from_dist(self, n: int, mu: Array, C: Array, rng: np.random.Generator) -> Array:
        """Helper to sample from the specific base distribution."""
        if n == 0:
            return np.zeros((0, self.p))
            
        if self.feature_dist == "gaussian":
            return sample_gaussian(n, mu, C, rng)
        elif self.feature_dist == "uniform_iid":
            var = np.diag(C)
            return sample_uniform_iid_from_mean_var(n, mu, var, rng)
        elif self.feature_dist == "uniform_affine":
            return sample_uniform_affine(n, mu, C, rng)
        else:
            raise ValueError(f"Unknown feature_dist={self.feature_dist}")

    def _sample_x_k(self, k: int, n: int, rng: np.random.Generator) -> Array:
        mu_center = self.mus[k]
        delta = self.sub_mode_deltas[k]
        C = self.covs[k]

        # If delta is essentially zero, use standard unimodal sampling
        if np.allclose(delta, 0):
            return self._sample_from_dist(n, mu_center, C, rng)

        # --- Bimodal Logic ---
        # We split the n samples into two sub-modes with 50/50 probability
        # 1. Assign each sample to mode 0 (left) or mode 1 (right)
        mode_choices = rng.integers(0, 2, size=n) # 0 or 1
        n_left = np.sum(mode_choices == 0)
        n_right = n - n_left

        # 2. Define centers for sub-modes
        # Mode 1: mu - delta/2
        # Mode 2: mu + delta/2
        mu_left = mu_center - 0.5 * delta
        mu_right = mu_center + 0.5 * delta

        # 3. Sample
        X_left = self._sample_from_dist(n_left, mu_left, C, rng)
        X_right = self._sample_from_dist(n_right, mu_right, C, rng)

        # 4. Combine preserving random order
        X = np.empty((n, self.p), dtype=float)
        X[mode_choices == 0] = X_left
        X[mode_choices == 1] = X_right
        # --- NEW: Gaussian replacement with same empirical statistics ---
        # Compute empirical mean and covariance of the sampled X
        if X.shape[0] <= 1: 
            X_gauss = X
        else:
            # print(X.shape)
            empirical_mu = np.mean(X, axis=0)
            empirical_C = np.cov(X, rowvar=False)
            # print(empirical_C.shape)

            # Sample Gaussian with the same statistics
            X_gauss = rng.multivariate_normal(empirical_mu, empirical_C, size=n)
        
        return X_gauss

    def sample_class(
        self, class_index: int, n: int, rng: Optional[np.random.Generator] = None
    ) -> Tuple[Array, Array]:
        rng = np.random.default_rng() if rng is None else rng
        k = int(class_index)
        if k < 0 or k >= self.num_classes:
            raise ValueError(f"class_index must be in [0, K-1], got {k}")
        X = self._sample_x_k(k, n, rng)
        y = np.full(shape=(n,), fill_value=self.y_values[k], dtype=float)
        return X, y

    def sample(self, n: int, rng: Optional[np.random.Generator] = None) -> Tuple[Array, Array]:
        rng = np.random.default_rng() if rng is None else rng
        K = self.num_classes
        class_idx = rng.choice(K, size=n, p=self.gamma)
        X = np.zeros((n, self.p), dtype=float)
        y = np.zeros(n, dtype=float)
        for k in range(K):
            mask = class_idx == k
            nk = int(np.sum(mask))
            if nk == 0:
                continue
            X[mask] = self._sample_x_k(k, nk, rng)
            y[mask] = self.y_values[k]
        return X, y

    def class_params(self) -> dict:
        return dict(
            p=self.p,
            num_classes=self.num_classes,
            gamma=self.gamma.copy(),
            mus=[m.copy() for m in self.mus],
            covs=[C.copy() for C in self.covs],
            sub_mode_deltas=[d.copy() for d in self.sub_mode_deltas],
            y_values=list(self.y_values),
        )





@dataclass
class MNISTDataModel(BaseDataModel): # Inherit from BaseDataModel if needed

    """
    Empirical MNIST data model with optional random-feature map and additive Gaussian noise.

    Parameters
    ----------
    data_path: str
        Local .npz file containing MNIST arrays.
    noise_std: float
        Standard deviation of Gaussian noise added to the FINAL representation.
        - If representation="raw", noise is added to pixels.
        - If representation="random_features", noise is added to the features.
    ... [Other parameters unchanged] ...
    """

    data_path: str

    split: Literal["train", "test", "full"] = "test"
    stats_split: Literal["train", "test", "full"] = "train"

    representation: Literal["raw", "random_features"] = "raw"
    
    noise_std: float = 0.0

    pixel_scaling: Literal["uint8", "unit_interval"] = "unit_interval"
    dtype: Any = np.float32

    cov_kind: Literal["full", "diag"] = "full"
    cov_reg: float = 1e-6

    # class filtering / task definition
    classes: Optional[Sequence[int]] = None
    task: Literal["multiclass", "binary"] = "multiclass"
    positive_classes: Optional[Sequence[int]] = None

    # sampling behavior
    replace: bool = True

    # random feature map
    W: Optional[Array] = None
    bias: Optional[Array] = None
    activation: Literal["identity", "relu", "tanh", "cos", "sign_pm1"] = "identity"
    feature_scale: float = 1.0

    # computed / cached
    p_raw: int = field(init=False)
    p: int = field(init=False)

    _X_train: Array = field(init=False, repr=False)
    _y_train: Array = field(init=False, repr=False)
    _X_test: Array = field(init=False, repr=False)
    _y_test: Array = field(init=False, repr=False)

    _class_labels: Array = field(init=False, repr=False)
    _label_map: Dict[int, int] = field(init=False, repr=False)

    _cached_params_raw: Optional[Dict[str, Any]] = field(default=None, init=False, repr=False)
    _cached_params_rf: Optional[Dict[str, Any]] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        # NOTE: Assuming _load_mnist_npz and helpers are available in scope
        x_train, y_train, x_test, y_test = _load_mnist_npz(self.data_path)

        Xtr = _as_2d_flattened_uint8(x_train)
        Xte = _as_2d_flattened_uint8(x_test)

        self._X_train = _normalize_pixels(Xtr, self.pixel_scaling, np.dtype(self.dtype))
        self._y_train = np.asarray(y_train, dtype=int)

        self._X_test = _normalize_pixels(Xte, self.pixel_scaling, np.dtype(self.dtype))
        self._y_test = np.asarray(y_test, dtype=int)

        self.p_raw = int(self._X_train.shape[1])

        # Determine which digit classes are kept
        if self.classes is None:
            kept = np.arange(10, dtype=int)
        else:
            kept = np.asarray(list(self.classes), dtype=int)
            if kept.ndim != 1 or kept.size == 0:
                raise ValueError("classes must be a non-empty 1D sequence of digit labels.")
            if np.any((kept < 0) | (kept > 9)):
                raise ValueError(f"classes must be digits in 0..9, got {kept.tolist()}")

        self._class_labels = kept
        self._label_map = {int(d): i for i, d in enumerate(self._class_labels.tolist())}

        # Validate task configuration
        if self.task == "binary":
            if self.positive_classes is None or len(self.positive_classes) == 0:
                raise ValueError("For task='binary', you must provide positive_classes.")
            pos = set(int(d) for d in self.positive_classes)
            if not pos.issubset(set(int(d) for d in range(10))):
                raise ValueError("positive_classes must be digit labels in 0..9.")
            if self.classes is not None:
                kept_set = set(int(d) for d in self._class_labels.tolist())
                if len(pos.intersection(kept_set)) == 0:
                    raise ValueError(
                        "Binary task: none of positive_classes are included in `classes`."
                    )

        # Set output dimension p
        if self.representation == "raw":
            self.p = self.p_raw
        else:
            if self.W is None:
                raise ValueError("representation='random_features' requires providing W.")
            W = np.asarray(self.W)
            if W.ndim != 2 or W.shape[1] != self.p_raw:
                raise ValueError(
                    f"W must have shape (m, p_raw) = (m, {self.p_raw}), got {W.shape}."
                )
            m = int(W.shape[0])
            if self.bias is not None:
                b = np.asarray(self.bias)
                if b.shape not in [(m,), (1, m)]:
                    raise ValueError(f"bias must have shape ({m},) (or (1,{m})), got {b.shape}")
            self.p = m

    @property
    def num_classes(self) -> int:
        if self.task == "binary":
            return 2
        return int(self._class_labels.size)

    # -----------------------------
    # Internal Helpers
    # -----------------------------
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

    def _filter_and_encode_labels(self, X: Array, y_digits: Array) -> Tuple[Array, Array]:
        if self.task == "binary":
            pos = set(int(d) for d in self.positive_classes or [])
            if self.classes is not None:
                kept_set = set(int(d) for d in self._class_labels.tolist())
                mask = np.array([int(yy) in kept_set for yy in y_digits], dtype=bool)
                X = X[mask]
                y_digits = y_digits[mask]
            y = np.where(np.isin(y_digits, list(pos)), 1.0, -1.0).astype(self.dtype, copy=False)
            return X, y

        if self.classes is None:
            y_idx = y_digits.astype(int, copy=False)
            return X, y_idx

        mask = np.isin(y_digits, self._class_labels)
        Xf = X[mask]
        yf = y_digits[mask].astype(int, copy=False)
        y_idx = np.array([self._label_map[int(yy)] for yy in yf], dtype=int)
        return Xf, y_idx

    def _sample_raw_clean(self, n: int, rng: np.random.Generator) -> Tuple[Array, Array]:
        """Internal: fetch 'n' clean raw pixel vectors and labels."""
        X_all, y_all_digits = self._get_split(self.split)
        Xf, yf = self._filter_and_encode_labels(X_all, y_all_digits)

        if Xf.shape[0] == 0:
            raise ValueError("No samples available after filtering.")

        idx = rng.choice(Xf.shape[0], size=int(n), replace=bool(self.replace))
        return Xf[idx], yf[idx]
    
    def _add_noise(self, X: Array, rng: np.random.Generator) -> Array:
        """Add Gaussian noise if noise_std > 0."""
        if self.noise_std > 0:
            return X + rng.normal(scale=self.noise_std, size=X.shape).astype(self.dtype)
        return X
    

@dataclass
class MNISTMergedDataModel(BaseDataModel):
    """
    Modified MNIST data model with bi-modal classes (two digits per class).
    
    Parameters
    ----------
    data_path: str
        Local .npz file containing MNIST arrays.
    noise_std: float
        Standard deviation of Gaussian noise added to the FINAL representation.
    representation: Literal["raw", "random_features"] = "raw"
    ...
    """

    data_path: str
    split: Literal["train", "test", "full"] = "test"
    stats_split: Literal["train", "test", "full"] = "train"
    representation: Literal["raw", "random_features"] = "raw"
    noise_std: float = 0.0
    pixel_scaling: Literal["uint8", "unit_interval"] = "unit_interval"
    dtype: Any = np.float32
    cov_kind: Literal["full", "diag"] = "full"
    cov_reg: float = 1e-6
    classes: Optional[Sequence[int]] = None  # Should be 0 and 1 as we are using bi-modal classification
    task: Literal["multiclass", "binary"] = "multiclass"
    positive_classes: Optional[Sequence[int]] = None
    replace: bool = True
    W: Optional[Array] = None
    bias: Optional[Array] = None
    activation: Literal["identity", "relu", "tanh", "cos", "sign_pm1"] = "identity"
    feature_scale: float = 1.0
    p_raw: int = field(init=False)
    p: int = field(init=False)
    _X_train: Array = field(init=False, repr=False)
    _y_train: Array = field(init=False, repr=False)
    _X_test: Array = field(init=False, repr=False)
    _y_test: Array = field(init=False, repr=False)
    _class_labels: Array = field(init=False, repr=False)
    _label_map: Dict[int, int] = field(init=False, repr=False)
    _cached_params_raw: Optional[Dict[str, Any]] = field(default=None, init=False, repr=False)
    _cached_params_rf: Optional[Dict[str, Any]] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        # Load data
        x_train, y_train, x_test, y_test = _load_mnist_npz(self.data_path)

        Xtr = _as_2d_flattened_uint8(x_train)
        Xte = _as_2d_flattened_uint8(x_test)

        self._X_train = _normalize_pixels(Xtr, self.pixel_scaling, np.dtype(self.dtype))
        self._y_train = np.asarray(y_train, dtype=int)

        self._X_test = _normalize_pixels(Xte, self.pixel_scaling, np.dtype(self.dtype))
        self._y_test = np.asarray(y_test, dtype=int)

        self.p_raw = int(self._X_train.shape[1])

        # Define classes as two-digit classes: class 0 is [3, 6], class 1 is [4, 7]
        if self.classes is None:
            self._class_labels = np.array([0, 1], dtype=int)  # Two classes: 0 and 1
        else:
            self._class_labels = np.asarray(self.classes, dtype=int)
            if self._class_labels.ndim != 1 or self._class_labels.size != 2:
                raise ValueError("For bi-modal classification, `classes` must contain exactly two classes.")
        
        # Map the two class labels to their respective digit pairs
        self._label_map = {
            0: [3, 6],  # Class 0: randomly sample from 3 and 6
            1: [4, 7],  # Class 1: randomly sample from 4 and 7
        }

        # Check task configuration for binary task (only two classes)
        if self.task == "binary":
            if self.positive_classes is None or len(self.positive_classes) == 0:
                raise ValueError("For task='binary', you must provide positive_classes.")
            pos = set(int(d) for d in self.positive_classes)
            if not pos.issubset({0, 1}):  # Must be either class 0 or class 1
                raise ValueError("positive_classes must be either 0 or 1.")
            if self.classes is not None:
                kept_set = set(self._class_labels.tolist())
                if len(pos.intersection(kept_set)) == 0:
                    raise ValueError(
                        "Binary task: none of positive_classes are included in `classes`."
                    )

        # Set output dimension p
        if self.representation == "raw":
            self.p = self.p_raw
        else:
            if self.W is None:
                raise ValueError("representation='random_features' requires providing W.")
            W = np.asarray(self.W)
            if W.ndim != 2 or W.shape[1] != self.p_raw:
                raise ValueError(
                    f"W must have shape (m, p_raw) = (m, {self.p_raw}), got {W.shape}."
                )
            m = int(W.shape[0])
            if self.bias is not None:
                b = np.asarray(self.bias)
                if b.shape not in [(m,), (1, m)]:
                    raise ValueError(f"bias must have shape ({m},) (or (1,{m})), got {b.shape}")
            self.p = m

    @property
    def num_classes(self) -> int:
        # For binary classification, we have two classes (class 0 and class 1)
        return 2

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

    def _filter_and_encode_labels(self, X: Array, y_digits: Array) -> Tuple[Array, Array]:
        """Filter and map the labels to the corresponding class digits."""
        if self.task == "binary":
            pos = set(int(d) for d in self.positive_classes or [])
            if self.classes is not None:
                kept_set = set(self._class_labels.tolist())
                mask = np.array([int(yy) in kept_set for yy in y_digits], dtype=bool)
                X = X[mask]
                y_digits = y_digits[mask]
            y = np.where(np.isin(y_digits, list(pos)), 1.0, -1.0).astype(self.dtype, copy=False)
            return X, y

        if self.classes is None:
            y_idx = y_digits.astype(int, copy=False)
            return X, y_idx

        # Use the label map to randomly select between two digits per class
        mask = np.isin(y_digits, self._class_labels)
        Xf = X[mask]
        yf = y_digits[mask].astype(int, copy=False)

        # Map class labels to their respective two-digit pairs
        y_idx = np.array([np.random.choice(self._label_map[yy]) for yy in yf], dtype=int)
        return Xf, y_idx

    def _sample_raw_clean(self, n: int, rng: np.random.Generator) -> Tuple[Array, Array]:
        """Internal: fetch 'n' clean raw pixel vectors and labels."""
        X_all, y_all_digits = self._get_split(self.split)
        Xf, yf = self._filter_and_encode_labels(X_all, y_all_digits)

        if Xf.shape[0] == 0:
            raise ValueError("No samples available after filtering.")

        idx = rng.choice(Xf.shape[0], size=int(n), replace=bool(self.replace))
        return Xf[idx], yf[idx]

    def _add_noise(self, X: Array, rng: np.random.Generator) -> Array:
        """Add Gaussian noise if noise_std > 0."""
        if self.noise_std > 0:
            return X + rng.normal(scale=self.noise_std, size=X.shape).astype(self.dtype)
        return X

    # -----------------------------
    # Transform
    # -----------------------------
    def transform(self, X_raw: Array) -> Array:
        if self.W is None:
            raise ValueError("transform() requires W to be provided.")
        W = np.asarray(self.W, dtype=self.dtype)
        Z = X_raw @ W.T
        if self.bias is not None:
            b = np.asarray(self.bias, dtype=self.dtype).reshape(-1)
            Z = Z + b
        act = _get_activation(self.activation)
        Phi = act(Z)
        if self.feature_scale != 1.0:
            Phi = Phi * self.dtype.type(self.feature_scale)
        return Phi.astype(self.dtype, copy=False)

    # -----------------------------
    # Sampling API
    # -----------------------------
    def sample_raw(self, n: int, rng: Optional[np.random.Generator] = None) -> Tuple[Array, Array]:
        """
        Returns raw pixel vectors.
        Adds noise ONLY if representation="raw".
        """
        rng = np.random.default_rng() if rng is None else rng
        X, y = self._sample_raw_clean(n, rng)
        
        # We only add noise here if the target representation IS raw.
        # If user calls sample_raw() while representation="random_features", 
        # we return CLEAN pixels so they can be transformed correctly later.
        if self.representation == "raw":
            X = self._add_noise(X, rng)
            
        return X, y

    def sample_features(
        self, n: int, rng: Optional[np.random.Generator] = None
    ) -> Tuple[Array, Array]:
        """
        Returns random features.
        Adds noise ONLY if representation="random_features".
        """
        rng = np.random.default_rng() if rng is None else rng
        # Always get clean pixels first
        X_raw, y = self._sample_raw_clean(n, rng)
        # Transform clean pixels
        Phi = self.transform(X_raw)
        
        # Add noise if this is the active representation
        if self.representation == "random_features":
            Phi = self._add_noise(Phi, rng)
            
        return Phi, y

    def sample_both(
        self, n: int, rng: Optional[np.random.Generator] = None
    ) -> Tuple[Array, Array, Array]:
        """
        Returns (X_raw, Phi, y).
        Applies noise to whichever component is the active representation.
        """
        rng = np.random.default_rng() if rng is None else rng
        X_raw, y = self._sample_raw_clean(n, rng)
        
        Phi = None
        if self.W is not None:
            Phi = self.transform(X_raw)
        
        if self.representation == "raw":
            X_raw = self._add_noise(X_raw, rng)
        elif self.representation == "random_features" and Phi is not None:
            Phi = self._add_noise(Phi, rng)
            
        if Phi is None and self.representation == "random_features":
             raise ValueError("sample_both() requires W.")

        return X_raw, Phi, y

    def sample(self, n: int, rng: Optional[np.random.Generator] = None) -> Tuple[Array, Array]:
        if self.representation == "raw":
            return self.sample_raw(n, rng=rng)
        return self.sample_features(n, rng=rng)

    def sample_class(
        self, class_index: int, n: int, rng: Optional[np.random.Generator] = None
    ) -> Tuple[Array, Array]:
        rng = np.random.default_rng() if rng is None else rng
        X_all, y_all_digits = self._get_split(self.split)

        # ... (filtering logic same as before, condensed for brevity) ...
        if self.task == "binary":
            if class_index not in (0, 1):
                raise ValueError("binary task expects class_index in {0,1}.")
            pos = set(int(d) for d in self.positive_classes or [])
            y_sign = np.where(np.isin(y_all_digits, list(pos)), 1, -1)
            
            if self.classes is not None:
                kept_set = set(int(d) for d in self._class_labels.tolist())
                keep_mask = np.isin(y_all_digits, list(kept_set))
            else:
                keep_mask = np.ones_like(y_all_digits, dtype=bool)

            desired = 1 if class_index == 1 else -1
            mask = keep_mask & (y_sign == desired)
            Xc = X_all[mask]
            yc = y_sign[mask].astype(self.dtype, copy=False)
        else:
            K = self.num_classes
            if not (0 <= class_index < K):
                raise ValueError(f"class_index out of range: {class_index}")
            digit = int(self._class_labels[class_index]) if self.classes is not None else class_index
            mask = (y_all_digits == digit)
            Xc = X_all[mask]
            yc = np.full(Xc.shape[0], class_index, dtype=int)

        if Xc.shape[0] == 0:
            raise ValueError(f"No samples for class {class_index}.")

        # Sampling with replacement
        idx = rng.choice(Xc.shape[0], size=int(n), replace=bool(self.replace))
        X_batch = Xc[idx]
        y_batch = yc[idx]

        # Apply transforms and noise based on representation
        if self.representation == "raw":
            return self._add_noise(X_batch, rng), y_batch
        else:
            Phi_batch = self.transform(X_batch)
            return self._add_noise(Phi_batch, rng), y_batch

    # -----------------------------
    # Moment / parameter estimation
    # -----------------------------
    def _compute_class_params_for_representation(
        self, representation: Literal["raw", "random_features"]
    ) -> Dict[str, Any]:
        """
        Estimate mixture moments on stats_split.
        Adjusts Covariances to account for added noise_std.
        """
        X_all, y_all_digits = self._get_split(self.stats_split)
        Xf, yf = self._filter_and_encode_labels(X_all, y_all_digits)

        if Xf.shape[0] == 0:
            raise ValueError("No stats samples available.")

        # Logic to iterate classes (Binary vs Multiclass)
        # Note: Code structure matches previous, just extracting loop body for brevity
        if self.task == "binary":
            masks = [yf < 0, yf > 0] # class 0 (-1), class 1 (+1)
            y_vals = np.asarray([-1.0, 1.0], dtype=self.dtype)
        else:
            masks = [(yf == k) for k in range(self.num_classes)]
            y_vals = self.classes if self.classes is not None else list(range(self.num_classes))

        gamma = []
        mus = []
        covs = []

        total_samples = float(Xf.shape[0])

        for mask in masks:
            nk = int(mask.sum())
            gamma.append(float(nk) / total_samples if total_samples > 0 else 0.0)

            if nk == 0:
                dim = self.p_raw if representation == "raw" else self.p
                mus.append(np.zeros(dim, dtype=self.dtype))
                # Base regularization
                covs.append(self.cov_reg * np.eye(dim, dtype=self.dtype))
                continue

            Xc_raw = Xf[mask]
            Xc = Xc_raw if representation == "raw" else self.transform(Xc_raw)

            # 1. Estimate Empirical Moments (on clean data)
            if self.cov_kind == "full":
                mu, C = _moment_estimates_full(Xc, reg=self.cov_reg)
            else:
                mu, C = _moment_estimates_diag(Xc, reg=self.cov_reg)

            # 2. Add Noise Variance (Theory Correction)
            # If the sampling adds noise N(0, noise_std^2 * I), the covariance
            # of the output is Sigma_clean + noise_std^2 * I.
            # We only apply this if we are computing params for the active representation
            # OR if we assume noise_std applies to the requested representation context.
            # Here we apply it if noise_std > 0.
            if self.noise_std > 0:
                noise_var = self.noise_std ** 2
                if self.cov_kind == "diag":
                    # C is diagonal matrix (or vector depending on implementation, 
                    # but _moment_estimates_diag returns matrix here)
                    # We add to diagonal elements.
                    np.fill_diagonal(C, C.diagonal() + noise_var)
                else:
                    # Full covariance
                    C += noise_var * np.eye(C.shape[0], dtype=C.dtype)

            mus.append(mu)
            covs.append(C)

        return dict(
            p=int(mus[0].shape[0]),
            num_classes=len(gamma),
            gamma=np.array(gamma, dtype=self.dtype),
            mus=mus,
            covs=covs,
            y_values=y_vals,
            stats_split=self.stats_split,
            representation=representation,
        )

    def class_params_raw(self) -> Dict[str, Any]:
        if self._cached_params_raw is None:
            self._cached_params_raw = self._compute_class_params_for_representation("raw")
        return self._cached_params_raw

    def class_params_features(self) -> Dict[str, Any]:
        if self.W is None:
            raise ValueError("class_params_features() requires W.")
        if self._cached_params_rf is None:
            self._cached_params_rf = self._compute_class_params_for_representation("random_features")
        return self._cached_params_rf

    def class_params(self) -> Dict[str, Any]:
        if self.representation == "raw":
            return self.class_params_raw()
        return self.class_params_features()


@dataclass
class MNISTMergedDataModelGaussian(BaseDataModel):
    """
    Modified MNIST data model with bi-modal classes (two digits per class).
    
    Parameters
    ----------
    data_path: str
        Local .npz file containing MNIST arrays.
    noise_std: float
        Standard deviation of Gaussian noise added to the FINAL representation.
    representation: Literal["raw", "random_features"] = "raw"
    ...
    """

    data_path: str
    split: Literal["train", "test", "full"] = "test"
    stats_split: Literal["train", "test", "full"] = "train"
    representation: Literal["raw", "random_features"] = "raw"
    noise_std: float = 0.0
    pixel_scaling: Literal["uint8", "unit_interval"] = "unit_interval"
    dtype: Any = np.float32
    cov_kind: Literal["full", "diag"] = "full"
    cov_reg: float = 1e-6
    classes: Optional[Sequence[int]] = None  # Should be 0 and 1 as we are using bi-modal classification
    task: Literal["multiclass", "binary"] = "multiclass"
    positive_classes: Optional[Sequence[int]] = None
    replace: bool = True
    W: Optional[Array] = None
    bias: Optional[Array] = None
    activation: Literal["identity", "relu", "tanh", "cos", "sign_pm1"] = "identity"
    feature_scale: float = 1.0
    p_raw: int = field(init=False)
    p: int = field(init=False)
    _X_train: Array = field(init=False, repr=False)
    _y_train: Array = field(init=False, repr=False)
    _X_test: Array = field(init=False, repr=False)
    _y_test: Array = field(init=False, repr=False)
    _class_labels: Array = field(init=False, repr=False)
    _label_map: Dict[int, int] = field(init=False, repr=False)
    _cached_params_raw: Optional[Dict[str, Any]] = field(default=None, init=False, repr=False)
    _cached_params_rf: Optional[Dict[str, Any]] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        # Load data
        x_train, y_train, x_test, y_test = _load_mnist_npz(self.data_path)

        Xtr = _as_2d_flattened_uint8(x_train)
        Xte = _as_2d_flattened_uint8(x_test)

        self._X_train = _normalize_pixels(Xtr, self.pixel_scaling, np.dtype(self.dtype))
        self._y_train = np.asarray(y_train, dtype=int)

        self._X_test = _normalize_pixels(Xte, self.pixel_scaling, np.dtype(self.dtype))
        self._y_test = np.asarray(y_test, dtype=int)

        self.p_raw = int(self._X_train.shape[1])

        # Define classes as two-digit classes: class 0 is [3, 6], class 1 is [4, 7]
        if self.classes is None:
            self._class_labels = np.array([0, 1], dtype=int)  # Two classes: 0 and 1
        else:
            self._class_labels = np.asarray(self.classes, dtype=int)
            if self._class_labels.ndim != 1 or self._class_labels.size != 2:
                raise ValueError("For bi-modal classification, `classes` must contain exactly two classes.")
        
        # Map the two class labels to their respective digit pairs
        self._label_map = {
            0: [3, 6],  # Class 0: randomly sample from 3 and 6
            1: [4, 7],  # Class 1: randomly sample from 4 and 7
        }

        # Check task configuration for binary task (only two classes)
        if self.task == "binary":
            if self.positive_classes is None or len(self.positive_classes) == 0:
                raise ValueError("For task='binary', you must provide positive_classes.")
            pos = set(int(d) for d in self.positive_classes)
            if not pos.issubset({0, 1}):  # Must be either class 0 or class 1
                raise ValueError("positive_classes must be either 0 or 1.")
            if self.classes is not None:
                kept_set = set(self._class_labels.tolist())
                if len(pos.intersection(kept_set)) == 0:
                    raise ValueError(
                        "Binary task: none of positive_classes are included in `classes`."
                    )

        # Set output dimension p
        if self.representation == "raw":
            self.p = self.p_raw
        else:
            if self.W is None:
                raise ValueError("representation='random_features' requires providing W.")
            W = np.asarray(self.W)
            if W.ndim != 2 or W.shape[1] != self.p_raw:
                raise ValueError(
                    f"W must have shape (m, p_raw) = (m, {self.p_raw}), got {W.shape}."
                )
            m = int(W.shape[0])
            if self.bias is not None:
                b = np.asarray(self.bias)
                if b.shape not in [(m,), (1, m)]:
                    raise ValueError(f"bias must have shape ({m},) (or (1,{m})), got {b.shape}")
            self.p = m

    @property
    def num_classes(self) -> int:
        # For binary classification, we have two classes (class 0 and class 1)
        return 2

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

    def _filter_and_encode_labels(self, X: Array, y_digits: Array) -> Tuple[Array, Array]:
        """Filter and map the labels to the corresponding class digits."""
        if self.task == "binary":
            pos = set(int(d) for d in self.positive_classes or [])
            if self.classes is not None:
                kept_set = set(self._class_labels.tolist())
                mask = np.array([int(yy) in kept_set for yy in y_digits], dtype=bool)
                X = X[mask]
                y_digits = y_digits[mask]
            y = np.where(np.isin(y_digits, list(pos)), 1.0, -1.0).astype(self.dtype, copy=False)
            return X, y

        if self.classes is None:
            y_idx = y_digits.astype(int, copy=False)
            return X, y_idx

        # Use the label map to randomly select between two digits per class
        mask = np.isin(y_digits, self._class_labels)
        Xf = X[mask]
        yf = y_digits[mask].astype(int, copy=False)

        # Map class labels to their respective two-digit pairs
        y_idx = np.array([np.random.choice(self._label_map[yy]) for yy in yf], dtype=int)
        return Xf, y_idx

    def _sample_raw_clean(self, n: int, rng: np.random.Generator, use_gaussian: bool = True) -> Tuple[Array, Array]:
        """Internal: fetch 'n' clean raw pixel vectors and labels, optionally replace with Gaussian."""
        X_all, y_all_digits = self._get_split(self.split)
        Xf, yf = self._filter_and_encode_labels(X_all, y_all_digits)

        if Xf.shape[0] == 0:
            raise ValueError("No samples available after filtering.")

        # Choose the samples
        idx = rng.choice(Xf.shape[0], size=int(n), replace=bool(self.replace))
        X = Xf[idx]
        y = yf[idx]

        # --- NEW: Replace with Gaussian if requested ---
        if use_gaussian:
            # Compute empirical mean and covariance
            empirical_mu = np.mean(X, axis=0)
            empirical_C = np.cov(X, rowvar=False)

            # Sample Gaussian with the same statistics
            X = rng.multivariate_normal(empirical_mu, empirical_C, size=n)

        return X, y

    def _add_noise(self, X: Array, rng: np.random.Generator) -> Array:
        """Add Gaussian noise if noise_std > 0."""
        if self.noise_std > 0:
            return X + rng.normal(scale=self.noise_std, size=X.shape).astype(self.dtype)
        return X

    # -----------------------------
    # Transform
    # -----------------------------
    def transform(self, X_raw: Array) -> Array:
        if self.W is None:
            raise ValueError("transform() requires W to be provided.")
        W = np.asarray(self.W, dtype=self.dtype)
        Z = X_raw @ W.T
        if self.bias is not None:
            b = np.asarray(self.bias, dtype=self.dtype).reshape(-1)
            Z = Z + b
        act = _get_activation(self.activation)
        Phi = act(Z)
        if self.feature_scale != 1.0:
            Phi = Phi * self.dtype.type(self.feature_scale)
        return Phi.astype(self.dtype, copy=False)

    # -----------------------------
    # Sampling API
    # -----------------------------
    def sample_raw(self, n: int, rng: Optional[np.random.Generator] = None) -> Tuple[Array, Array]:
        """
        Returns raw pixel vectors.
        Adds noise ONLY if representation="raw".
        """
        rng = np.random.default_rng() if rng is None else rng
        X, y = self._sample_raw_clean(n, rng)
        
        # We only add noise here if the target representation IS raw.
        # If user calls sample_raw() while representation="random_features", 
        # we return CLEAN pixels so they can be transformed correctly later.
        if self.representation == "raw":
            X = self._add_noise(X, rng)
            
        return X, y

    def sample_features(
        self, n: int, rng: Optional[np.random.Generator] = None
    ) -> Tuple[Array, Array]:
        """
        Returns random features.
        Adds noise ONLY if representation="random_features".
        """
        rng = np.random.default_rng() if rng is None else rng
        # Always get clean pixels first
        X_raw, y = self._sample_raw_clean(n, rng)
        # Transform clean pixels
        Phi = self.transform(X_raw)
        
        # Add noise if this is the active representation
        if self.representation == "random_features":
            Phi = self._add_noise(Phi, rng)
            
        return Phi, y

    def sample_both(
        self, n: int, rng: Optional[np.random.Generator] = None
    ) -> Tuple[Array, Array, Array]:
        """
        Returns (X_raw, Phi, y).
        Applies noise to whichever component is the active representation.
        """
        rng = np.random.default_rng() if rng is None else rng
        X_raw, y = self._sample_raw_clean(n, rng)
        
        Phi = None
        if self.W is not None:
            Phi = self.transform(X_raw)
        
        if self.representation == "raw":
            X_raw = self._add_noise(X_raw, rng)
        elif self.representation == "random_features" and Phi is not None:
            Phi = self._add_noise(Phi, rng)
            
        if Phi is None and self.representation == "random_features":
             raise ValueError("sample_both() requires W.")

        return X_raw, Phi, y

    def sample(self, n: int, rng: Optional[np.random.Generator] = None) -> Tuple[Array, Array]:
        if self.representation == "raw":
            return self.sample_raw(n, rng=rng)
        return self.sample_features(n, rng=rng)

    def sample_class(
        self, class_index: int, n: int, rng: Optional[np.random.Generator] = None
    ) -> Tuple[Array, Array]:
        rng = np.random.default_rng() if rng is None else rng
        X_all, y_all_digits = self._get_split(self.split)

        # ... (filtering logic same as before, condensed for brevity) ...
        if self.task == "binary":
            if class_index not in (0, 1):
                raise ValueError("binary task expects class_index in {0,1}.")
            pos = set(int(d) for d in self.positive_classes or [])
            y_sign = np.where(np.isin(y_all_digits, list(pos)), 1, -1)
            
            if self.classes is not None:
                kept_set = set(int(d) for d in self._class_labels.tolist())
                keep_mask = np.isin(y_all_digits, list(kept_set))
            else:
                keep_mask = np.ones_like(y_all_digits, dtype=bool)

            desired = 1 if class_index == 1 else -1
            mask = keep_mask & (y_sign == desired)
            Xc = X_all[mask]
            yc = y_sign[mask].astype(self.dtype, copy=False)
        else:
            K = self.num_classes
            if not (0 <= class_index < K):
                raise ValueError(f"class_index out of range: {class_index}")
            digit = int(self._class_labels[class_index]) if self.classes is not None else class_index
            mask = (y_all_digits == digit)
            Xc = X_all[mask]
            yc = np.full(Xc.shape[0], class_index, dtype=int)

        if Xc.shape[0] == 0:
            raise ValueError(f"No samples for class {class_index}.")

        # Sampling with replacement
        idx = rng.choice(Xc.shape[0], size=int(n), replace=bool(self.replace))
        X_batch = Xc[idx]
        y_batch = yc[idx]

        # Apply transforms and noise based on representation
        if self.representation == "raw":
            return self._add_noise(X_batch, rng), y_batch
        else:
            Phi_batch = self.transform(X_batch)
            return self._add_noise(Phi_batch, rng), y_batch

    # -----------------------------
    # Moment / parameter estimation
    # -----------------------------
    def _compute_class_params_for_representation(
        self, representation: Literal["raw", "random_features"]
    ) -> Dict[str, Any]:
        """
        Estimate mixture moments on stats_split.
        Adjusts Covariances to account for added noise_std.
        """
        X_all, y_all_digits = self._get_split(self.stats_split)
        Xf, yf = self._filter_and_encode_labels(X_all, y_all_digits)

        if Xf.shape[0] == 0:
            raise ValueError("No stats samples available.")

        # Logic to iterate classes (Binary vs Multiclass)
        # Note: Code structure matches previous, just extracting loop body for brevity
        if self.task == "binary":
            masks = [yf < 0, yf > 0] # class 0 (-1), class 1 (+1)
            y_vals = np.asarray([-1.0, 1.0], dtype=self.dtype)
        else:
            masks = [(yf == k) for k in range(self.num_classes)]
            y_vals = self.classes if self.classes is not None else list(range(self.num_classes))

        gamma = []
        mus = []
        covs = []

        total_samples = float(Xf.shape[0])

        for mask in masks:
            nk = int(mask.sum())
            gamma.append(float(nk) / total_samples if total_samples > 0 else 0.0)

            if nk == 0:
                dim = self.p_raw if representation == "raw" else self.p
                mus.append(np.zeros(dim, dtype=self.dtype))
                # Base regularization
                covs.append(self.cov_reg * np.eye(dim, dtype=self.dtype))
                continue

            Xc_raw = Xf[mask]
            Xc = Xc_raw if representation == "raw" else self.transform(Xc_raw)

            # 1. Estimate Empirical Moments (on clean data)
            if self.cov_kind == "full":
                mu, C = _moment_estimates_full(Xc, reg=self.cov_reg)
            else:
                mu, C = _moment_estimates_diag(Xc, reg=self.cov_reg)

            # 2. Add Noise Variance (Theory Correction)
            # If the sampling adds noise N(0, noise_std^2 * I), the covariance
            # of the output is Sigma_clean + noise_std^2 * I.
            # We only apply this if we are computing params for the active representation
            # OR if we assume noise_std applies to the requested representation context.
            # Here we apply it if noise_std > 0.
            if self.noise_std > 0:
                noise_var = self.noise_std ** 2
                if self.cov_kind == "diag":
                    # C is diagonal matrix (or vector depending on implementation, 
                    # but _moment_estimates_diag returns matrix here)
                    # We add to diagonal elements.
                    np.fill_diagonal(C, C.diagonal() + noise_var)
                else:
                    # Full covariance
                    C += noise_var * np.eye(C.shape[0], dtype=C.dtype)

            mus.append(mu)
            covs.append(C)

        return dict(
            p=int(mus[0].shape[0]),
            num_classes=len(gamma),
            gamma=np.array(gamma, dtype=self.dtype),
            mus=mus,
            covs=covs,
            y_values=y_vals,
            stats_split=self.stats_split,
            representation=representation,
        )

    def class_params_raw(self) -> Dict[str, Any]:
        if self._cached_params_raw is None:
            self._cached_params_raw = self._compute_class_params_for_representation("raw")
        return self._cached_params_raw

    def class_params_features(self) -> Dict[str, Any]:
        if self.W is None:
            raise ValueError("class_params_features() requires W.")
        if self._cached_params_rf is None:
            self._cached_params_rf = self._compute_class_params_for_representation("random_features")
        return self._cached_params_rf

    def class_params(self) -> Dict[str, Any]:
        if self.representation == "raw":
            return self.class_params_raw()
        return self.class_params_features()
    
def _moment_estimates_full(X, reg):
    # X: (n, p)
    mu = np.mean(X, axis=0)
    # Centered X
    Xc = X - mu
    # Cov = (Xc.T @ Xc) / (n - 1)
    # We usually use n for MLE or n-1 for unbiased. Using n here for simplicity/stability
    n = X.shape[0]
    if n > 1:
        C = (Xc.T @ Xc) / (n - 1)
    else:
        C = np.zeros((X.shape[1], X.shape[1]), dtype=X.dtype)
    # Regularization
    C = C + reg * np.eye(X.shape[1], dtype=X.dtype)
    return mu, C

def _moment_estimates_diag(X, reg):
    mu = np.mean(X, axis=0)
    Xc = X - mu
    n = X.shape[0]
    if n > 1:
        # Variance along columns
        vars_ = np.var(X, axis=0, ddof=1)
    else:
        vars_ = np.zeros(X.shape[1], dtype=X.dtype)
    
    C = np.diag(vars_ + reg)
    return mu, C

# --- Placeholder for activation (assuming this exists in your codebase) ---
def _get_activation(name):
    if name == "identity": return lambda x: x
    if name == "relu": return lambda x: np.maximum(0, x)
    if name == "tanh": return np.tanh
    if name == "sign_pm1": return lambda x: np.sign(x)
    if name == "cos": return np.cos
    raise ValueError(f"Unknown activation {name}")

# --- Placeholder loading functions (assuming these exist) ---
# You must ensure these are imported or defined in your actual file
# from .utils import _load_mnist_npz, _as_2d_flattened_uint8, _normalize_pixels

def create_downsampling_matrix(target_m, original_dim=28):
    """
    Creates a matrix W of shape (target_m^2, original_dim^2).
    Multiplication W @ x extracts pixels to form a smaller image.
    
    It centers the grid to capture the middle of the digit.
    """
    stride = original_dim // target_m
    
    # Calculate offset to center the grid on the image
    # The grid covers: (target_m - 1) * stride + 1 pixels
    grid_span = (target_m - 1) * stride + 1
    start_offset = (original_dim - grid_span) // 2
    
    W = np.zeros((target_m * target_m, original_dim * original_dim))
    
    row_idx = 0
    
    # Loop over the target small grid coordinates
    for i in range(target_m):
        for j in range(target_m):
            # Map to original image coordinates
            orig_row = start_offset + i * stride
            orig_col = start_offset + j * stride
            
            # Convert to flat indices
            flat_idx_orig = orig_row * original_dim + orig_col
            flat_idx_target = row_idx
            
            # Set the entry to 1
            W[flat_idx_target, flat_idx_orig] = 1.0
            row_idx += 1
            
    return W




def _as_2d_flattened_uint8(images: Array) -> Array:
    """
    images: (N, 28, 28) or already flattened (N, 784).
    returns: (N, 784)
    """
    if images.ndim == 3:
        n, h, w = images.shape
        return images.reshape(n, h * w)
    if images.ndim == 2:
        return images
    raise ValueError(f"Expected images with ndim 2 or 3, got shape {images.shape}")


def _normalize_pixels(
    X: Array,
    pixel_scaling: Literal["uint8", "unit_interval"],
    dtype: np.dtype,
) -> Array:
    """
    pixel_scaling:
      - "uint8": keep in [0,255] but cast to dtype
      - "unit_interval": scale to [0,1] assuming 8-bit images (divide by 255 if needed)
    """
    X = X.astype(dtype, copy=False)
    if pixel_scaling == "uint8":
        return X
    if pixel_scaling == "unit_interval":
        # If already in [0,1], don't rescale; else divide by 255.
        maxv = float(np.max(X)) if X.size else 0.0
        if maxv > 1.5:
            return X / dtype.type(255.0)
        return X
    raise ValueError(f"Unknown pixel_scaling={pixel_scaling!r}")


def _moment_estimates_full(
    X: Array,
    reg: float = 0.0,
) -> Tuple[Array, Array]:
    """
    Population mean/cov (uses 1/N scaling):
      mu = E[x]
      C  = E[xx^T] - mu mu^T

    reg: adds reg*I to covariance for numerical stability.
    """
    X64 = np.asarray(X, dtype=np.float64)
    n, p = X64.shape
    if n == 0:
        raise ValueError("Cannot estimate moments from empty set.")
    mu = X64.mean(axis=0)
    exx = (X64.T @ X64) / float(n)
    C = exx - np.outer(mu, mu)
    # Symmetrize for numerical stability
    C = 0.5 * (C + C.T)
    if reg > 0:
        C = C + reg * np.eye(p, dtype=C.dtype)
    return mu.astype(X.dtype, copy=False), C.astype(X.dtype, copy=False)


def _moment_estimates_diag(
    X: Array,
    reg: float = 0.0,
) -> Tuple[Array, Array]:
    """
    Mean + diagonal covariance estimate.
    """
    X64 = np.asarray(X, dtype=np.float64)
    n, p = X64.shape
    if n == 0:
        raise ValueError("Cannot estimate moments from empty set.")
    mu = X64.mean(axis=0)
    ex2 = (X64**2).mean(axis=0)
    var = ex2 - mu**2
    var = np.maximum(var, 0.0)  # guard tiny negatives
    if reg > 0:
        var = var + reg
    C = np.diag(var)
    return mu.astype(X.dtype, copy=False), C.astype(X.dtype, copy=False)


def _relu(z: Array) -> Array:
    return np.maximum(z, 0.0)


def _identity(z: Array) -> Array:
    return z


def _sign_pm1(z: Array) -> Array:
    # map zeros to +1 for stability
    return np.where(z >= 0, 1.0, -1.0)


def _get_activation(
    activation: Literal["identity", "relu", "tanh", "cos", "sign_pm1"]
) -> Callable[[Array], Array]:
    if activation == "identity":
        return _identity
    if activation == "relu":
        return _relu
    if activation == "tanh":
        return np.tanh
    if activation == "cos":
        return np.cos
    if activation == "sign_pm1":
        return _sign_pm1
    raise ValueError(f"Unknown activation={activation!r}")


def _load_mnist_npz(path: str) -> Tuple[Array, Array, Array, Array]:
    """
    Load MNIST arrays from a local .npz file.

    Supports keys:
      - x_train, y_train, x_test, y_test (Keras convention)
      - X_train, y_train, X_test, y_test (capitalization variants)
    """
    with np.load(path) as f:
        keys = set(f.files)

        # Keras style
        if {"x_train", "y_train", "x_test", "y_test"} <= keys:
            x_train = f["x_train"]
            y_train = f["y_train"]
            x_test = f["x_test"]
            y_test = f["y_test"]
            return x_train, y_train, x_test, y_test

        # Capitalized variants
        if {"X_train", "y_train", "X_test", "y_test"} <= keys:
            x_train = f["X_train"]
            y_train = f["y_train"]
            x_test = f["X_test"]
            y_test = f["y_test"]
            return x_train, y_train, x_test, y_test

        # Some people store images/labels directly
        if {"images_train", "labels_train", "images_test", "labels_test"} <= keys:
            return (
                f["images_train"],
                f["labels_train"],
                f["images_test"],
                f["labels_test"],
            )

        raise ValueError(
            f"Unrecognized MNIST .npz format at {path!r}. "
            f"Available keys: {sorted(keys)}"
        )
    
