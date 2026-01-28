# -*- coding: utf-8 -*-
"""
Created on Tue Jan 27 15:04:35 2026

@author: m00633566
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.neighbors import KernelDensity  # For Kernel Density Estimation
from data_models import MNISTDataModel, create_downsampling_matrix
from losses_regularizers import LogisticLoss, QuadraticRegularizer, PseudoHuberRegularizer
from erm_theory import ERMTrainer, TheoryFixedPointSolver

# =========================
# CONFIGURE FIGURE VISIBILITY
# =========================
plt.rcParams.update({
    'font.size': 12,          # Base font size
    'axes.labelsize': 14,     # x/y labels
    'axes.titlesize': 16,     # subplot titles
    'legend.fontsize': 12,    # legend text
    'lines.linewidth': 2,     # curve thickness
    'xtick.labelsize': 12,    # x-ticks
    'ytick.labelsize': 12     # y-ticks
})

# =========================
# Load MNIST Data and Preprocess
# =========================
m = 30  # Downsampling dimension
W_downsample = create_downsampling_matrix(m)
num_trials = 2000

# Model to use for raw data (binary classification between class 0 and 1)
model_raw = MNISTDataModel(
    data_path="mnist.npz",
    split="test",
    stats_split="train",
    noise_std=0.000,
    representation="raw",
    classes=[3, 0],  # Binary classification: class 0 and class 1
    task="multiclass",
)

# Model to use for feature-based data (random features)
model = MNISTDataModel(
    data_path="mnist.npz",
    split="test",
    stats_split="train",
    noise_std=0.000,
    representation="random_features",
    classes=[3, 0],  # Binary classification: class 0 and class 1
    task="multiclass",
    W=W_downsample,
    activation="identity",  # Identity function for linear transformation
)

# data_path = 'path_to_your_data.csv'  # Provide the path to your dataset
# model = BinaryClassificationDataModel(data_path=data_path, representation="random_features", noise_std=0.01)


# =========================
# Loss and Regularizer
# =========================
loss = LogisticLoss(max_iter=500, tol=1e-10)

# =========================
# Train the Empirical Model (ERM)
# =========================
n = 10 * 6  # Example size for training
raw = True
if raw:
    dimension = 784
else:
    dimension = m * m
# reg = QuadraticRegularizer(a=np.zeros(dimension), H=2 * np.eye(dimension))  # Regularizer
reg = PseudoHuberRegularizer(lam = 2,  delta = 0.5)
trainer = ERMTrainer(model=model, loss=loss, regularizer=reg)

# Run empirical trials
emp = trainer.run_trials(
    n_train=n,
    num_trials=2000,
    rng=np.random.default_rng(),
    solver_maxiter=2000,
    tol=1e-6,
    method="L-BFGS-B",
    verbose=True,
)

# =========================
# Theory-based Model
# =========================
# solver = TheoryFixedPointSolver(
#     model=model,
#     loss=loss,
#     regularizer=reg,
#     n_train=n,
#     mc_samples=8000,
#     rng=np.random.default_rng(123),
# )

# Solve the theoretical fixed point
# th = solver.solve(
#     max_iter=1000,
#     tol=1e-4,
#     damping=0.1,
#     verbose=False,
# )

# =========================
# Collect Empirical and Theoretical Scores for both classes
# =========================
num_trials_th = 20000
cl = [0, 1]  # Loop over class 0 and class 1
scores = {}
scores_th = {}

# Collect empirical and theoretical decision scores for both classes
for c in cl:
    rng = np.random.default_rng()
    Xte, yte = trainer.model.sample_class(c, num_trials, rng=rng)
    scores[c] = np.array([Xte[i] @ trainer.thetas[i] for i in range(num_trials)])

    # Collect theoretical decision scores
#     z_samples = rng.standard_normal(size=num_trials_th)
#     Xtetst, ytetst = trainer.model.sample_class(c, num_trials_th, rng=rng)
#     scores_th[c] = Xtetst @ th["mu_star"] + th["alpha"][c] * z_samples

# print(th['converged'])
# =========================
# Plotting the Histogram and Curves
# =========================
plt.figure(figsize=(12, 6))

# Define x-values for plotting the Gaussian fits
x_vals = np.linspace(-8, 8, 1000)

# Plot empirical histograms and theoretical Gaussian fits
for c in cl:
    # Empirical data histogram (class 0 and 1)
    plt.hist(scores[c], bins=50, density=True, alpha=0.7, color='cyan' if c == 0 else 'orange', 
             edgecolor='black', label=f'Empirical (Class {c})', linewidth=1.5)

    # Theoretical data (using Kernel Density Estimation for smooth curve)
#     kde = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(scores_th[c].reshape(-1, 1))  # KDE smooth curve
#     log_dens = kde.score_samples(x_vals.reshape(-1, 1))
#     plt.plot(x_vals, np.exp(log_dens), color='blue', linewidth=2, label=f'Theoretical Curve (Class {c})')

    # Plot Gaussian PDFs for comparison (red)
#     mean_class = np.mean(scores_th[c])
#     std_class = np.std(scores_th[c])
#     plt.plot(x_vals, norm.pdf(x_vals, loc=mean_class, scale=std_class), 
#              color='red', linewidth=2, linestyle='--', label=f'Gaussian Fit (Class {c})')

# Add title, labels, and legend
plt.title("Empirical vs Theoretical Distribution of Decision Scores")
plt.xlabel('Decision Score')
plt.ylabel('Density')
plt.legend(loc='upper right')

# Adjust X-axis for better visibility
# plt.xlim(0, 3)

# Display the plot
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
