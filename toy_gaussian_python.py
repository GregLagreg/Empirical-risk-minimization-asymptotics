import sys
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm


# Make sure local modules are importable
ROOT = pathlib.Path().resolve() / "Empirical-risk-minimization-asymptotics"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# --- Import project code ---
import data_models as dm
import losses_regularizers as lr
import erm_theory as et
import data_augmentation as augm

rng_train = np.random.default_rng(0)
rng_test  = np.random.default_rng(1)

# ERM model

n_train = 50
n_test = 2000
p = 30
m = 1
sigma = 1


def generate_source_model(p, delta, sigma, gamma_pos=0.5):
    C = sigma**2*np.eye(p)
    e1 = np.zeros(p)
    e1[0] = 1.0
    mu_pos = delta * e1
    mu_neg = -delta * e1
    x_neg = dm.GaussianModel(p=p, mu=mu_neg, C=C)
    x_pos = dm.GaussianModel(p=p, mu=mu_pos, C=C)
    model = dm.MultiClassModel(
        components=[x_neg, x_pos],
        gamma=[1.0 - gamma_pos, gamma_pos],
        y_values=[-1.0, +1.0],
    )
    return model


def generate_target_model(p, delta, sigma, rotation, biais, gamma_pos=0.5):
    C = sigma**2*np.eye(p)  # Pas d'hypothèse ici sur la cov de e
    e1 = np.zeros(p)
    e1[0] = 1.0
    mu_pos = delta * e1
    mu_neg = -delta * e1
    mean_pos = (rotation @ mu_pos) + biais
    mean_neg = (rotation @ mu_neg) + biais
    x_neg = dm.GaussianModel(p=p, mu=mean_neg, C=C)
    x_pos = dm.GaussianModel(p=p, mu=mean_pos, C=C)
    model = dm.MultiClassModel(
        components=[x_neg, x_pos],
        gamma=[1.0 - gamma_pos, gamma_pos],
        y_values=[-1.0, +1.0],
    )
    return model


def rotation(theta, p):
    R = np.eye(p)
    R[0, 0] = np.cos(theta)
    R[0, 1] = -np.sin(theta)
    R[1, 0] = np.sin(theta)
    R[1, 1] = np.cos(theta)
    return R


# Déf des paramètres de l'ERM
loss1 = lr.LogisticLoss()
loss2 = lr.HingeLoss()
loss3 = lr.SquaredLoss()
lam = 0.1
reg = lr.QuadraticRegularizer(a=np.zeros(p), H=lam * np.eye(p))
theta_e = 1.2
s_e = 0.1
rho = 0.1
beta = 0.1

"""
Déf des paramètres de l'hôpital e,
la matrice de rotation R_e et le biais b_e,
en ayant fait le choix de générer les paramètres
theta_e et s_e selon des lois uniformes
"""
R_e = rotation(theta_e, p)
b_e = np.zeros(p)
b_e[1] = s_e

model_train = generate_source_model(sigma=sigma, p=p, delta=m)
model_test = generate_target_model(sigma=sigma, p=p, delta=m, rotation=R_e, biais=b_e)

# Cas 1 : logistic loss
trainer1 = et.ERMTrainer(model=model_train, loss=loss1, regularizer=reg)
theta_hat1, _, _ = trainer1.sample_theta_hat(n=n_train, rng=rng_train, solver_maxiter=180)
erreur_log = model_test.error_classif_emp([theta_hat1], n_test=n_test, rng=rng_test)
erreur_log_source = model_train.error_classif_emp([theta_hat1], n_test=n_test, rng=rng_test)

# Cas 2 : Hinge loss
trainer2 = et.ERMTrainer(model=model_train, loss=loss2, regularizer=reg)
theta_hat2, _, _ = trainer2.sample_theta_hat(n=n_train, rng=rng_train, solver_maxiter=180)
erreur_hinge = model_test.error_classif_emp([theta_hat2], n_test=n_test, rng=rng_test)
erreur_hinge_source = model_train.error_classif_emp([theta_hat2], n_test=n_test,
                                                    rng=rng_test)

# Cas 3 : Squared loss
trainer3 = et.ERMTrainer(model=model_train, loss=loss3, regularizer=reg)
theta_hat3, _, _ = trainer3.sample_theta_hat(n=n_train, rng=rng_train, solver_maxiter=180)
erreur_squared = model_test.error_classif_emp([theta_hat3], n_test=n_test, rng=rng_test)
erreur_squared_source = model_train.error_classif_emp([theta_hat3], n_test=n_test,
                                                      rng=rng_test)

print("Logistic error source:", erreur_log_source, "\n",
      "Hinge error source:", erreur_hinge_source, "\n",
      "Squared error source", erreur_squared_source, "\n")

print("Logistic error:", erreur_log, "\n",
      "Hinge error:", erreur_hinge, "\n",
      "Squared error", erreur_squared, "\n")


# Generalisation à la génération de plusieurs hopitaux
nb_hopit = 100
erreur_log_final = 0
erreur_hinge_final = 0
erreur_squared_final = 0
theta = theta_e
s = s_e
for i in range(0, nb_hopit):
    theta += 0.002
    s += 0.002
    R_e = rotation(theta, p)
    b_e = np.zeros(p)
    b_e[1] = s
    model_test = generate_target_model(sigma=sigma, p=p, delta=m, rotation=R_e,
                                       biais=b_e)
    erreur_log = model_test.error_classif_emp([theta_hat1], n_test=n_test, rng=rng_test)
    erreur_hinge = model_test.error_classif_emp([theta_hat2], n_test=n_test, rng=rng_test)
    erreur_squared = model_test.error_classif_emp([theta_hat3], n_test=n_test, rng=rng_test)
    erreur_log_final += erreur_log
    erreur_hinge_final += erreur_hinge
    erreur_squared_final += erreur_squared

print("Logistic error final:", erreur_log_final/nb_hopit, "\n",
      "Hinge error final:", erreur_hinge_final/nb_hopit, "\n",
      "Squared error final", erreur_squared_final/nb_hopit, "\n")


## Data augmentation model

# Genere epsilon
def random_rotation_from_identity(n, theta=1.0):
    """
    Generate a random rotation matrix at 'distance' theta from identity.
    theta=0 → Identity matrix
    theta large → Far from identity
    """
    # Generate random skew-symmetric matrix (tangent vector on SO(n))
    A = rng_train.standard_normal((n, n))
    K = (A - A.T) / 2  # skew-symmetric part

    # Scale by theta and exponentiate
    R = expm(theta * K)
    return R

# def la transfo aléatoire


def make_transform(rho, beta, rng):
    def transform(X):
        R_eps = random_rotation_from_identity(p, theta=rho)
        eps = R_eps - np.eye(p)
        eta = rng.uniform(-beta, beta, size=p)
        return X @ (np.eye(p) + eps).T + eta
    return transform


transform = make_transform(rho, beta, rng_train)

# Simulation

# Cas 1 : logistic loss
trainer1_augm = augm.DataAugmTrainer(model=model_train, loss=loss1, regularizer=reg,
                                    transform=transform)
theta_hat1_augm, _, _ = trainer1_augm.sample_theta_hat(n=n_train, rng=rng_train, solver_maxiter=180)
erreur_log_augm = model_test.error_classif_emp([theta_hat1_augm], n_test=n_test, rng=rng_test)
erreur_log_source_augm = model_train.error_classif_emp([theta_hat1_augm], n_test=n_test, rng=rng_test)

# Cas 2 : Hinge loss
trainer2_augm = augm.DataAugmTrainer(model=model_train, loss=loss2, regularizer=reg,
                                    transform=transform)
theta_hat2_augm, _, _ = trainer2_augm.sample_theta_hat(n=n_train, rng=rng_train, solver_maxiter=180)
erreur_hinge_augm = model_test.error_classif_emp([theta_hat2_augm], n_test=n_test, rng=rng_test)
erreur_hinge_source_augm = model_train.error_classif_emp([theta_hat2_augm], n_test=n_test,
                                                    rng=rng_test)

# Cas 3 : Squared loss
trainer3_augm = augm.DataAugmTrainer(model=model_train, loss=loss3, regularizer=reg,
                                    transform=transform)
theta_hat3_augm, _, _ = trainer3_augm.sample_theta_hat(n=n_train, rng=rng_train, solver_maxiter=180)
erreur_squared_augm = model_test.error_classif_emp([theta_hat3_augm], n_test=n_test, rng=rng_test)
erreur_squared_source_augm = model_train.error_classif_emp([theta_hat3_augm], n_test=n_test,
                                                      rng=rng_test)

print("Logistic error augm source:", erreur_log_source_augm, "\n",
      "Hinge error augm source:", erreur_hinge_source_augm, "\n",
      "Squared error augm source", erreur_squared_source_augm, "\n")

print("Logistic error augm:", erreur_log_augm, "\n",
      "Hinge error augm:", erreur_hinge_augm, "\n",
      "Squared error augm", erreur_squared_augm, "\n")

# Generalisation à la génération de plusieurs hopitaux
nb_hopit2 = 100
erreur_log_final_augm = 0
erreur_hinge_final_augm = 0
erreur_squared_final_augm = 0
theta = theta_e
s = s_e

for i in range(0, nb_hopit2):
    theta += 0.002
    s += 0.002
    R_e = rotation(theta, p)
    b_e = np.zeros(p)
    b_e[1] = s
    model_test = generate_target_model(sigma=sigma, p=p, delta=m, rotation=R_e,
                                       biais=b_e)
    erreur_log = model_test.error_classif_emp([theta_hat1_augm], n_test=n_test, rng=rng_test)
    erreur_hinge = model_test.error_classif_emp([theta_hat2_augm], n_test=n_test, rng=rng_test)
    erreur_squared = model_test.error_classif_emp([theta_hat3_augm], n_test=n_test, rng=rng_test)
    erreur_log_final_augm += erreur_log
    erreur_hinge_final_augm += erreur_hinge
    erreur_squared_final_augm += erreur_squared

print("Logistic error final:", erreur_log_final_augm/nb_hopit2, "\n",
      "Hinge error final:", erreur_hinge_final_augm/nb_hopit2, "\n",
      "Squared error final", erreur_squared_final_augm/nb_hopit2, "\n")

# Exportation des résultats

html = (
    "<!DOCTYPE html>\n"
    "<html>\n"
    "<head>\n"
    "<meta charset='utf-8'>\n"
    "<style>\n"
    "body { font-family: Arial, sans-serif; margin: 40px; }\n"
    "h2 { color: #333; }\n"
    ".params { background: #f5f5f5; padding: 10px; border-radius: 6px; margin-bottom: 20px; }\n"
    "table { border-collapse: collapse; width: 100%; }\n"
    "th, td { border: 1px solid #ccc; padding: 10px; text-align: center; }\n"
    "th { background: #4a90d9; color: white; }\n"
    "tr:nth-child(even) { background: #f9f9f9; }\n"
    "</style>\n"
    "</head>\n"
    "<body>\n"
    "<h2>Comparaison ERM vs Data Augmentation</h2>\n"
    "<div class='params'>"
    "<b>Paramètres :</b> "
    "n=" + str(n_train) + ", p=" + str(p) + ", m=" + str(m) + ", sigma=" + str(sigma) + ", "
    "theta_e=" + str(theta_e) + ", s_e=" + str(s_e) + ", rho=" + str(rho) + ", beta=" + str(beta) +
    "</div>\n"
    "<table>\n"
    "<tr><th>Cas</th><th>Loss</th><th>ERM</th><th>Augmentation</th></tr>\n"
    "<tr><td rowspan='3'>Source</td>"
    "<td>Logistic</td><td>" + str(round(erreur_log_source, 4)) + "</td><td>" + str(round(erreur_log_source_augm, 4)) + "</td></tr>\n"
    "<tr><td>Hinge</td><td>" + str(round(erreur_hinge_source, 4)) + "</td><td>" + str(round(erreur_hinge_source_augm, 4)) + "</td></tr>\n"
    "<tr><td>Squared</td><td>" + str(round(erreur_squared_source, 4)) + "</td><td>" + str(round(erreur_squared_source_augm, 4)) + "</td></tr>\n"
    "<tr><td rowspan='3'>1 hôpital cible</td>"
    "<td>Logistic</td><td>" + str(round(erreur_log, 4)) + "</td><td>" + str(round(erreur_log_augm, 4)) + "</td></tr>\n"
    "<tr><td>Hinge</td><td>" + str(round(erreur_hinge, 4)) + "</td><td>" + str(round(erreur_hinge_augm, 4)) + "</td></tr>\n"
    "<tr><td>Squared</td><td>" + str(round(erreur_squared, 4)) + "</td><td>" + str(round(erreur_squared_augm, 4)) + "</td></tr>\n"
    "<tr><td rowspan='3'>Moyenne 100 hôpitaux</td>"
    "<td>Logistic</td><td>" + str(round(erreur_log_final/nb_hopit, 4)) + "</td><td>" + str(round(erreur_log_final_augm/nb_hopit2, 4)) + "</td></tr>\n"
    "<tr><td>Hinge</td><td>" + str(round(erreur_hinge_final/nb_hopit, 4)) + "</td><td>" + str(round(erreur_hinge_final_augm/nb_hopit2, 4)) + "</td></tr>\n"
    "<tr><td>Squared</td><td>" + str(round(erreur_squared_final/nb_hopit, 4)) + "</td><td>" + str(round(erreur_squared_final_augm/nb_hopit2, 4)) + "</td></tr>\n"
    "</table>\n"
    "</body>\n"
    "</html>\n"
)

with open("resultats.html", "w") as f:
    f.write(html)

print("Tableau sauvegardé dans resultats.html")
