from ml_model import nn
from numerical_model import lorenz96

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use 'Agg' for non-GUI backend

# Time step for the ML model; use the same for the numerical integration
dt = 0.05
n_steps = 1000

# Generate a random state
index = np.arange(40)
x0 = np.random.randn(40)
observed = solve_ivp(lorenz96, [0, 0.2], x0, t_eval=np.linspace(0, 0.2, 11)).y.T[-1]
gam = 0.5
observed = observed + gam * np.std(observed) * np.random.randn(40)
observed = observed[::2]

def Phi(u1, obs, h=40//2, gamma=gam):
    diff = u1.reshape(-1) - obs.reshape(-1)
    return 1/(2*gamma**2)*h*np.linalg.norm(diff)**2

def ml_model(v, nstep=4):
    for t in range(nstep):
        v = nn._smodel.predict(v.reshape((1, 40, 1)))[0, :, 0]
    return v

u = np.zeros(40,)
# u1 = solve_ivp(lorenz96, [0, 0.2], u, t_eval=np.linspace(0, 0.2, 11)).y.T[-1]
u1 = ml_model(u)
u1 = u1[::2]

beta = 0.05
samples = []
for i in range(n_steps):
    v = np.sqrt(1-beta**2)*u + beta * np.random.randn(40)
    # v1 = solve_ivp(lorenz96, [0, 0.2], v, t_eval=np.linspace(0, 0.2, 11)).y.T[-1]
    v1 = ml_model(v)
    v1 = v1[::2]

    a = min(1, np.exp(Phi(u1, observed) - Phi(v1, observed) ))
    if np.random.rand(1) < a:
        u = v
        u1 = v1
    samples.append(u)

print(n_steps, len(samples))
samples = np.stack(samples, axis=0)
posterior_mean = np.mean(samples, axis=0)
diff=np.linalg.norm(posterior_mean-x0)

# Create the plot
plt.figure(figsize=(10, 6))

# Plotting the lines with enhanced styles
plt.plot(index, x0, label="Truth", color="blue", linestyle='-', linewidth=2)
plt.plot(index, posterior_mean, label="Posterior Mean", color="red", linestyle='--', linewidth=2)

# Adding legend with a better location
plt.legend(loc="upper left", fontsize=12)

# Adding title and labels with increased font size
plt.title("RK45 Solver, Partially Observed, Noise Level = 0.5", fontsize=16)
plt.xlabel("Index", fontsize=14)
plt.ylabel("Values", fontsize=14)

plt.text(0.82, 0.95, f'L2 Norm = {diff:.2f}', transform=plt.gca().transAxes, fontsize=12,
         verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))

# Enhancing grid for better readability
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Saving the plot with a tight layout
plt.tight_layout()
plt.savefig("problem2_noisy_halfobs.pdf", dpi=300)

# Display the plot
plt.show()
