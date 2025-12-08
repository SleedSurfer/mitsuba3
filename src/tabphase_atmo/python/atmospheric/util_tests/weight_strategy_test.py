import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# --- CONFIG ---
r_mean = 100.0
sigma_g = 1.3
num_samples = 50
seed = 42

# --- SETUP DISTRIBUTION ---
def _compute_lognormal_params(radius_mean_um, radius_std_um):
    cv_squared = (radius_std_um / radius_mean_um) ** 2
    sigma_squared = np.log(1 + cv_squared)
    sigma = np.sqrt(sigma_squared)
    mu = np.log(radius_mean_um) - sigma_squared / 2.0
    return mu, sigma

# Calculate parameters
radius_std = r_mean * 0.5 # Assume some variance
mu_log, sigma_log = _compute_lognormal_params(r_mean, radius_std)
dist = stats.lognorm(s=sigma_log, scale=np.exp(mu_log))

# Generate ground truth curve for reference
x_truth = np.linspace(10, 300, 1000)
y_truth = dist.pdf(x_truth)

# --- METHOD 1: RANDOM SAMPLING (The Trap) ---
# This is what happens if you use .rvs() AND weight by .pdf()
np.random.seed(seed)
r_rand = dist.rvs(size=num_samples)
w_rand_doubledip = dist.pdf(r_rand) # <--- The "Double Dip"
w_rand_doubledip /= w_rand_doubledip.sum() # Normalize

# --- METHOD 2: GRID SAMPLING (The Fix) ---
# Spanning +/- 3 sigmas
lower = np.exp(mu_log - 3*sigma_log)
upper = np.exp(mu_log + 3*sigma_log)
r_grid = np.logspace(np.log10(lower), np.log10(upper), num_samples)
w_grid = dist.pdf(r_grid)
w_grid /= w_grid.sum()

# --- PLOTTING ---
plt.figure(figsize=(12, 6))

# 1. Ground Truth
plt.plot(x_truth, y_truth, 'k-', alpha=0.3, label='Theoretical PDF (Ground Truth)', linewidth=3)

# 2. Random Samples (Double Dipped)
# We scale the stem height to show relative importance
plt.stem(r_rand, w_rand_doubledip * (np.max(y_truth)/np.max(w_rand_doubledip)),
         linefmt='r-', markerfmt='ro', basefmt=' ', label='Old: Random + PDF Weighted (Biased)')

# 3. Grid Samples (Correct)
# Offset slightly so you can see them
plt.stem(r_grid, w_grid * (np.max(y_truth)/np.max(w_grid)),
         linefmt='g-', markerfmt='go', basefmt=' ', label='New: Grid + PDF Weighted (Stable)')

plt.title(f"Sampling Strategy Comparison (N={num_samples})")
plt.xlabel("Particle Radius (microns)")
plt.ylabel("Relative Weight / Importance")
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(0, 300)

plt.tight_layout()
plt.show()