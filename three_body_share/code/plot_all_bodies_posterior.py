import os
os.environ["KERAS_BACKEND"] = "torch"

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import keras
import torch

# Import custom modules
from diagnostics import load_test_data, read_lambda_median, sample_posterior
from device import get_device

def main():
    device = get_device()
    theta_all, obs_noisy_all, t0_all = load_test_data()
    lambda_median = read_lambda_median("results/lyapunov.txt")
    
    # Sort by t0 to find a late measurement (hard case due to chaos)
    lateness_all = t0_all * lambda_median
    late_idx = np.argmax(lateness_all)

    theta_true = theta_all[late_idx]
    obs = obs_noisy_all[late_idx:late_idx+1]
    t0_val = t0_all[late_idx:late_idx+1].reshape(-1, 1)
    
    print(f"Selected test sample index {late_idx} with lateness = {lateness_all[late_idx]:.2f}")
    
    model = keras.saving.load_model("models/trained_model.keras")
    
    num_draws = 2000
    print("Sampling posterior...")
    estimates = sample_posterior(model, obs, t0_val, num_draws)
    # estimates shape: (1, num_draws, 8)
    samples = estimates[0]
    
    # Extract positions for body 1 and 2
    # theta order: x1, y1, x2, y2, vx1, vy1, vx2, vy2
    x1_samples = samples[:, 0]
    y1_samples = samples[:, 1]
    x2_samples = samples[:, 2]
    y2_samples = samples[:, 3]
    
    # True positions
    x1_true = theta_true[0]
    y1_true = theta_true[1]
    x2_true = theta_true[2]
    y2_true = theta_true[3]
    
    # Masses
    m1, m2, m3 = 1.0 / 6.0, 2.0 / 6.0, 3.0 / 6.0
    
    # Calculate body 3 positions (COM constraint)
    x3_samples = -(m1 * x1_samples + m2 * x2_samples) / m3
    y3_samples = -(m1 * y1_samples + m2 * y2_samples) / m3
    x3_true = -(m1 * x1_true + m2 * x2_true) / m3
    y3_true = -(m1 * y1_true + m2 * y2_true) / m3
    
    plt.figure(figsize=(8, 8))
    
    # Plot posteriors
    plt.scatter(x1_samples, y1_samples, alpha=0.1, color='blue', label='Body 1 Posterior', s=10)
    plt.scatter(x2_samples, y2_samples, alpha=0.1, color='red', label='Body 2 Posterior', s=10)
    plt.scatter(x3_samples, y3_samples, alpha=0.1, color='green', label='Body 3 Posterior', s=10)
    
    # Plot ground truth
    plt.plot(x1_true, y1_true, 'x', color='black', markersize=10, markeredgewidth=3, label='Ground Truth')
    plt.plot(x2_true, y2_true, 'x', color='black', markersize=10, markeredgewidth=3)
    plt.plot(x3_true, y3_true, 'x', color='black', markersize=10, markeredgewidth=3)
    
    plt.title(f"Initial State Posterior for All 3 Bodies (Lateness: {lateness_all[late_idx]:.2f})")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.axis("equal")
    plt.grid(True, linestyle='--', alpha=0.6)
    
    out_path = "results/all_bodies_initial_posterior.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"Saved plot to {out_path}")

if __name__ == "__main__":
    main()
