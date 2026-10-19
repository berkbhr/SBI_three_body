import os
# the keras backend must be chosen before bayesflow is imported
os.environ["KERAS_BACKEND"] = "torch"

import gc
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import keras
import torch
import bayesflow as bf
from device import get_device

# theta order per data_contracts.md: x1, y1, x2, y2, vx1, vy1, vx2, vy2
PARAM_NAMES = ["x1", "y1", "x2", "y2", "vx1", "vy1", "vx2", "vy2"]
NUM_POSTERIOR_DRAWS = 100

# chaos summary bins in units of 1/lambda; t0 ranges over [0, 6/lambda], so
# lateness = t0 * lambda ranges over [0, 6]
LATENESS_BIN_EDGES = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
MIN_BIN_COUNT = 30

def load_test_data():
    # anti-leak rule: the network inputs are the noisy observation and the known
    # window offset t0; theta is the true target used only to score the posterior
    data = np.load("data/test.npz")
    theta = data["theta"].astype("float32")
    obs_noisy = data["obs_noisy"].astype("float32")
    t0 = data["t0"].astype("float32")
    return theta, obs_noisy, t0

def read_lambda_median(path):
    # results/lyapunov.txt is never retyped by hand
    f = open(path, "r")
    lambda_median = None
    for line in f:
        line = line.strip()
        if line.startswith("lambda_median "):
            lambda_median = float(line.split(" ", 1)[1])
    f.close()
    return lambda_median

def sample_posterior(model, obs_window, t0_window, num_draws):
    # both the observation and the known offset t0 are conditions
    samples = model.sample(
        conditions={"obs_noisy": obs_window, "t0": t0_window}, num_samples=num_draws
    )
    theta_samples = np.array(samples["theta"])
    del samples
    gc.collect()
    torch.cuda.empty_cache()
    return theta_samples

def plot_contraction_bars(values, variable_names, path):
    # bayesflow posterior contraction per parameter, 1 = full contraction,
    # 0 = posterior as wide as the prior
    plt.figure(figsize=(8, 6))
    plt.bar(variable_names, values, color="#132a70")
    plt.ylim(0, 1)
    plt.xlabel("parameter")
    plt.ylabel("posterior contraction")
    plt.title("posterior contraction per parameter (all test simulations)")
    plt.savefig(path)
    plt.close()

def plot_chaos_summary(latenesses, contractions, calib_errors, path):
    fig, ax1 = plt.subplots(figsize=(8, 6))
    ax1.plot(latenesses, contractions, "o-", color="#132a70", label="posterior contraction")
    ax1.set_xlabel("window offset t0 (elapsed 1/lambda units)")
    ax1.set_ylabel("posterior contraction (mean over parameters)", color="#132a70")
    ax1.set_ylim(0, 1)

    ax2 = ax1.twinx()
    ax2.plot(latenesses, calib_errors, "s-", color="#8f2727", label="calibration error")
    ax2.set_ylabel("calibration error (mean over parameters)", color="#8f2727")
    ax2.set_ylim(0, max(0.2, max(calib_errors) * 1.2))

    plt.title("contraction and calibration versus observation lateness (binned by t0)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)

def main():
    device = get_device()

    theta, obs_noisy, t0 = load_test_data()
    n_test = theta.shape[0]
    print("loaded", n_test, "test simulations")

    lambda_median = read_lambda_median("results/lyapunov.txt")
    print("lambda_median:", lambda_median)

    model = keras.saving.load_model("models/trained_model.keras")
    print("loaded models/trained_model.keras on", device)

    # sample the posterior once for the whole test set, using each simulation's
    # own observation and its own known offset t0
    print("sampling posterior for all test simulations...")
    t0_col = t0.reshape(-1, 1)
    estimates = sample_posterior(model, obs_noisy, t0_col, NUM_POSTERIOR_DRAWS)

    print("building results/calibration_ecdf.png")
    fig_ecdf = bf.diagnostics.calibration_ecdf(
        estimates=estimates, targets=theta, variable_names=PARAM_NAMES
    )
    fig_ecdf.savefig("results/calibration_ecdf.png")
    plt.close(fig_ecdf)

    print("building results/recovery.png")
    fig_recovery = bf.diagnostics.recovery(
        estimates=estimates, targets=theta, variable_names=PARAM_NAMES
    )
    fig_recovery.savefig("results/recovery.png")
    plt.close(fig_recovery)

    print("building results/contraction.png")
    contraction_full = bf.diagnostics.posterior_contraction(
        estimates=estimates, targets=theta, variable_names=PARAM_NAMES
    )
    plot_contraction_bars(contraction_full["values"], PARAM_NAMES, "results/contraction.png")

    # chaos summary: bin the test simulations by how late their window starts,
    # lateness = t0 * lambda in units of 1/lambda. Every simulation is in
    # distribution because the model trained on the same t0 range, so this
    # reads how identifiability changes with observation lateness honestly.
    print("building results/chaos_summary.png, binning by t0...")
    lateness_all = t0 * lambda_median

    latenesses = []
    contractions = []
    calib_errors = []
    for b in range(len(LATENESS_BIN_EDGES) - 1):
        lo = LATENESS_BIN_EDGES[b]
        hi = LATENESS_BIN_EDGES[b + 1]
        if b == len(LATENESS_BIN_EDGES) - 2:
            mask = (lateness_all >= lo) & (lateness_all <= hi)
        else:
            mask = (lateness_all >= lo) & (lateness_all < hi)
        n_in_bin = int(np.sum(mask))
        if n_in_bin < MIN_BIN_COUNT:
            print("  bin", lo, "-", hi, "skipped, only", n_in_bin, "simulations")
            continue

        est_bin = estimates[mask]
        tgt_bin = theta[mask]
        contraction_bin = bf.diagnostics.posterior_contraction(estimates=est_bin, targets=tgt_bin)
        calib_bin = bf.diagnostics.calibration_error(estimates=est_bin, targets=tgt_bin)

        center = 0.5 * (lo + hi)
        contraction_mean = float(np.mean(contraction_bin["values"]))
        calib_error_mean = float(np.mean(calib_bin["values"]))

        latenesses.append(center)
        contractions.append(contraction_mean)
        calib_errors.append(calib_error_mean)
        print(
            "  bin", lo, "-", hi, "center", center, "n", n_in_bin,
            "contraction", contraction_mean, "calibration_error", calib_error_mean,
        )

    plot_chaos_summary(latenesses, contractions, calib_errors, "results/chaos_summary.png")

    print("all four figures saved to results/")

if __name__ == "__main__":
    main()
