import numpy as np
import matplotlib.pyplot as plt
from simulator import get_masses, EPSILON, integrate_system
from prior import sample_initial_condition

# settings for this measurement, not the final dataset prior
D0 = 1e-8
T_MAX = 10.0
N_TIMES = 400
TARGET_N = 20
MAX_ATTEMPTS = 60
BASE_SEED = 42

def measure_divergence(rng, state0, masses, d0, t_max, n_times):
    # clones state0 with a random direction perturbation of size d0,
    # integrates both, and returns the phase space distance over time
    times = np.linspace(0.0, t_max, n_times)
    direction = rng.normal(0.0, 1.0, size=12)
    direction = direction / np.sqrt(np.sum(direction ** 2))
    state0_perturbed = state0 + d0 * direction

    traj_a, success_a = integrate_system(state0, times, masses, EPSILON)
    traj_b, success_b = integrate_system(state0_perturbed, times, masses, EPSILON)

    if not success_a or not success_b:
        return times, None
    if np.any(np.isnan(traj_a)) or np.any(np.isnan(traj_b)):
        return times, None

    distances = np.zeros(n_times)
    for k in range(n_times):
        diff = traj_a[k] - traj_b[k]
        distances[k] = np.sqrt(np.sum(diff ** 2))
    return times, distances

def fit_lambda(times, distances, d0):
    # fits the slope of ln d(t) versus t in the clean exponential region,
    # defined as distances between 2*d0 (past the initial transient) and
    # 0.3 (before the bounded system saturates)
    lower = 2.0 * d0
    upper = 0.3
    mask_indices = []
    for k in range(len(times)):
        if distances[k] > lower and distances[k] < upper:
            mask_indices.append(k)

    if len(mask_indices) < 5:
        return None, None, mask_indices

    mask_indices = np.array(mask_indices)
    t_fit = times[mask_indices]
    log_d_fit = np.log(distances[mask_indices])
    slope, intercept = np.polyfit(t_fit, log_d_fit, 1)
    return slope, intercept, mask_indices

def run_lyapunov_measurement():
    masses = get_masses()
    rng = np.random.default_rng(BASE_SEED)

    lambda_values = []
    stored_curves = []
    n_attempted = 0

    while len(lambda_values) < TARGET_N and n_attempted < MAX_ATTEMPTS:
        n_attempted = n_attempted + 1
        state0 = sample_initial_condition(rng, masses)
        times, distances = measure_divergence(rng, state0, masses, D0, T_MAX, N_TIMES)
        if distances is None:
            continue

        slope, intercept, mask_indices = fit_lambda(times, distances, D0)
        if slope is None or slope <= 0:
            continue

        lambda_values.append(slope)
        if len(stored_curves) < 6:
            stored_curves.append({
                "times": times,
                "distances": distances,
                "slope": slope,
                "intercept": intercept,
                "mask_indices": mask_indices,
            })

    lambda_values = np.array(lambda_values)
    n_batch = len(lambda_values)
    print("draws attempted:", n_attempted)
    print("valid lambda measurements:", n_batch)
    print("lambda values:", lambda_values)

    plt.figure(figsize=(8, 6))
    for idx, curve in enumerate(stored_curves):
        log_d = np.log(curve["distances"])
        plt.plot(curve["times"], log_d, alpha=0.6, label="draw " + str(idx))
        t_fit = curve["times"][curve["mask_indices"]]
        fit_line = curve["slope"] * t_fit + curve["intercept"]
        plt.plot(t_fit, fit_line, "k--", linewidth=1)
    plt.xlabel("t")
    plt.ylabel("ln d(t)")
    plt.title("divergence of nearby trajectories, dashed lines are the fits")
    plt.legend()
    plt.savefig("results/divergence.png")
    print("saved plot to results/divergence.png")

    if n_batch == 0:
        print("gate failed: no draw produced a clear exponential growth region")
        return False

    lambda_median = np.median(lambda_values)
    lambda_min = np.min(lambda_values)
    lambda_max = np.max(lambda_values)
    ratio = lambda_max / lambda_min
    print("lambda median:", lambda_median)
    print("lambda min:", lambda_min, "lambda max:", lambda_max, "ratio:", ratio)

    gate_pass = True
    if n_batch < TARGET_N * 0.5:
        gate_pass = False
        print("gate check failed: fewer than half of the target draws gave a clear exponential region")
    if lambda_min <= 0:
        gate_pass = False
        print("gate check failed: a non positive lambda was found")
    if ratio > 10.0:
        gate_pass = False
        print("gate check failed: lambda values vary by more than a factor of 10")

    if not gate_pass:
        print("GATE FAILED, stopping, results/lyapunov.txt was not written")
        return False

    print("gate passed")
    T = 3.0 / lambda_median
    K = 10
    obs_times = np.linspace(T / K, T, K)

    f = open("results/lyapunov.txt", "w")
    f.write("lambda_median " + str(lambda_median) + "\n")
    f.write("lambda_all " + ",".join(str(v) for v in lambda_values) + "\n")
    f.write("T " + str(T) + "\n")
    f.write("K " + str(K) + "\n")
    f.write("obs_times " + ",".join(str(v) for v in obs_times) + "\n")
    f.write("d0 " + str(D0) + "\n")
    f.write("n_batch " + str(n_batch) + "\n")
    f.close()
    print("saved results/lyapunov.txt")
    print("T:", T)
    print("obs_times:", obs_times)
    return True

if __name__ == "__main__":
    run_lyapunov_measurement()
