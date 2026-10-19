import os
import time
import numpy as np
from simulator import get_masses, EPSILON, integrate_system, total_energy

# acceptance thresholds
# collision_radius was retuned from 0.05 to 0.03 after the first prior check
# showed a rejection rate near 80 percent, almost entirely from collisions;
# see WORKLOG.md for the exploration. Still 1.5x the softening length, so
# accepted trajectories stay outside the softening-dominated regime.
ESCAPE_RADIUS = 10.0
COLLISION_RADIUS = 0.03
ENERGY_DRIFT_THRESHOLD = 1e-8

# number of points used to check that a trajectory stays bounded over the
# base window [0, T]. For a shifted window the integration span [0, t0 + T] is
# longer, so the actual point count is scaled up from this to keep the same
# time resolution as the original [0, T] check (see evaluate_initial_condition).
FINE_N = 300

def sample_initial_condition(rng, masses):
    # positions: each body uniform in a disc, COM subtracted, then rescaled
    # so the mean pairwise separation is 1
    pos = np.zeros((3, 2))
    for i in range(3):
        radius = np.sqrt(rng.uniform(0.0, 1.0))
        angle = rng.uniform(0.0, 2.0 * np.pi)
        pos[i][0] = radius * np.cos(angle)
        pos[i][1] = radius * np.sin(angle)

    com_pos = masses[0] * pos[0] + masses[1] * pos[1] + masses[2] * pos[2]
    for i in range(3):
        pos[i] = pos[i] - com_pos

    mean_sep = 0.0
    n_pairs = 0
    for i in range(3):
        for j in range(i + 1, 3):
            diff = pos[j] - pos[i]
            mean_sep = mean_sep + np.sqrt(diff[0] ** 2 + diff[1] ** 2)
            n_pairs = n_pairs + 1
    mean_sep = mean_sep / n_pairs
    pos = pos / mean_sep

    # potential energy at these positions, used to tune the velocity scale
    # so the virial ratio 2T / |W| is of order one
    potential = 0.0
    for i in range(3):
        for j in range(i + 1, 3):
            diff = pos[j] - pos[i]
            dist = np.sqrt(diff[0] ** 2 + diff[1] ** 2 + EPSILON ** 2)
            potential = potential - masses[i] * masses[j] / dist

    sigma_v = np.sqrt(abs(potential) / 2.0)
    vel = rng.normal(0.0, sigma_v, size=(3, 2))
    com_vel = masses[0] * vel[0] + masses[1] * vel[1] + masses[2] * vel[2]
    for i in range(3):
        vel[i] = vel[i] - com_vel

    state = np.zeros(12)
    state[0:2] = pos[0]
    state[2:4] = pos[1]
    state[4:6] = pos[2]
    state[6:8] = vel[0]
    state[8:10] = vel[1]
    state[10:12] = vel[2]
    return state

def compute_at_t0(state0, masses):
    # energy, angular momentum and virial ratio at t = 0
    pos = state0[0:6].reshape(3, 2)
    vel = state0[6:12].reshape(3, 2)
    n = len(masses)

    kinetic = 0.0
    for i in range(n):
        speed2 = vel[i][0] ** 2 + vel[i][1] ** 2
        kinetic = kinetic + 0.5 * masses[i] * speed2

    potential = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            diff = pos[j] - pos[i]
            dist = np.sqrt(diff[0] ** 2 + diff[1] ** 2 + EPSILON ** 2)
            potential = potential - masses[i] * masses[j] / dist

    energy0 = kinetic + potential
    virial0 = 2.0 * kinetic / abs(potential)

    angmom0 = 0.0
    for i in range(n):
        angmom0 = angmom0 + masses[i] * (pos[i][0] * vel[i][1] - pos[i][1] * vel[i][0])

    return energy0, angmom0, virial0

def check_bounded_trajectory(trajectory, times, masses):
    # walks the fine grid trajectory and checks the three rejection
    # conditions: ejection, collision, energy drift
    energy0 = total_energy(trajectory[0], masses, EPSILON)
    max_drift = 0.0
    min_dist = None
    t_min_dist = None

    for k in range(len(times)):
        row = trajectory[k]
        e = total_energy(row, masses, EPSILON)
        drift = abs(e - energy0) / abs(energy0)
        if drift > max_drift:
            max_drift = drift

        pos = row[0:6].reshape(3, 2)
        for i in range(3):
            for j in range(i + 1, 3):
                diff = pos[j] - pos[i]
                dist = np.sqrt(diff[0] ** 2 + diff[1] ** 2)
                if min_dist is None or dist < min_dist:
                    min_dist = dist
                    t_min_dist = times[k]
                if dist > ESCAPE_RADIUS:
                    return False, "ejection", max_drift, min_dist, t_min_dist
                if dist < COLLISION_RADIUS:
                    return False, "collision", max_drift, min_dist, t_min_dist

    if max_drift > ENERGY_DRIFT_THRESHOLD:
        return False, "energy", max_drift, min_dist, t_min_dist

    return True, "accepted", max_drift, min_dist, t_min_dist

def evaluate_initial_condition(state0, masses, T, obs_times, t0):
    # the observation window starts at offset t0 and lasts T, so the trajectory
    # is integrated over [0, t0 + T]. theta stays the state at t = 0. obs_times
    # here are relative to the window start, the pattern from lyapunov.txt over
    # (0, T]; the absolute observation times are t0 + obs_times.
    # integrates on a fine grid over [0, t0 + T] to check acceptance, then, only
    # if accepted, integrates again at exactly the K absolute observation times.
    # returns (result_dict, reason); result_dict is None unless accepted
    start_time = time.time()
    span = t0 + T

    # scale the number of fine-grid points with the span so the time resolution
    # of the bounded-trajectory check matches the original [0, T] check
    fine_n = int(round(FINE_N * span / T))
    if fine_n < FINE_N:
        fine_n = FINE_N
    fine_times = np.linspace(0.0, span, fine_n)
    trajectory_fine, success = integrate_system(state0, fine_times, masses, EPSILON)

    if not success or np.any(np.isnan(trajectory_fine)):
        return None, "integrator"

    accepted, reason, max_drift, min_dist, t_min_dist = check_bounded_trajectory(
        trajectory_fine, fine_times, masses
    )
    if not accepted:
        return None, reason

    abs_obs_times = t0 + obs_times
    trajectory_obs, success_obs = integrate_system(state0, abs_obs_times, masses, EPSILON)
    if not success_obs or np.any(np.isnan(trajectory_obs)):
        return None, "integrator"

    walltime = time.time() - start_time
    energy0, angmom0, virial0 = compute_at_t0(state0, masses)

    result = {
        "theta": np.concatenate([state0[0:4], state0[6:10]]),
        "t0": t0,
        "traj_clean": trajectory_obs,
        "energy_drift": max_drift,
        "energy0": energy0,
        "angmom0": angmom0,
        "virial0": virial0,
        "min_dist": min_dist,
        "t_min_dist": t_min_dist,
        "walltime": walltime,
    }
    return result, "accepted"

def read_lyapunov_file(path):
    # parses results/lyapunov.txt, never retype lambda, T or obs_times by hand
    values = {}
    f = open(path, "r")
    for line in f:
        line = line.strip()
        if line == "":
            continue
        key, value = line.split(" ", 1)
        values[key] = value
    f.close()

    lambda_median = float(values["lambda_median"])
    T = float(values["T"])
    obs_times = np.array([float(x) for x in values["obs_times"].split(",")])
    return lambda_median, T, obs_times

def run_prior_check():
    masses = get_masses()
    lyapunov_path = os.path.join("results", "lyapunov.txt")
    lambda_median, T, obs_times = read_lyapunov_file(lyapunov_path)
    t0_max = 6.0 / lambda_median
    print("read T =", T, "lambda_median =", lambda_median, "t0_max =", t0_max)
    print("read obs_times =", obs_times)

    n_check = 50
    base_seed = 123
    rejection_counts = {"ejection": 0, "collision": 0, "integrator": 0, "energy": 0}
    n_accepted = 0

    for i in range(n_check):
        rng = np.random.default_rng(base_seed + i)
        state0 = sample_initial_condition(rng, masses)
        t0 = rng.uniform(0.0, t0_max)
        result, reason = evaluate_initial_condition(state0, masses, T, obs_times, t0)
        if reason == "accepted":
            n_accepted = n_accepted + 1
        else:
            rejection_counts[reason] = rejection_counts[reason] + 1

    print("prior check: drew", n_check, "candidates")
    print("accepted:", n_accepted, "rejection rate:", (n_check - n_accepted) / n_check)
    print("rejection reasons:", rejection_counts)

if __name__ == "__main__":
    run_prior_check()
