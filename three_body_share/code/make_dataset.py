import os
import sys
import time
import multiprocessing
import numpy as np
from simulator import get_masses, EPSILON
from prior import (
    sample_initial_condition,
    evaluate_initial_condition,
    read_lyapunov_file,
    ESCAPE_RADIUS,
    COLLISION_RADIUS,
)

# default targets from the master plan, only used when no command line
# override is given
TARGET_TRAIN = 50000
TARGET_VALIDATION = 2000
TARGET_TEST = 2000

# number of candidate seeds attempted per wave. Each wave is one
# multiprocessing.Pool.map call and one checkpoint save, so this is both
# the parallel batch size and the chunk size for resuming.
CHUNK_SIZE = 500
SIGMA_X = 0.01
SIGMA_V = 0.01

# fixed, non overlapping seed ranges per split, so seeds never collide
# across splits no matter which split is generated in which run
BASE_SEED = 1000
SPLIT_SEED_OFFSET = {"train": 0, "validation": 10000000, "test": 20000000}

def add_noise(rng, traj_clean, sigma_x, sigma_v):
    noise = np.zeros(traj_clean.shape)
    for k in range(traj_clean.shape[0]):
        noise[k][0:6] = rng.normal(0.0, sigma_x, size=6)
        noise[k][6:12] = rng.normal(0.0, sigma_v, size=6)
    return traj_clean + noise

def evaluate_seed(args):
    # runs in a worker process: draws one candidate from its own seed,
    # evaluates it, and adds noise if accepted. Defined at module level
    # because Windows multiprocessing needs to pickle and re-import it.
    seed, T, obs_times, t0_max = args
    masses = get_masses()
    rng = np.random.default_rng(seed)
    state0 = sample_initial_condition(rng, masses)
    # window offset drawn from the same seeded rng, so the whole candidate is
    # reproducible from its seed
    t0 = rng.uniform(0.0, t0_max)
    result, reason = evaluate_initial_condition(state0, masses, T, obs_times, t0)
    if reason == "accepted":
        result["seed"] = seed
        noise_rng = np.random.default_rng(seed + 500000)
        result["obs_noisy"] = add_noise(noise_rng, result["traj_clean"], SIGMA_X, SIGMA_V)
    return reason, result

def generate_split(split_name, target_n, T, obs_times, t0_max, pool):
    # generates one split, in chunks, resumable from a partial file if the
    # run is interrupted before target_n accepted simulations are reached.
    # each chunk is one wave of CHUNK_SIZE candidate seeds run in parallel.
    base_seed = BASE_SEED + SPLIT_SEED_OFFSET[split_name]
    partial_path = "data/" + split_name + "_partial.npz"

    accepted = []
    rejection_counts = {"ejection": 0, "collision": 0, "integrator": 0, "energy": 0}
    seed_counter = 0
    n_chunks = 0

    if os.path.exists(partial_path):
        saved = np.load(partial_path, allow_pickle=True)
        seed_counter = int(saved["seed_counter"])
        rejection_counts = saved["rejection_counts"].item()
        accepted = list(saved["accepted"])
        n_chunks = int(saved["n_chunks"])
        print("resuming", split_name, "from", len(accepted), "accepted, seed_counter", seed_counter)

    start_time = time.time()
    while len(accepted) < target_n:
        seeds = []
        for i in range(CHUNK_SIZE):
            seeds.append(base_seed + seed_counter + i)
        seed_counter = seed_counter + CHUNK_SIZE
        n_chunks = n_chunks + 1

        tasks = [(seed, T, obs_times, t0_max) for seed in seeds]
        results = pool.map(evaluate_seed, tasks)

        for reason, result in results:
            if reason == "accepted":
                accepted.append(result)
            else:
                rejection_counts[reason] = rejection_counts[reason] + 1

        print(split_name, "chunk", n_chunks, "progress:", len(accepted), "/", target_n,
              "accepted, seed_counter", seed_counter, "rejections so far", rejection_counts)
        np.savez(
            partial_path,
            accepted=np.array(accepted, dtype=object),
            seed_counter=seed_counter,
            rejection_counts=rejection_counts,
            n_chunks=n_chunks,
        )

    # a wave can overshoot the target, keep exactly target_n, in seed order
    accepted = accepted[0:target_n]

    walltime_total = time.time() - start_time
    print(split_name, "generation finished in", walltime_total, "seconds,", n_chunks, "chunks")

    n = len(accepted)
    theta = np.zeros((n, 8))
    t0_arr = np.zeros(n)
    obs_noisy = np.zeros((n, len(obs_times), 12))
    traj_clean = np.zeros((n, len(obs_times), 12))
    seed_arr = np.zeros(n, dtype=int)
    energy_drift_arr = np.zeros(n)
    energy0_arr = np.zeros(n)
    angmom0_arr = np.zeros(n)
    virial0_arr = np.zeros(n)
    min_dist_arr = np.zeros(n)
    t_min_dist_arr = np.zeros(n)
    walltime_arr = np.zeros(n)

    for i in range(n):
        r = accepted[i]
        theta[i] = r["theta"]
        t0_arr[i] = r["t0"]
        obs_noisy[i] = r["obs_noisy"]
        traj_clean[i] = r["traj_clean"]
        seed_arr[i] = r["seed"]
        energy_drift_arr[i] = r["energy_drift"]
        energy0_arr[i] = r["energy0"]
        angmom0_arr[i] = r["angmom0"]
        virial0_arr[i] = r["virial0"]
        min_dist_arr[i] = r["min_dist"]
        t_min_dist_arr[i] = r["t_min_dist"]
        walltime_arr[i] = r["walltime"]

    np.savez(
        "data/" + split_name + ".npz",
        theta=theta,
        t0=t0_arr,
        obs_noisy=obs_noisy,
        traj_clean=traj_clean,
        obs_times=obs_times,
        seed=seed_arr,
        energy_drift=energy_drift_arr,
        energy0=energy0_arr,
        angmom0=angmom0_arr,
        virial0=virial0_arr,
        min_dist=min_dist_arr,
        t_min_dist=t_min_dist_arr,
        walltime=walltime_arr,
    )
    print("saved data/" + split_name + ".npz with", n, "accepted simulations")

    if os.path.exists(partial_path):
        os.remove(partial_path)

    return n, rejection_counts, n_chunks

def write_split_stats(split_name, n_accepted, rejection_counts, n_chunks):
    f = open("data/" + split_name + "_stats.txt", "w")
    f.write("n_accepted " + str(n_accepted) + "\n")
    f.write("rejected_ejection " + str(rejection_counts["ejection"]) + "\n")
    f.write("rejected_collision " + str(rejection_counts["collision"]) + "\n")
    f.write("rejected_integrator " + str(rejection_counts["integrator"]) + "\n")
    f.write("rejected_energy " + str(rejection_counts["energy"]) + "\n")
    f.write("n_chunks " + str(n_chunks) + "\n")
    f.close()

def read_split_stats(split_name):
    path = "data/" + split_name + "_stats.txt"
    empty_counts = {"ejection": 0, "collision": 0, "integrator": 0, "energy": 0}
    if not os.path.exists(path):
        return 0, empty_counts, 0

    values = {}
    f = open(path, "r")
    for line in f:
        line = line.strip()
        if line == "":
            continue
        key, value = line.split(" ")
        values[key] = int(value)
    f.close()

    rejection_counts = {
        "ejection": values["rejected_ejection"],
        "collision": values["rejected_collision"],
        "integrator": values["rejected_integrator"],
        "energy": values["rejected_energy"],
    }
    return values["n_accepted"], rejection_counts, values["n_chunks"]

def write_dataset_info(T, K, lambda_median, t0_max):
    n_train, rejected_train, chunks_train = read_split_stats("train")
    n_validation, rejected_validation, chunks_validation = read_split_stats("validation")
    n_test, rejected_test, chunks_test = read_split_stats("test")

    rejection_totals = {"ejection": 0, "collision": 0, "integrator": 0, "energy": 0}
    for counts in [rejected_train, rejected_validation, rejected_test]:
        for reason in rejection_totals:
            rejection_totals[reason] = rejection_totals[reason] + counts[reason]
    n_rejected_total = sum(rejection_totals.values())

    chunks_completed = (
        "train=" + str(chunks_train)
        + ",validation=" + str(chunks_validation)
        + ",test=" + str(chunks_test)
    )

    f = open("data/dataset_info.txt", "w")
    f.write("n_train " + str(n_train) + "\n")
    f.write("n_validation " + str(n_validation) + "\n")
    f.write("n_test " + str(n_test) + "\n")
    f.write("n_rejected_total " + str(n_rejected_total) + "\n")
    f.write("rejected_ejection " + str(rejection_totals["ejection"]) + "\n")
    f.write("rejected_collision " + str(rejection_totals["collision"]) + "\n")
    f.write("rejected_integrator " + str(rejection_totals["integrator"]) + "\n")
    f.write("rejected_energy " + str(rejection_totals["energy"]) + "\n")
    f.write("escape_radius " + str(ESCAPE_RADIUS) + "\n")
    f.write("collision_radius " + str(COLLISION_RADIUS) + "\n")
    f.write("epsilon " + str(EPSILON) + "\n")
    f.write("sigma_x " + str(SIGMA_X) + "\n")
    f.write("sigma_v " + str(SIGMA_V) + "\n")
    f.write("T " + str(T) + "\n")
    f.write("K " + str(K) + "\n")
    f.write("lambda_median " + str(lambda_median) + "\n")
    f.write("t0_max " + str(t0_max) + "\n")
    f.write("window_design offset_t0_uniform_0_to_6_over_lambda_then_span_T\n")
    f.write("rtol 1e-10\n")
    f.write("atol 1e-12\n")
    f.write("method DOP853\n")
    f.write("base_seed " + str(BASE_SEED) + "\n")
    f.write("chunks_completed " + chunks_completed + "\n")
    f.close()
    print("saved data/dataset_info.txt")

def main():
    lambda_median, T, obs_times = read_lyapunov_file("results/lyapunov.txt")
    K = len(obs_times)
    # window offset drawn uniformly from [0, 6/lambda] per simulation
    t0_max = 6.0 / lambda_median
    print("read T =", T, "K =", K, "lambda_median =", lambda_median, "t0_max =", t0_max)
    print("obs_times (relative to window start t0):", obs_times)

    splits_to_run = ["train", "validation", "test"]
    targets = {"train": TARGET_TRAIN, "validation": TARGET_VALIDATION, "test": TARGET_TEST}

    # optional override: python make_dataset.py <split_name> <target_n>
    # generates only that split with that target, used for small test runs
    if len(sys.argv) >= 3:
        splits_to_run = [sys.argv[1]]
        targets[sys.argv[1]] = int(sys.argv[2])

    n_processes = os.cpu_count()
    print("starting multiprocessing pool with", n_processes, "processes")

    with multiprocessing.Pool(processes=n_processes) as pool:
        for split_name in splits_to_run:
            n_accepted, rejection_counts, n_chunks = generate_split(
                split_name, targets[split_name], T, obs_times, t0_max, pool
            )
            write_split_stats(split_name, n_accepted, rejection_counts, n_chunks)
            print(split_name, "done:", n_accepted, "accepted, rejections:", rejection_counts)

    write_dataset_info(T, K, lambda_median, t0_max)

if __name__ == "__main__":
    main()
