import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# fixed masses, ratio 1:2:3, sum to 1 (nondimensional units, G = 1)
def get_masses():
    return np.array([1.0 / 6.0, 2.0 / 6.0, 3.0 / 6.0])

# softening length, stated openly wherever the physics is used
EPSILON = 0.02

def make_com_consistent_state(r1, v1, r2, v2, masses):
    # builds the full 12 number state from bodies 1 and 2, body 3 comes
    # from the center of mass constraints (zero total position and momentum)
    m1, m2, m3 = masses
    r1 = np.array(r1)
    v1 = np.array(v1)
    r2 = np.array(r2)
    v2 = np.array(v2)
    r3 = -(m1 * r1 + m2 * r2) / m3
    v3 = -(m1 * v1 + m2 * v2) / m3
    state = np.zeros(12)
    state[0:2] = r1
    state[2:4] = r2
    state[4:6] = r3
    state[6:8] = v1
    state[8:10] = v2
    state[10:12] = v3
    return state

def example_initial_conditions(masses):
    # a few hand picked initial conditions, only for this sanity check
    # the real prior sampling for the dataset lives in prior.py
    examples = []
    examples.append(make_com_consistent_state([1.0, 0.0], [0.0, 0.4], [-0.6, 0.8], [-0.3, -0.2], masses))
    examples.append(make_com_consistent_state([0.8, -0.3], [0.1, 0.6], [-0.9, 0.2], [-0.2, -0.5], masses))
    examples.append(make_com_consistent_state([0.5, 0.9], [-0.4, 0.1], [-1.0, -0.4], [0.3, -0.3], masses))
    return examples

def acceleration(pos, masses, epsilon):
    # pos has shape (3, 2), returns acceleration with shape (3, 2)
    n = len(masses)
    acc = np.zeros((n, 2))
    for i in range(n):
        for j in range(n):
            if i != j:
                diff = pos[j] - pos[i]
                dist2 = diff[0] ** 2 + diff[1] ** 2 + epsilon ** 2
                dist3 = dist2 ** 1.5
                acc[i] = acc[i] + masses[j] * diff / dist3
    return acc

def equations_of_motion(t, state, masses, epsilon):
    pos = state[0:6].reshape(3, 2)
    vel = state[6:12].reshape(3, 2)
    acc = acceleration(pos, masses, epsilon)
    derivative = np.zeros(12)
    derivative[0:6] = vel.reshape(6)
    derivative[6:12] = acc.reshape(6)
    return derivative

def total_energy(state, masses, epsilon):
    # kinetic plus softened potential energy, self consistent with acceleration()
    pos = state[0:6].reshape(3, 2)
    vel = state[6:12].reshape(3, 2)
    n = len(masses)
    kinetic = 0.0
    for i in range(n):
        speed2 = vel[i][0] ** 2 + vel[i][1] ** 2
        kinetic = kinetic + 0.5 * masses[i] * speed2
    potential = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            diff = pos[j] - pos[i]
            dist = (diff[0] ** 2 + diff[1] ** 2 + epsilon ** 2) ** 0.5
            potential = potential - masses[i] * masses[j] / dist
    return kinetic + potential

def integrate_system(state0, times, masses, epsilon):
    # integrates from t=0 to the last requested time, returns the state
    # at every requested time and whether the integrator succeeded
    sol = solve_ivp(
        equations_of_motion,
        (0.0, times[-1]),
        state0,
        t_eval=times,
        args=(masses, epsilon),
        method="DOP853",
        rtol=1e-10,
        atol=1e-12,
    )
    trajectory = sol.y.T
    return trajectory, sol.success

def run_sanity_check():
    masses = get_masses()
    t_end = 5.0
    n_times = 300
    times = np.linspace(0.0, t_end, n_times)
    examples = example_initial_conditions(masses)

    drift_lines = []
    plt.figure(figsize=(15, 5))

    for idx, state0 in enumerate(examples):
        trajectory, success = integrate_system(state0, times, masses, EPSILON)
        if not success:
            print("example", idx, "integrator failed")
            drift_lines.append("example_" + str(idx) + " integrator_failed")
            continue

        energy0 = total_energy(trajectory[0], masses, EPSILON)
        max_drift = 0.0
        for row in trajectory:
            e = total_energy(row, masses, EPSILON)
            drift = abs(e - energy0) / abs(energy0)
            if drift > max_drift:
                max_drift = drift
        accepted = max_drift < 1e-8
        print("example", idx, "max relative energy drift:", max_drift, "accepted:", accepted)
        drift_lines.append(
            "example_" + str(idx)
            + " max_relative_energy_drift " + str(max_drift)
            + " accepted " + str(accepted)
        )

        plt.subplot(1, len(examples), idx + 1)
        plt.plot(trajectory[:, 0], trajectory[:, 1], label="body1")
        plt.plot(trajectory[:, 2], trajectory[:, 3], label="body2")
        plt.plot(trajectory[:, 4], trajectory[:, 5], label="body3")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("example " + str(idx))
        plt.legend()
        plt.axis("equal")

    plt.tight_layout()
    plt.savefig("results/example_orbits.png")
    print("saved plot to results/example_orbits.png")

    f = open("results/energy_check.txt", "w")
    f.write("simulator sanity check, energy drift per example\n")
    f.write("epsilon " + str(EPSILON) + "\n")
    f.write("integrator DOP853 rtol 1e-10 atol 1e-12\n")
    for line in drift_lines:
        f.write(line + "\n")
    f.close()
    print("saved energy check to results/energy_check.txt")

if __name__ == "__main__":
    run_sanity_check()
