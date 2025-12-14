import numpy as np
from numpy import sin, cos
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ----------------------------
# Physical constants
# ----------------------------
G = 9.8  
L1 = 1.0  
L2 = 1.0  
M1 = 1.0  
M2 = 1.0  
L = L1 + L2

# ----------------------------
# Double pendulum ODE
# ----------------------------
def derivs(t, state):
    dydx = np.zeros_like(state)

    dydx[0] = state[1]

    delta = state[2] - state[0]
    den1 = (M1 + M2) * L1 - M2 * L1 * cos(delta) ** 2
    dydx[1] = (
        M2 * L1 * state[1] ** 2 * sin(delta) * cos(delta)
        + M2 * G * sin(state[2]) * cos(delta)
        + M2 * L2 * state[3] ** 2 * sin(delta)
        - (M1 + M2) * G * sin(state[0])
    ) / den1

    dydx[2] = state[3]

    den2 = (L2 / L1) * den1
    dydx[3] = (
        -M2 * L2 * state[3] ** 2 * sin(delta) * cos(delta)
        + (M1 + M2) * G * sin(state[0]) * cos(delta)
        - (M1 + M2) * L1 * state[1] ** 2 * sin(delta)
        - (M1 + M2) * G * sin(state[2])
    ) / den2

    return dydx


# ----------------------------
# Simulation parameters
# ----------------------------
N_TRAJ = 1000            # number of trajectories
dt = 0.01              # time step
t_stop = 40           # 3× longer than your 2.5 seconds
t = np.arange(0, t_stop, dt)
T = len(t)

# Storage arrays
y_all = np.zeros((N_TRAJ, T, 4))
dy_all = np.zeros((N_TRAJ, T, 4))
x1_all = np.zeros((N_TRAJ, T))
y1_all = np.zeros((N_TRAJ, T))
x2_all = np.zeros((N_TRAJ, T))
y2_all = np.zeros((N_TRAJ, T))

# ----------------------------
# Generate multiple trajectories
# ----------------------------
# for k in range(N_TRAJ):
#     # Random initial conditions
#     th1 = np.random.uniform(-np.pi, np.pi)
#     th2 = np.random.uniform(-np.pi, np.pi)
#     w1 = np.random.uniform(-1.0, 1.0)
#     w2 = np.random.uniform(-1.0, 1.0)
#     state0 = np.array([th1, w1, th2, w2])

for k in range(N_TRAJ):

    if k < N_TRAJ // 3:
        # Low-energy regime (near-linear motion)
        th1 = np.random.uniform(-0.5, 0.5)
        th2 = np.random.uniform(-0.5, 0.5)
        w1 = np.random.uniform(-0.5, 0.5)
        w2 = np.random.uniform(-0.5, 0.5)

    elif k < 2 * N_TRAJ // 3:
        # Medium-energy regime
        th1 = np.random.uniform(-1.5, 1.5)
        th2 = np.random.uniform(-1.5, 1.5)
        w1 = np.random.uniform(-2.0, 2.0)
        w2 = np.random.uniform(-2.0, 2.0)

    else:
        # High-energy / chaotic regime
        th1 = np.random.uniform(-np.pi, np.pi)
        th2 = np.random.uniform(-np.pi, np.pi)
        w1 = np.random.uniform(-4.0, 4.0)
        w2 = np.random.uniform(-4.0, 4.0)

    state0 = np.array([th1, w1, th2, w2])


    sol = solve_ivp(
        derivs, [t[0], t[-1]], state0,
        t_eval=t, method="RK45", rtol=1e-10, atol=1e-10
    )

    y = sol.y.T
    y_all[k] = y

    # Compute derivatives for HNN learning
    dy_all[k] = np.array([derivs(t[i], y[i]) for i in range(T)])

    # Cartesian positions
    x1 = L1 * sin(y[:, 0])
    y1 = -L1 * cos(y[:, 0])
    x2 = x1 + L2 * sin(y[:, 2])
    y2 = y1 - L2 * cos(y[:, 2])

    x1_all[k] = x1
    y1_all[k] = y1
    x2_all[k] = x2
    y2_all[k] = y2

    print(f"Trajectory {k+1}/{N_TRAJ} done.")

# ----------------------------
# Save everything
# ----------------------------
np.savez(
    "double_pendulum_multi_trajectory.npz",
    t=t,
    y=y_all,
    dy=dy_all,
    x1=x1_all,
    y1=y1_all,
    x2=x2_all,
    y2=y2_all
)

print("Saved dataset to double_pendulum_multi_trajectory.npz")

plt.hist(y_all[:, 0, 0], bins=50)
plt.xlabel("Initial θ1")
plt.show()

plt.hist(y_all[:, 0, 1], bins=50)
plt.xlabel("Initial ω1")
plt.show()