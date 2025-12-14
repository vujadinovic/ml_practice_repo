import torch
import torch.nn as nn
from torchdiffeq import odeint
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ----------------------------
# Device
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

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
# Load dataset
# ----------------------------
data = np.load("/home/blanche/Documents/ML/double_pendulum_multi_trajectory.npz")

t = torch.tensor(data["t"], dtype=torch.float32).to(device)          # (T,)
y = torch.tensor(data["y"], dtype=torch.float32).to(device)          # (N, T, 4)

N_TRAJ, T, D = y.shape

# ----------------------------
# Neural ODE model
# ----------------------------
class ODEFunc(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, dim)
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, t, x):
        return self.net(x)

odefunc = ODEFunc(D).to(device)

# ----------------------------
# Training setup
# ----------------------------
optimizer = torch.optim.Adam(odefunc.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()
n_epochs = 300

# Use first 30 trajectories (batch training)
train_y = y[:30]                     # (B, T, D)
B = train_y.shape[0]

# ----------------------------
# Normalize data
# ----------------------------
# Compute mean and std across batch and time
y_mean = train_y.mean(dim=(0, 1), keepdim=True)   # shape: (1, 1, D)
y_std = train_y.std(dim=(0, 1), keepdim=True)     # shape: (1, 1, D)

# Avoid division by zero
y_std[y_std == 0] = 1.0

# Normalize training data
train_y_norm = (train_y - y_mean) / y_std       # (B, T, D)

# ----------------------------
# Training loop with normalized data
# ----------------------------
for epoch in range(n_epochs):
    optimizer.zero_grad()

    y0_norm = train_y_norm[:, 0]                  # (B, D)

    # odeint output: (T, B, D)
    pred_norm = odeint(odefunc, y0_norm, t, method="rk4")

    # match shape (B, T, D)
    pred_norm = pred_norm.permute(1, 0, 2)

    loss = loss_fn(pred_norm, train_y_norm)
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(f"Epoch {epoch:03d} | Loss {loss.item():.6f}")

# ----------------------------
# Single-trajectory visualization
# ----------------------------
y0_norm = train_y_norm[0, 0]
pred_norm = odeint(odefunc, y0_norm, t, method="rk4").permute(1,0,2).detach()
pred = pred_norm * y_std + y_mean               # unnormalize
pred = pred.cpu().numpy()
true = train_y[0].cpu().numpy()
t_cpu = t.cpu().numpy()


plt.figure(figsize=(8,6))
for i in range(D):
    plt.plot(t_cpu, true[:, i], label=f"true {i}")
    plt.plot(t_cpu, pred[:, i], "--", label=f"pred {i}")
plt.xlabel("Time [s]")
plt.ylabel("State")
plt.legend()
plt.show()

# ----------------------------
# Cartesian coordinates
# ----------------------------
x1_true = L1 * np.sin(true[:, 0])
y1_true = -L1 * np.cos(true[:, 0])
x2_true = x1_true + L2 * np.sin(true[:, 2])
y2_true = y1_true - L2 * np.cos(true[:, 2])

x1_pred = L1 * np.sin(pred[:, 0])
y1_pred = -L1 * np.cos(pred[:, 0])
x2_pred = x1_pred + L2 * np.sin(pred[:, 2])
y2_pred = y1_pred - L2 * np.cos(pred[:, 2])

plt.figure()
plt.plot(x2_true, y2_true, label="True bob 2")
plt.plot(x2_pred, y2_pred, "--", label="Pred bob 2")
plt.axis("equal")
plt.legend()
plt.show()

# ----------------------------
# Animation
# ----------------------------
fig = plt.figure(figsize=(5,4))
ax = fig.add_subplot(autoscale_on=False, xlim=(-L, L), ylim=(-L, 1))
ax.set_aspect("equal")
ax.grid()

line_true, = ax.plot([], [], "o-", lw=2, label="True")
line_pred, = ax.plot([], [], "o-", lw=2, color="red", alpha=0.7, label="Pred")

time_text = ax.text(0.05, 0.9, "", transform=ax.transAxes)
ax.legend()

def animate(i):
    line_true.set_data([0, x1_true[i], x2_true[i]],
                       [0, y1_true[i], y2_true[i]])

    line_pred.set_data([0, x1_pred[i], x2_pred[i]],
                       [0, y1_pred[i], y2_pred[i]])

    time_text.set_text(f"time = {t_cpu[i]:.2f}s")
    return line_true, line_pred, time_text

ani = animation.FuncAnimation(
    fig, animate, frames=T, interval=30, blit=True
)

plt.show()
