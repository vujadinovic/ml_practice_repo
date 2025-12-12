import torch
import torch.nn as nn
from torchdiffeq import odeint
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


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

t = torch.tensor(data["t"], dtype=torch.float32)            # (T,)
y = torch.tensor(data["y"], dtype=torch.float32)            # (N_TRAJ, T, 4)
dy = torch.tensor(data["dy"], dtype=torch.float32)          # derivatives (optional)

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

        # Initialize small weights for stability
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, x):
        return self.net(x)

odefunc = ODEFunc(D)

# ----------------------------
# Training setup
# ----------------------------
optimizer = torch.optim.Adam(odefunc.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()
n_epochs = 300

# Use first 30 trajectories for quick demo
train_y = y[:30]
train_N = train_y.shape[0]

# ----------------------------
# Training loop
# ----------------------------
for epoch in range(n_epochs):
    optimizer.zero_grad()
    loss = 0.0
    for i in range(train_N):
        y0 = train_y[i, 0]                   # initial state
        pred = odeint(odefunc, y0, t)        # (T, D)
        loss += loss_fn(pred, train_y[i])    # compare full trajectory
    loss /= train_N
    loss.backward()
    optimizer.step()
    
    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

# ----------------------------
# Quick visualization
# ----------------------------
y0 = train_y[0, 0]
pred = odeint(odefunc, y0, t).detach().numpy()
true = train_y[0].numpy()

plt.figure(figsize=(8,6))
for i in range(D):
    plt.plot(t, true[:, i], label=f"true {i}")
    plt.plot(t, pred[:, i], '--', label=f"pred {i}")
plt.xlabel("Time [s]")
plt.ylabel("State")
plt.legend()
plt.show()

# This should visualize it
x2_pred = L1*np.sin(pred[:,0]) + L2*np.sin(pred[:,2])
y2_pred = -L1*np.cos(pred[:,0]) - L2*np.cos(pred[:,2])
x2_true = L1*np.sin(true[:,0]) + L2*np.sin(true[:,2])
y2_true = -L1*np.cos(true[:,0]) - L2*np.cos(true[:,2])

plt.figure()
plt.plot(x2_true, y2_true, label='True bob 2')
plt.plot(x2_pred, y2_pred, '--', label='Pred bob 2')
plt.legend()
plt.axis('equal')
plt.show()



# True pendulum coordinates
x1_true = L1 * np.sin(true[:, 0])
y1_true = -L1 * np.cos(true[:, 0])
x2_true = x1_true + L2 * np.sin(true[:, 2])
y2_true = y1_true - L2 * np.cos(true[:, 2])

# Predicted pendulum coordinates
x1_pred = L1 * np.sin(pred[:, 0])
y1_pred = -L1 * np.cos(pred[:, 0])
x2_pred = x1_pred + L2 * np.sin(pred[:, 2])
y2_pred = y1_pred - L2 * np.cos(pred[:, 2])

# ----------------------------
# Animation
# ----------------------------

fig = plt.figure(figsize=(5,4))
ax = fig.add_subplot(autoscale_on=False, xlim=(-L, L), ylim=(-L, 1))
ax.set_aspect('equal')
ax.grid()

# True trajectory line (blue)
line_true, = ax.plot([], [], 'o-', lw=2, label="True")

# Predicted trajectory line (red)
line_pred, = ax.plot([], [], 'o-', lw=2, color='red', alpha=0.7, label="Predicted")

time_template = 'time = %.2fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
ax.legend()


def animate(i):
    # TRUE
    thisx_true = [0, x1_true[i], x2_true[i]]
    thisy_true = [0, y1_true[i], y2_true[i]]
    line_true.set_data(thisx_true, thisy_true)

    # PRED
    thisx_pred = [0, x1_pred[i], x2_pred[i]]
    thisy_pred = [0, y1_pred[i], y2_pred[i]]
    line_pred.set_data(thisx_pred, thisy_pred)

    time_text.set_text(time_template % (i * float(t[1]-t[0])))

    return line_true, line_pred, time_text


ani = animation.FuncAnimation(
    fig, animate, frames=T, interval=30, blit=True
)

plt.show()