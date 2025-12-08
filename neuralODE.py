# synthetic data
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchdiffeq import odeint
import torch.nn.functional as F

def generateSpirals(t, noise = 0.01):
    x = np.sin(t) * np.exp(-0.1 * t)
    y = np.cos(t) * np.exp(-0.1 * t)
    x += np.random.normal(0, noise, size=t.shape)
    y += np.random.normal(0, noise, size=t.shape)
    return np.stack([x, y], axis=1)

t = np.linspace(0, 10, 100)
data = generateSpirals(t)


plt.figure(figsize=(6, 6))
plt.plot(data[:, 0], data[:, 1], label="Spiral Trajectory", color="blue")
plt.title("Synthetic Spiral Data")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid()
plt.show()



class ODEFunction(nn.Module):
    def __init__(self):
        super(ODEFunction, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2,50),
            nn.Tanh(),
            nn.Linear(50, 2)
        )

    def forward(self, t, y):
        return self.net(y)
    


ode_function = ODEFunction()
y0 = torch.tensor([1.0,0.0])
timeSteps = torch.linspace(0, 10, 100)

predicted_trajectory = odeint(ode_function, y0, timeSteps)

print(predicted_trajectory.shape)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(ode_function.parameters(), lr = 0.01)

data = torch.tensor(data, dtype=torch.float32)

for epoch in range(1000):
    optimizer.zero_grad()
    predicted_trajectory = odeint(ode_function, y0, timeSteps)
    loss = criterion(predicted_trajectory, data)
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")


rmse = torch.sqrt(F.mse_loss(predicted_trajectory, data))
print(f"RMSE: {rmse.item():.4f}")

# Plot the ground truth and predicted trajectories
plt.figure(figsize=(6, 6))
plt.plot(data[:, 0], data[:, 1], label="True Trajectory", color="blue")
plt.plot(predicted_trajectory[:, 0].detach().numpy(), predicted_trajectory[:, 1].detach().numpy(), 
         label="Predicted Trajectory", color="red", linestyle="--")
plt.title("Ground Truth vs Predicted Trajectory")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid()
plt.show()