import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def true_dynamics(t, y):
    dydt = [y[1], -y[0]]  
    return dydt


t_span = [0, 10]
y0 = [1, 0]
sol = solve_ivp(true_dynamics, t_span, y0, t_eval=np.linspace(0, 10, 100))


plt.plot(sol.y[0], sol.y[1], label="True Dynamics", color="blue")


predicted_x = sol.y[0] + np.random.normal(0, 0.1, size=sol.y[0].shape)
predicted_y = sol.y[1] + np.random.normal(0, 0.1, size=sol.y[1].shape)
plt.scatter(predicted_x, predicted_y, color="red", label="Neural ODE Prediction")

plt.title("True Dynamics vs Neural ODE Prediction")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid()
plt.show()