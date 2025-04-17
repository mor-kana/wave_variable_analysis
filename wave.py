import numpy as np
import matplotlib.pyplot as plt

# === Parameters ===
dt = 0.01
steps = 1000
b_v = 50.0  # translational impedance
b_w = 10.0  # rotational impedance
B = np.diag([b_v, b_w])
B_sqrt_inv = np.diag([1/np.sqrt(2*b_v), 1/np.sqrt(2*b_w)])
B_inv = np.linalg.inv(B)

# === Initial States ===
p_slave = np.zeros(3)  # [x, y, theta]
trajectory = []

# === Main Simulation Loop ===
for _ in range(steps):
    # Master sends velocity
    delta_m = np.array([0.5, 0.1])  # e.g., constant forward and slight rotation
    tau_m = B @ delta_m  # virtual force

    # Master constructs wave variable
    u = B_sqrt_inv @ (tau_m + B @ delta_m)

    # Slave receives u (simulate v=0 for one-way)
    delta_m_recv = B_inv @ u / np.sqrt(2)

    # Update slave position
    v, omega = delta_m_recv
    theta = p_slave[2]
    dx = v * np.cos(theta) * dt
    dy = v * np.sin(theta) * dt
    dtheta = omega * dt
    p_slave += np.array([dx, dy, dtheta])
    trajectory.append(p_slave.copy())

# === Plot trajectory ===
trajectory = np.array(trajectory)
plt.plot(trajectory[:,0], trajectory[:,1])
plt.title("Slave Robot Trajectory")
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.grid()
plt.show()