import numpy as np
import matplotlib.pyplot as plt
from time import time

# Parameters
Re = 100.0
Lx, Ly = 1.0, 1.0
nx, ny = 100, 100
dx, dy = Lx / (nx - 1), Ly / (ny - 1)
dt = 0.001
T_final = 1

# Grid
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
X, Y = np.meshgrid(x, y)

# Function to compute the exact solution
def exact_solution(xg, yg, t):
    u = 3/4 - 1./(4 * (1 + np.exp((-4 * xg + 4 * yg - t) * Re / 32)))
    v = 3/4 + 1./(4 * (1 + np.exp((-4 * xg + 4 * yg - t) * Re / 32)))
    return u, v

# Initialize u and v arrays using the exact solution
u_numerical, v_numerical = exact_solution(X, Y, 0)

# Temporary arrays to hold new time step calculations
u_next = np.zeros_like(u_numerical)
v_next = np.zeros_like(v_numerical)

# Arrays to store derivatives from the previous time step
prev_dudt = np.zeros_like(u_numerical)
prev_dvdt = np.zeros_like(v_numerical)

# Time integration loop for numerical solution
t0 = time()

for t in np.arange(dt, T_final + dt, dt):
    # Compute gradients
    dudx, dudy = np.gradient(u_numerical, dx, dy)
    dvdx, dvdy = np.gradient(v_numerical, dx, dy)
    d2udx2, _ = np.gradient(dudx, dx)
    _, d2udy2 = np.gradient(dudy, dy)
    d2vdx2, _ = np.gradient(dvdx, dx)
    _, d2vdy2 = np.gradient(dvdy, dy)

    # Update derivatives
    dudt = -(u_numerical * dudx + v_numerical * dudy) + (d2udx2 + d2udy2) / Re
    dvdt = -(u_numerical * dvdx + v_numerical * dvdy) + (d2vdx2 + d2vdy2) / Re

    # Adams-Bashforth 2nd Order for t > dt
    if t > dt:
        u_next = u_numerical + dt * (1.5 * dudt - 0.5 * prev_dudt)
        v_next = v_numerical + dt * (1.5 * dvdt - 0.5 * prev_dvdt)
    else:  # Euler method for the first step
        u_next = u_numerical + dt * dudt
        v_next = v_numerical + dt * dvdt

    # Update previous time step derivatives
    prev_dudt, prev_dvdt = dudt, dvdt

    # Swap arrays for the next step
    u_numerical, u_next = u_next, u_numerical
    v_numerical, v_next = v_next, v_numerical

    # Optional: Compute exact solution for visualization or error analysis
    if t == T_final:
        u_exact, v_exact = exact_solution(X, Y, t)

print(f"Simulation finished in {time() - t0:.2f}s")

# You can add plotting here to visualize the results
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.contourf(X, Y, u_numerical, levels=50, cmap='viridis')
plt.colorbar()
plt.title('Numerical u')

plt.subplot(1, 2, 2)
plt.contourf(X, Y, u_exact, levels=50, cmap='viridis')
plt.colorbar()
plt.title('Exact u')
plt.show()
