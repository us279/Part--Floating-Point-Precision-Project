import numpy as np
import matplotlib.pyplot as plt
from time import time

# Parameters
Re = 100.0
Lx, Ly = 1.0, 1.0
nx, ny = 100, 100
dx, dy = Lx / (nx - 1), Ly / (ny - 1)
dt = 0.001
T_final = 0.1

# Grid
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
X, Y = np.meshgrid(x, y)

# Initialize u and v arrays
u_exact = np.zeros((nx, ny))
v_exact = np.zeros((nx, ny))
u_numerical = np.zeros((nx, ny))
v_numerical = np.zeros((nx, ny))

# Exact solution function
def exact_solution(xg, yg, t):
    u_exact = 3/4 - 1./(4 * (1 + np.exp((-4 * xg + 4 * yg - t) * Re / 32)))
    v_exact = 3/4 + 1./(4 * (1 + np.exp((-4 * xg + 4 * yg - t) * Re / 32)))
    return u_exact, v_exact

# Initial condition (using exact solution for initialization)
u_exact, v_exact = exact_solution(X, Y, 0)
u_numerical = u_exact.copy()
v_numerical = v_exact.copy()

# Time integration loop
t0 = time()
for t in np.arange(dt, T_final + dt, dt):
    # Adams-Bashforth method for u and v
    if t == dt:
        # First-order Euler for initialization
        dudx, dudy = np.gradient(u_exact, dx, dy)
        dvdx, dvdy = np.gradient(v_exact, dx, dy)
        d2udx2, _ = np.gradient(dudx, dx)
        _, d2udy2 = np.gradient(dudy, dy)
        d2vdx2, _ = np.gradient(dvdx, dx)
        _, d2vdy2 = np.gradient(dvdy, dy)

        u_n = u_exact + dt * (-(u_exact * dudx + v_exact * dudy) +
                              (d2udx2 + d2udy2) / Re)
        v_n = v_exact + dt * (-(u_exact * dvdx + v_exact * dvdy) +
                              (d2vdx2 + d2vdy2) / Re)
    else:
        dudx, dudy = np.gradient(u_n, dx, dy)
        dvdx, dvdy = np.gradient(v_n, dx, dy)
        d2udx2, _ = np.gradient(dudx, dx)
        _, d2udy2 = np.gradient(dudy, dy)
        d2vdx2, _ = np.gradient(dvdx, dx)
        _, d2vdy2 = np.gradient(dvdy, dy)

        u_n = u_n + dt * (-(u_n * dudx + v_n * dudy) + (d2udx2 + d2udy2) / Re)
        v_n = v_n + dt * (-(u_n * dvdx + v_n * dvdy) + (d2vdx2 + d2vdy2) / Re)

    # Exact solution for comparison (at time t)
    u_numerical , v_numerical= u_n,v_n

    u_exact, v_exact = exact_solution(X, Y, t)

# Time taken
print(f"Time taken: {time() - t0:.2f} seconds")

# Plotting
fig, axs = plt.subplots(2, 3, figsize=(18, 12))

# Plot exact u
cf1 = axs[0, 0].contourf(X, Y, u_exact, cmap='viridis')
plt.colorbar(cf1, ax=axs[0, 0], orientation='vertical', label='u')
axs[0, 0].set_title('Exact Solution for u')
axs[0, 0].set_xlabel('x')
axs[0, 0].set_ylabel('y')

# Plot exact v
cf2 = axs[0, 1].contourf(X, Y, v_exact, cmap='viridis')
plt.colorbar(cf2, ax=axs[0, 1], orientation='vertical', label='v')
axs[0, 1].set_title('Exact Solution for v')
axs[0, 1].set_xlabel('x')
axs[0, 1].set_ylabel('y')

# Plot numerical u
cf3 = axs[1, 0].contourf(X, Y, u_numerical, cmap='viridis')
plt.colorbar(cf3, ax=axs[1, 0], orientation='vertical', label='u')
axs[1, 0].set_title('Numerical Solution for u')
axs[1, 0].set_xlabel('x')
axs[1, 0].set_ylabel('y')

# Plot numerical v
cf4 = axs[1, 1].contourf(X, Y, v_numerical, cmap='viridis')
plt.colorbar(cf4, ax=axs[1, 1], orientation='vertical', label='v')
axs[1, 1].set_title('Numerical Solution for v')
axs[1, 1].set_xlabel('x')
axs[1, 1].set_ylabel('y')

# Plot difference between exact and numerical u
diff_u = np.abs(u_exact - u_numerical)
cf5 = axs[1, 2].contourf(X, Y, diff_u, cmap='magma')
plt.colorbar(cf5, ax=axs[1, 2], orientation='vertical', label='|u_exact - u_numerical|')
axs[1, 2].set_title('Absolute Difference (u)')
axs[1, 2].set_xlabel('x')
axs[1, 2].set_ylabel('y')

# Plot difference between exact and numerical v
diff_v = np.abs(v_exact - v_numerical)
cf6 = axs[0, 2].contourf(X, Y, diff_v, cmap='magma')
plt.colorbar(cf6, ax=axs[0, 2], orientation='vertical', label='|v_exact - v_numerical|')
axs[0, 2].set_title('Absolute Difference (v)')
axs[0, 2].set_xlabel('x')
axs[0, 2].set_ylabel('y')

plt.tight_layout()
plt.show()
