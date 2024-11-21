import numpy as np
import matplotlib.pyplot as plt
from time import time

# Parameters
Re = 100.0
Lx, Ly = 1.0, 1.0
nx, ny = 100, 100
dx, dy = Lx / (nx - 1), Ly / (ny - 1)

# Grid
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
X, Y = np.meshgrid(x, y)

# Function to compute the exact solution
def exact_solution(xg, yg, t):
    u = 3/4 - 1./(4 * (1 + np.exp((-4 * xg + 4 * yg - t) * Re / 32)))
    v = 3/4 + 1./(4 * (1 + np.exp((-4 * xg + 4 * yg - t) * Re / 32)))
    return u, v

CFL = 0.01
u, v = exact_solution(X, Y, 0)
speed = np.sqrt(u**2 + v**2)
print("This is the mean speed:" + str(np.mean(speed)))
max_speed = np.max(speed)
print(max_speed)
dt = CFL * ((dx**2 + dy**2)**(0.5)) / max_speed
T_final = 0.35

# Initialize u and v arrays using the exact solution
u_exact, v_exact = exact_solution(X, Y, 0)
u_numerical = u_exact.copy()
v_numerical = v_exact.copy()

# Arrays for storing u, v values at a specific point over time
u_time_series = []
v_time_series = []

# Temporary arrays to hold new time step calculations
u_next = np.zeros_like(u_numerical)
v_next = np.zeros_like(v_numerical)

# Arrays to store derivatives from the previous time step
prev_dudt = np.zeros_like(u_numerical)
prev_dvdt = np.zeros_like(v_numerical)

# Time integration loop for numerical solution
t0 = time()
n = 0
for t in np.arange(dt, T_final + dt, dt):
    n += 1
    # Compute gradients
    dudx, dudy = np.gradient(u_numerical, dx, dy)
    dvdx, dvdy = np.gradient(v_numerical, dx, dy)
    d2udx2, d2udy2 = np.gradient(dudx, dx, dy)
    d2vdx2, d2vdy2 = np.gradient(dvdx, dx, dy)

    # Update derivatives
    dudt = -(u_numerical * dudx + v_numerical * dudy) + (d2udx2 + d2udy2) / Re
    dvdt = -(u_numerical * dvdx + v_numerical * dvdy) + (d2vdx2 + d2vdy2) / Re


    if t == dt:
        u_next = u_numerical + dt * dudt
        v_next = v_numerical + dt * dvdt
    else:  
        u_next = u_numerical + dt * 0.5 * (3 * dudt - prev_dudt)
        v_next = v_numerical + dt * 0.5 * (3 * dvdt - prev_dvdt)


    # Apply boundary conditions
    u_exact, v_exact = exact_solution(X, Y, t)
    u_next[0, :], u_next[:, 0] = u_exact[0, :], u_exact[:, 0]
    v_next[0, :], v_next[:, 0] = v_exact[0, :], v_exact[:, 0]

    # Update previous time step derivatives
    prev_dudt, prev_dvdt = dudt, dvdt

    # Swap arrays for the next step
    u_numerical, u_next = u_next, u_numerical
    v_numerical, v_next = v_next, v_numerical

    # Record time series or compute visualization data as needed

    


# Select the middle index in the y-direction
mid_y_idx = ny // 2

# Extract u and v along x at the middle y-coordinate
u_mid_y = u_numerical[mid_y_idx, :]
v_mid_y = v_numerical[mid_y_idx, :]

# Plotting
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Plot numerical u along x at mid y
axs[0].plot(x, u_mid_y, label='Numerical u at mid y')
axs[0].set_title('Numerical u vs. x at Mid y')
axs[0].set_xlabel('x')
axs[0].set_ylabel('u')
axs[0].legend()

# Plot numerical v along x at mid y
axs[1].plot(x, v_mid_y, label='Numerical v at mid y')
axs[1].set_title('Numerical v vs. x at Mid y')
axs[1].set_xlabel('x')
axs[1].set_ylabel('v')
axs[1].legend()

plt.show()

# Time taken
print(f"Time taken: {time() - t0:.2f} seconds")
print(f"Number of iterations:{n}")

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
