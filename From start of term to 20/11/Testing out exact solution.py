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

def exact_solution(xg, yg, t):
    u = 3/4 - 1./(4 * (1 + np.exp((-4 * xg + 4 * yg - t) * Re / 32)))
    v = 3/4 + 1./(4 * (1 + np.exp((-4 * xg + 4 * yg - t) * Re / 32)))
    return u, v

CFL = 0.01
u, v = exact_solution(X, Y, 0)
speed = np.sqrt(u**2 + v**2)
max_speed = np.max(speed)
dt = CFL * min(dx, dy) / max_speed
T_final = 1

u_numerical = np.zeros((ny, nx))
v_numerical = np.zeros((ny, nx))
u_numerical, v_numerical = exact_solution(X, Y, 0)

u_time_series = []
v_time_series = []
u_exact_time_series = []
v_exact_time_series = []
t_list = []

t0 = time()
n = 0
for t in np.arange(dt, T_final + dt, dt):
    t_list.append(t)
    n += 1
    dudx, dudy = np.gradient(u_numerical, dx, dy)
    dvdx, dvdy = np.gradient(v_numerical, dx, dy)
    d2udx2, d2udy2 = np.gradient(dudx, dx, dy)
    d2vdx2, d2vdy2 = np.gradient(dvdx, dx, dy)

    dudt = -(u_numerical * dudx + v_numerical * dudy) + (d2udx2 + d2udy2) / Re
    dvdt = -(u_numerical * dvdx + v_numerical * dvdy) + (d2vdx2 + d2vdy2) / Re

    u_next = u_numerical + dt * dudt
    v_next = v_numerical + dt * dvdt

    u_numerical, u_next = u_next, u_numerical
    v_numerical, v_next = v_next, v_numerical

    # Store u, v values at mid y for plotting against time
    u_time_series.append(u_numerical[:, ny//2])
    v_time_series.append(v_numerical[:, ny//2])
    
    # Compute and store exact solutions for comparison
    u_exact, v_exact = exact_solution(X, Y, t)
    u_exact_time_series.append(u_exact[:, ny//2])
    v_exact_time_series.append(v_exact[:, ny//2])

# Convert lists to arrays for easier handling in plotting
u_time_series = np.array(u_time_series).T
v_time_series = np.array(v_time_series).T
u_exact_time_series = np.array(u_exact_time_series).T
v_exact_time_series = np.array(v_exact_time_series).T
t_list = np.array(t_list)

# Plotting
fig, axs = plt.subplots(2, 2, figsize=(12, 12))

# Numerical u vs. time contour plot
contour_u_num = axs[0,0].contourf(t_list, x, u_time_series, 20, cmap='viridis')
axs[0,0].set_title('Numerical u vs. Time at Mid y')
axs[0,0].set_xlabel('Time')
axs[0,0].set_ylabel('x')
fig.colorbar(contour_u_num, ax=axs[0,0], orientation='vertical')

# Exact u vs. time contour plot
contour_u_exact = axs[0,1].contourf(t_list, x, u_exact_time_series, 20, cmap='viridis')
axs[0,1].set_title('Exact u vs. Time at Mid y')
axs[0,1].set_xlabel('Time')
axs[0,1].set_ylabel('x')
fig.colorbar(contour_u_exact, ax=axs[0,1], orientation='vertical')

# Numerical v vs. time contour plot
contour_v_num = axs[1,0].contourf(t_list, x, v_time_series, 20, cmap='viridis')
axs[1,0].set_title('Numerical v vs. Time at Mid y')
axs[1,0].set_xlabel('Time')
axs[1,0].set_ylabel('x')
fig.colorbar(contour_v_num, ax=axs[1,0], orientation='vertical')

# Exact v vs. time contour plot
contour_v_exact = axs[1,1].contourf(t_list, x, v_exact_time_series, 20, cmap='viridis')
axs[1,1].set_title('Exact v vs. Time at Mid y')
axs[1,1].set_xlabel('Time')
axs[1,1].set_ylabel('x')
fig.colorbar(contour_v_exact, ax=axs[1,1], orientation='vertical')

plt.tight_layout()
plt.show()

print(f"Time taken: {time() - t0:.2f} seconds")
print(f"Number of iterations: {n}")
