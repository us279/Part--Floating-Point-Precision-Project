import numpy as np
import matplotlib.pyplot as plt
#from time import time

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

def testing_fp_precision(precision):
    from time import time
    def exact_solution(xg, yg, t):
        u = np.float64(3/4 - 1./(4 * (1 + np.exp((-4 * xg + 4 * yg - t) * Re / 32))))
        v = np.float64(3/4 + 1./(4 * (1 + np.exp((-4 * xg + 4 * yg - t) * Re / 32))))
        return u, v

    # Initialize u and v arrays using the exact solution
    u_numerical, v_numerical = precision(exact_solution(X, Y, 0))

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
        dudx, dudy = precision(np.gradient(u_numerical, dx, dy))
        dvdx, dvdy = precision(np.gradient(v_numerical, dx, dy))
        d2udx2, _ = precision(np.gradient(dudx, dx))
        _, d2udy2 = precision(np.gradient(dudy, dy))
        d2vdx2, _ = precision(np.gradient(dvdx, dx))
        _, d2vdy2 = precision(np.gradient(dvdy, dy))
        # Update derivatives
        dudt = precision(-(u_numerical * dudx + v_numerical * dudy) + (d2udx2 + d2udy2) / Re)
        dvdt = precision(-(u_numerical * dvdx + v_numerical * dvdy) + (d2vdx2 + d2vdy2) / Re)

        # Adams-Bashforth 2nd Order for t > dt
        if t > dt:
            u_next = precision(u_numerical + dt * (1.5 * dudt - 0.5 * prev_dudt))
            v_next = precision(v_numerical + dt * (1.5 * dvdt - 0.5 * prev_dvdt))
        else:  # Euler method for the first step
            u_next = precision(u_numerical + dt * dudt)
            v_next = precision(v_numerical + dt * dvdt)

        # Update previous time step derivatives
        prev_dudt, prev_dvdt = dudt, dvdt

        # Swap arrays for the next step
        u_numerical, u_next = u_next, u_numerical
        v_numerical, v_next = v_next, v_numerical

        # Optional: Compute exact solution for visualization or error analysis
        if t == T_final:
            u_exact, v_exact = exact_solution(X, Y, t)

    # Time taken
    print(f"Time taken: {time() - t0:.2f} seconds")

    def compute_error(u_exact, v_exact, u_numerical, v_numerical):
        error_u = np.abs(u_exact - u_numerical)
        error_v = np.abs(v_exact - v_numerical)
        max_error_u = np.max(error_u)
        mean_error_u = np.mean(error_u)
        max_error_v = np.max(error_v)
        mean_error_v = np.mean(error_v)
        return (max_error_u, mean_error_u, max_error_v, mean_error_v)
    max_error_u, mean_error_u, max_error_v, mean_error_v = compute_error(u_exact, v_exact, u_numerical, v_numerical)
    print("For precision " + str(precision) + "max error in u is = " + str(max_error_u))
    print("For precision " + str(precision) + "mean error in u is = " + str(mean_error_u))
    print("For precision " + str(precision) + "max error in v is = " + str(max_error_v))
    print("For precision " + str(precision) + "mean error in v is = " + str(mean_error_v))


    # Plotting
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))

    # Plot exact u
    cf1 = axs[0, 0].contourf(X, Y, u_exact,levels= 100, cmap='viridis')
    plt.colorbar(cf1, ax=axs[0, 0], orientation='vertical', label='u')
    axs[0, 0].set_title('Exact Solution for u')
    axs[0, 0].set_xlabel('x')
    axs[0, 0].set_ylabel('y')

    # Plot exact v
    cf2 = axs[0, 1].contourf(X, Y, v_exact,levels= 100, cmap='viridis')
    plt.colorbar(cf2, ax=axs[0, 1], orientation='vertical', label='v')
    axs[0, 1].set_title('Exact Solution for v')
    axs[0, 1].set_xlabel('x')
    axs[0, 1].set_ylabel('y')

    # Plot numerical u
    cf3 = axs[1, 0].contourf(X, Y, u_numerical,levels= 100, cmap='viridis')
    plt.colorbar(cf3, ax=axs[1, 0], orientation='vertical', label='u')
    axs[1, 0].set_title('Numerical Solution for u')
    axs[1, 0].set_xlabel('x')
    axs[1, 0].set_ylabel('y')

    # Plot numerical v
    cf4 = axs[1, 1].contourf(X, Y, v_numerical,levels= 100, cmap='viridis')
    plt.colorbar(cf4, ax=axs[1, 1], orientation='vertical', label='v')
    axs[1, 1].set_title('Numerical Solution for v')
    axs[1, 1].set_xlabel('x')
    axs[1, 1].set_ylabel('y')

    # Plot difference between exact and numerical u
    diff_u = np.abs(u_exact - u_numerical)
    cf5 = axs[1, 2].contourf(X, Y, diff_u,levels= 100, cmap='magma')
    plt.colorbar(cf5, ax=axs[1, 2], orientation='vertical', label='|u_exact - u_numerical|')
    axs[1, 2].set_title('Absolute Difference (u)')
    axs[1, 2].set_xlabel('x')
    axs[1, 2].set_ylabel('y')

    # Plot difference between exact and numerical v
    diff_v = np.abs(v_exact - v_numerical)
    cf6 = axs[0, 2].contourf(X, Y, diff_v,levels= 100, cmap='magma')
    plt.colorbar(cf6, ax=axs[0, 2], orientation='vertical', label='|v_exact - v_numerical|')
    axs[0, 2].set_title('Absolute Difference (v)')
    axs[0, 2].set_xlabel('x')
    axs[0, 2].set_ylabel('y')

    plt.tight_layout()
    plt.show()

for fp in [np.float64, np.float32, np.float16]:
    testing_fp_precision(fp)
