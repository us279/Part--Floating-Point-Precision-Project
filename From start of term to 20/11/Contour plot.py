import numpy as np
import matplotlib.pyplot as plt

# Parameters
Re = 100.0
Lx, Ly = 1.0, 1.0
dt = 0.001
T_final = 0.1
nx_values = [50, 100, 200]  # Example grid sizes for x-axis contour
ny_values = [50, 100, 200]  # Example grid sizes for y-axis contour

# Initialize data storage for error metrics as 2D arrays
error_data = {
    'max_error_u': np.zeros((len(nx_values), len(ny_values))),
    'mean_error_u': np.zeros((len(nx_values), len(ny_values))),
    'rmse_u': np.zeros((len(nx_values), len(ny_values))),
    'max_error_v': np.zeros((len(nx_values), len(ny_values))),
    'mean_error_v': np.zeros((len(nx_values), len(ny_values))),
    'rmse_v': np.zeros((len(nx_values), len(ny_values)))
}

# Modify the testing_fp_precision function to return error metrics
def testing_fp_precision(precision, nx, ny):
    # Set up grid
    dx, dy = Lx / (nx - 1), Ly / (ny - 1)
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    X, Y = np.meshgrid(x, y)

    def exact_solution(xg, yg, t):
        u = 3/4 - 1./(4 * (1 + np.exp((-4 * xg + 4 * yg - t) * Re / 32)))
        v = 3/4 + 1./(4 * (1 + np.exp((-4 * xg + 4 * yg - t) * Re / 32)))
        return precision(u), precision(v)

    u_numerical, v_numerical = exact_solution(X, Y, 0)
    u_next, v_next = np.zeros_like(u_numerical), np.zeros_like(v_numerical)
    prev_dudt, prev_dvdt = np.zeros_like(u_numerical), np.zeros_like(v_numerical)

    for t in np.arange(dt, T_final + dt, dt):
        dudx, dudy = precision(np.gradient(u_numerical, dx, dy))
        dvdx, dvdy = precision(np.gradient(v_numerical, dx, dy))
        d2udx2, _ = precision(np.gradient(dudx, dx))
        _, d2udy2 = precision(np.gradient(dudy, dy))
        d2vdx2, _ = precision(np.gradient(dvdx, dx))
        _, d2vdy2 = precision(np.gradient(dvdy, dy))
        
        dudt = precision(-(u_numerical * dudx + v_numerical * dudy) + (d2udx2 + d2udy2) / Re)
        dvdt = precision(-(u_numerical * dvdx + v_numerical * dvdy) + (d2vdx2 + d2vdy2) / Re)

        if t > dt:
            u_next = precision(u_numerical + dt * (1.5 * dudt - 0.5 * prev_dudt))
            v_next = precision(v_numerical + dt * (1.5 * dvdt - 0.5 * prev_dvdt))
        else:
            u_next = precision(u_numerical + dt * dudt)
            v_next = precision(v_numerical + dt * dvdt)

        u_exact, v_exact = exact_solution(X, Y, t)
        u_next[0, :], u_next[:, 0] = u_exact[0, :], u_exact[:, 0]
        v_next[0, :], v_next[:, 0] = v_exact[0, :], v_exact[:, 0]

        prev_dudt, prev_dvdt = dudt, dvdt
        u_numerical, u_next = u_next, u_numerical
        v_numerical, v_next = v_next, v_numerical

    def compute_error(u_exact, v_exact, u_numerical, v_numerical):
        error_u, error_v = np.abs(u_exact - u_numerical), np.abs(v_exact - v_numerical)
        return np.max(error_u), np.mean(error_u), np.sqrt(np.mean(error_u**2)), \
               np.max(error_v), np.mean(error_v), np.sqrt(np.mean(error_v**2))

    return compute_error(u_exact, v_exact, u_numerical, v_numerical)

# Run tests over precision and grid sizes
for fp in [np.float16, np.float32, np.float64]:
    precision_str = {np.float16: "16-bit", np.float32: "32-bit", np.float64: "64-bit"}[fp]
    for i, nx in enumerate(nx_values):
        for j, ny in enumerate(ny_values):
            max_error_u, mean_error_u, rmse_u, max_error_v, mean_error_v, rmse_v = testing_fp_precision(fp, nx, ny)
            error_data['max_error_u'][i, j] = max_error_u
            error_data['mean_error_u'][i, j] = mean_error_u
            error_data['rmse_u'][i, j] = rmse_u
            error_data['max_error_v'][i, j] = max_error_v
            error_data['mean_error_v'][i, j] = mean_error_v
            error_data['rmse_v'][i, j] = rmse_v

    # Generate contour plots for each error type and precision for both u and v
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Error Analysis at {precision_str} Precision')

    error_types = ['max_error', 'mean_error', 'rmse']
    titles = ['Max Error', 'Mean Error', 'RMSE']

    for idx, (error_type, title) in enumerate(zip(error_types, titles)):
        # Plot for u
        cf_u = axs[0, idx].contourf(nx_values, ny_values, error_data[f"{error_type}_u"], levels=20, cmap='viridis')
        fig.colorbar(cf_u, ax=axs[0, idx], orientation='vertical')
        axs[0, idx].set_title(f'{title} for u')
        axs[0, idx].set_xlabel('nx')
        axs[0, idx].set_ylabel('ny')

        # Plot for v
        cf_v = axs[1, idx].contourf(nx_values, ny_values, error_data[f"{error_type}_v"], levels=20, cmap='viridis')
        fig.colorbar(cf_v, ax=axs[1, idx], orientation='vertical')
        axs[1, idx].set_title(f'{title} for v')
        axs[1, idx].set_xlabel('nx')
        axs[1, idx].set_ylabel('ny')

    plt.tight_layout()
    plt.show()
