import numpy as np
import matplotlib.pyplot as plt
from time import time
import pandas as pd


def main_code(nx,ny,pc):
    print("Grid size is: " +str(nx)+ " by " + str(ny))

    if pc == np.float64:
        print("--------------------This is for float 64----------------------")
    elif pc == np.float32:
        print("--------------------This is for float 32----------------------")
    elif pc == np.float16:
        print("--------------------This is for float 16----------------------")
    # Parameters
    Re = 100.0
    Lx, Ly = 1.0, 1.0
    #nx, ny = 100, 100
    dx, dy = Lx / (nx - 1), Ly / (ny - 1)

    # Grid
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    X, Y = np.meshgrid(x, y)

    # Function to compute the exact solution
    def exact_solution(xg, yg, t):
        u = np.float64(3/4 - 1./(4 * (1 + np.exp((-4 * xg + 4 * yg - t) * Re / 32))))
        v = np.float64(3/4 + 1./(4 * (1 + np.exp((-4 * xg + 4 * yg - t) * Re / 32))))
        return u, v

    CFL = 0.01
    u, v = exact_solution(X, Y, 0)
    speed = np.sqrt(u**2 + v**2)
    #print("This is the mean speed: " + str(np.mean(speed)))
    max_speed = np.max(speed)
    #print("This is the max speed: " + str(max_speed))
    dt = CFL * ((dx**2 + dy**2)**(0.5)) / max_speed
    T_final = 0.36

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
        dudx, dudy = pc(np.gradient(u_numerical, dx, dy))
        dvdx, dvdy = pc(np.gradient(v_numerical, dx, dy))
        d2udx2, d2udy2 = pc(np.gradient(dudx, dx, dy))
        d2vdx2, d2vdy2 = pc(np.gradient(dvdx, dx, dy))

        # Update derivatives
        dudt = pc(-(pc(u_numerical * dudx) + pc(v_numerical * dudy)) + pc((pc(d2udx2 + d2udy2)) / Re))
        dvdt = pc(-(pc(u_numerical * dvdx) + pc(v_numerical * dvdy)) + pc((pc(d2vdx2 + d2vdy2)) / Re))


        if t == dt:
            u_next = pc(u_numerical + pc(dt * dudt))
            v_next = pc(v_numerical + pc(dt * dvdt))
        else:  
            u_next = pc(u_numerical +pc( dt * 0.5 * (pc(pc(3 * dudt) - prev_dudt))))
            v_next = pc(v_numerical +pc(dt * 0.5 * (pc(3 * dvdt) - prev_dvdt)))


        # Apply boundary conditions
        u_exact, v_exact = exact_solution(X, Y, t)
        u_next[0, :], u_next[:, 0] = u_exact[0, :], u_exact[:, 0]
        v_next[0, :], v_next[:, 0] = v_exact[0, :], v_exact[:, 0]

        # Update previous time step derivatives
        prev_dudt, prev_dvdt = dudt, dvdt

        # Swap arrays for the next step
        u_numerical, u_next = u_next, u_numerical
        v_numerical, v_next = v_next, v_numerical


    error_u = np.abs(u_exact - u_numerical)
    error_v = np.abs(v_exact - v_numerical)
    ## Time taken
    print(f"Time taken: {time() - t0:.2f} seconds")
    print(f"Number of iterations:{n}")

    def error_df():
        # Calculate metrics
        metrics = {
            "Metric": ["Mean Error"],#, "Max Error", "RMS Error"],
            "u": [
                np.mean(error_u),
                #np.max(error_u),
                #np.sqrt(np.mean(np.square(error_u)))

            ],
            "v": [
                np.mean(error_v),
                #np.max(error_v),
                #np.sqrt(np.mean(np.square(error_v)))
            ]
        }

        # Create a pandas DataFrame
        df = pd.DataFrame(metrics)

        # Display the DataFrame to verify
        print(df)
        # Plot the metrics
        # df.plot(x="Metric", kind="bar", figsize=(8, 6))
        # plt.title("Error Metrics for u and v")
        # plt.ylabel("Error Value")
        # plt.xticks(rotation=0)  # Rotate x-axis labels for better readability
        # plt.legend(title="Variable")
        # plt.tight_layout()
        # #plt.show()
        return
    

    error_df()

    # def plotting_function():

    #     # Select the middle index in the y-direction
    #     mid_y_idx = ny // 2

    #     # Extract u and v along x at the middle y-coordinate
    #     u_mid_y = u_numerical[mid_y_idx, :]
    #     v_mid_y = v_numerical[mid_y_idx, :]
    #     # Plotting
    #     fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    #     # Plot numerical u along x at mid y
    #     axs[0].plot(x, u_mid_y, label='Numerical u at mid y')
    #     axs[0].set_title('Numerical u vs. x at Mid y')
    #     axs[0].set_xlabel('x')
    #     axs[0].set_ylabel('u')
    #     axs[0].legend()

    #     # Plot numerical v along x at mid y
    #     axs[1].plot(x, v_mid_y, label='Numerical v at mid y')
    #     axs[1].set_title('Numerical v vs. x at Mid y')
    #     axs[1].set_xlabel('x')
    #     axs[1].set_ylabel('v')
    #     axs[1].legend()

    #     #plt.show()


    #     # Plotting
    #     fig, axs = plt.subplots(2, 3, figsize=(18, 12))

    #     # Plot exact u
    #     cf1 = axs[0, 0].contourf(X, Y, u_exact, cmap='viridis')
    #     plt.colorbar(cf1, ax=axs[0, 0], orientation='vertical', label='u')
    #     axs[0, 0].set_title('Exact Solution for u')
    #     axs[0, 0].set_xlabel('x')
    #     axs[0, 0].set_ylabel('y')

    #     # Plot exact v
    #     cf2 = axs[0, 1].contourf(X, Y, v_exact, cmap='viridis')
    #     plt.colorbar(cf2, ax=axs[0, 1], orientation='vertical', label='v')
    #     axs[0, 1].set_title('Exact Solution for v')
    #     axs[0, 1].set_xlabel('x')
    #     axs[0, 1].set_ylabel('y')

    #     # Plot numerical u
    #     cf3 = axs[1, 0].contourf(X, Y, u_numerical, cmap='viridis')
    #     plt.colorbar(cf3, ax=axs[1, 0], orientation='vertical', label='u')
    #     axs[1, 0].set_title('Numerical Solution for u')
    #     axs[1, 0].set_xlabel('x')
    #     axs[1, 0].set_ylabel('y')

    #     # Plot numerical v
    #     cf4 = axs[1, 1].contourf(X, Y, v_numerical, cmap='viridis')
    #     plt.colorbar(cf4, ax=axs[1, 1], orientation='vertical', label='v')
    #     axs[1, 1].set_title('Numerical Solution for v')
    #     axs[1, 1].set_xlabel('x')
    #     axs[1, 1].set_ylabel('y')

    #     # Plot difference between exact and numerical u
    #     diff_u = np.abs(u_exact - u_numerical)
    #     cf5 = axs[1, 2].contourf(X, Y, diff_u, cmap='magma')
    #     plt.colorbar(cf5, ax=axs[1, 2], orientation='vertical', label='|u_exact - u_numerical|')
    #     axs[1, 2].set_title('Absolute Difference (u)')
    #     axs[1, 2].set_xlabel('x')
    #     axs[1, 2].set_ylabel('y')

    #     # Plot difference between exact and numerical v
    #     diff_v = np.abs(v_exact - v_numerical)
    #     cf6 = axs[0, 2].contourf(X, Y, diff_v, cmap='magma')
    #     plt.colorbar(cf6, ax=axs[0, 2], orientation='vertical', label='|v_exact - v_numerical|')
    #     axs[0, 2].set_title('Absolute Difference (v)')
    #     axs[0, 2].set_xlabel('x')
    #     axs[0, 2].set_ylabel('y')

    #     plt.tight_layout()
    #     #plt.show()
    #     return

    # plotting_function()
    return np.mean(error_u), np.mean(error_v)

grid_sizes = [power for power in range(2, 120)]
errors_u_64 = []
errors_v_64 = []
errors_u_32 = []
errors_v_32 = []
errors_u_16 = []
errors_v_16 = []



for size in grid_sizes:
    for pc in [np.float16,np.float32,np.float64]:
        if pc == np.float64:
            error_u, error_v = main_code(size, size, pc)
            errors_u_64.append(error_u)
            errors_v_64.append(error_v)
        elif pc == np.float32:
            error_u, error_v = main_code(size, size, pc)
            errors_u_32.append(error_u)
            errors_v_32.append(error_v)
        elif pc == np.float16:
            error_u, error_v = main_code(size, size, pc)
            errors_u_16.append(error_u)
            errors_v_16.append(error_v)




# Plotting
plt.figure(figsize=(10, 5))
plt.plot(grid_sizes, errors_u_16, label='Mean Error u fp = 16')
plt.plot(grid_sizes, errors_v_16, label='Mean Error v fp = 16')
plt.plot(grid_sizes, errors_u_32, label='Mean Error u fp = 32')
plt.plot(grid_sizes, errors_v_32, label='Mean Error v fp = 32')
plt.plot(grid_sizes, errors_u_64, label='Mean Error u fp = 64')
plt.plot(grid_sizes, errors_v_64, label='Mean Error v fp = 64')
plt.xlabel('Grid Size')
plt.ylabel('Mean Error')
plt.title('Mean Error vs Grid Size')
plt.legend()
plt.grid(True)
plt.xscale('log')
plt.yscale('log')
plt.show()
