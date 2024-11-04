import numpy as np
import matplotlib.pyplot as plt

def initial_condition(x):
    return np.sin(np.pi * x)

def crank_nicolson(u, nu, dx, dt, N):
    # Set up the coefficients matrix for the Crank-Nicolson scheme
    sigma = nu * dt / (2 * dx**2)
    A = np.diag((1 + 2 * sigma) * np.ones(N)) + \
        np.diag(-sigma * np.ones(N - 1), k=1) + \
        np.diag(-sigma * np.ones(N - 1), k=-1)
    B = np.diag((1 - 2 * sigma) * np.ones(N)) + \
        np.diag(sigma * np.ones(N - 1), k=1) + \
        np.diag(sigma * np.ones(N - 1), k=-1)
    A[0, :] = A[-1, :] = B[0, :] = B[-1, :] = 0  # Boundary conditions
    A[0, 0] = A[-1, -1] = B[0, 0] = B[-1, -1] = 1

    # Solve the linear system
    u_next = np.linalg.solve(A, B.dot(u))
    return u_next

def solve_burgers(x, t, nu, dx, dt):
    N = len(x)
    u = initial_condition(x)
    U = np.zeros((len(t), N))
    U[0, :] = u

    for n in range(1, len(t)):
        u = crank_nicolson(u, nu, dx, dt, N)
        # Apply upwind scheme for nonlinear term
        u[1:-1] = u[1:-1] - dt / (4 * dx) * (u[2:]**2 - u[:-2]**2)
        U[n, :] = u

    return U

# Parameters
L = 1.0     # Length of the domain
T = 1.0     # Total time
N = 100     # Number of spatial points
M = 200     # Number of time points
nu = 0.01   # Viscosity coefficient
x = np.linspace(0, L, N)
t = np.linspace(0, T, M)
dx = x[1] - x[0]
dt = t[1] - t[0]

# Solve
U = solve_burgers(x, t, nu, dx, dt)

# Plotting
plt.figure(figsize=(10, 5))
plt.imshow(U, extent=[0, T, 0, L], origin='lower', aspect='auto')
plt.colorbar(label='u(x,t)')
plt.xlabel('Time')
plt.ylabel('Space')
plt.title('Numerical solution of the Burgers’ equation')
plt.show()
