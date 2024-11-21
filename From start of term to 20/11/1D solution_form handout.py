import numpy as np
import matplotlib.pyplot as plt

# Parameters
nx = 100                     # Number of spatial points
nt = 120                     # Number of time steps
dx = 2 * np.pi / (nx - 1)    # Spatial step size
dt = 0.0036                  # Time step size
nu = 0.07                    # Viscosity coefficient

# Initialization
x = np.linspace(0, 2*np.pi, nx)
u = np.where((0.5 < x) & (x < 1), 1.0, 0.5)  # Initial condition

def upwind(u):
    un = np.copy(u)
    for n in range(nt):
        un = u.copy()
        for i in range(1, nx-1):
            u[i] = un[i] - un[i] * dt / dx * (un[i] - un[i-1]) + nu*dt/dx**2 * (un[i+1] - 2*un[i] + un[i-1])
        u[0] = un[0] - un[0] * dt / dx * (un[0] - un[-1]) + nu*dt/dx**2 * (un[1] - 2*un[0] + un[-1])  # Handling periodic BC
        u[-1] = u[0]  # Periodic boundary condition
    return u

def maccormack(u):
    un = np.copy(u)
    ustar = np.copy(u)
    for n in range(nt):
        for i in range(1, nx-1):
            ustar[i] = un[i] - dt/dx * 0.5 * (un[i+1]**2 - un[i]**2) + nu*dt/dx**2 * (un[i+1] - 2*un[i] + un[i-1])
        ustar[-1] = un[-1] - dt/dx * 0.5 * (un[0]**2 - un[-1]**2) + nu*dt/dx**2 * (un[0] - 2*un[-1] + un[-2])  # Handling periodic BC
        for i in range(1, nx):
            u[i] = 0.5 * (un[i] + ustar[i] - dt/dx * 0.5 * (ustar[i]**2 - ustar[i-1]**2))
        u[0] = 0.5 * (un[0] + ustar[0] - dt/dx * 0.5 * (ustar[0]**2 - ustar[-1]**2))  # Handling periodic BC
        u[-1] = u[0]  # Periodic boundary condition
    return u

u_upwind = upwind(np.copy(u))
u_maccormack = maccormack(np.copy(u))

plt.figure(figsize=(10, 8))
plt.subplot(211)
plt.plot(x, u, label='Initial Condition')
plt.plot(x, u_upwind, 'o-', label='Upwind Scheme')
plt.title('Figure 19: Viscous Burgers Equation Using Upwind Scheme')
plt.xlabel('x')
plt.ylabel('u')
plt.legend()

plt.subplot(212)
plt.plot(x, u, label='Initial Condition')
plt.plot(x, u_maccormack, 'o-', label='MacCormack Scheme')
plt.title('Figure 20: Viscous Burgers Equation Using MacCormack Scheme')
plt.xlabel('x')
plt.ylabel('u')
plt.legend()

plt.tight_layout()
plt.show()
