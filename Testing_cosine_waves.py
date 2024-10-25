import numpy as np
import matplotlib.pyplot as plt
import mpmath as mp

# Set the decimal places to maximum for mpmath
mp.dps = 50  # set the precision

# Generate x values
x_values = np.linspace(-2*np.pi, 2*np.pi, 10000)

# Calculate cosine values using mpmath
cosine_mpmath = np.array([float(mp.cos(x)) for x in x_values])

# Calculate cosine values using numpy with different precisions
cosine_np_float64 = np.cos(x_values.astype(np.float64))
cosine_np_float32 = np.cos(x_values.astype(np.float32))
cosine_np_float16 = np.cos(x_values.astype(np.float16))

# Function to compute the central difference for sine
def central_difference(x, h, dtype):
    x = np.array(x, dtype=dtype)
    f_x_plus_h = np.sin(x + h)
    f_x_minus_h = np.sin(x - h)
    return (f_x_plus_h - f_x_minus_h) / (2 * h)

# Define a small step size for the central difference
h = 1e-5

# Calculate using central difference method for sine
sine_cd_float64 = central_difference(x_values, h, np.float64)
sine_cd_float32 = central_difference(x_values, h, np.float32)
sine_cd_float16 = central_difference(x_values, h, np.float16)

# Plotting all cosine and sine central difference waves on the same graph
plt.figure(figsize=(10, 6))
plt.plot(x_values, cosine_mpmath, label='mpmath Cosine (High Precision)', color='black')
plt.plot(x_values, cosine_np_float64, label='NumPy Float64 Cosine', linestyle='--')
plt.plot(x_values, cosine_np_float32, label='NumPy Float32 Cosine', linestyle=':')
plt.plot(x_values, cosine_np_float16, label='NumPy Float16 Cosine', linestyle='-.')
plt.plot(x_values, sine_cd_float64, label='Central Diff Sine Float64', linestyle='--', color='red')
plt.plot(x_values, sine_cd_float32, label='Central Diff Sine Float32', linestyle=':', color='green')
plt.plot(x_values, sine_cd_float16, label='Central Diff Sine Float16', linestyle='-.', color='blue')

# Customizing the x-axis to display pi notations
plt.xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi],
           ['0', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$', r'$2\pi$'])

plt.title('Comparison of Cosine Calculations and Central Difference Sine at Different Precisions')
plt.xlabel('x (radians)')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()

# Continue with error plots...
# Calculate errors for cosine and sine central difference
cosine_errors = {
    'Float64': cosine_np_float64,
    'Float32': cosine_np_float32,
    'Float16': cosine_np_float16
}

cd_sine_errors = {
    'Float64': sine_cd_float64,
    'Float32': sine_cd_float32,
    'Float16': sine_cd_float16
}

# Plotting errors for NumPy cosine vs. mpmath
plt.figure(figsize=(14, 14))
for i, (key, cosine) in enumerate(cosine_errors.items(), 1):
    abs_error_cos = np.abs(cosine_mpmath - cosine)
    rel_error_cos = abs_error_cos / (np.abs(cosine_mpmath) + 1e-20)
    
    plt.subplot(3, 2, 2*i-1)
    plt.plot(x_values, abs_error_cos, label=f'Abs Error Cosine {key}', color='blue')
    plt.title(f'Absolute Errors for Cosine {key}')
    plt.xlabel('x (radians)')
    plt.ylabel('Absolute Error')
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 2, 2*i)
    plt.plot(x_values, rel_error_cos, label=f'Rel Error Cosine {key}', color='blue')
    plt.title(f'Relative Errors for Cosine {key}')
    plt.xlabel('x (radians)')
    plt.ylabel('Relative Error')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()

# Plotting errors for central difference sine vs. mpmath
plt.figure(figsize=(14, 14))
for i, (key, cd_sine) in enumerate(cd_sine_errors.items(), 1):
    abs_error_cd = np.abs(cosine_mpmath - cd_sine)
    rel_error_cd = abs_error_cd / (np.abs(cosine_mpmath) + 1e-20)
    
    plt.subplot(3, 2, 2*i-1)
    plt.plot(x_values, abs_error_cd, label=f'Abs Error CD Sine {key}', color='red')
    plt.title(f'Absolute Errors for CD Sine {key}')
    plt.xlabel('x (radians)')
    plt.ylabel('Absolute Error')
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 2, 2*i)
    plt.plot(x_values, rel_error_cd, label=f'Rel Error CD Sine {key}', color='red')
    plt.title(f'Relative Errors for CD Sine {key}')
    plt.xlabel('x (radians)')
    plt.ylabel('Relative Error')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()
