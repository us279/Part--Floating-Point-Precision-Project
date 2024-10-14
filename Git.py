import numpy as np
import matplotlib.pyplot as plt

def test_floating_point_precision_range_with_plot(min_large_num, max_large_num, small_num, num_steps):
    # Step size in log space to generate the range of large numbers
    large_nums = np.logspace(np.log10(min_large_num), np.log10(max_large_num), num=num_steps)

    # Lists to store errors
    abs_errors_16 = []
    rel_errors_16 = []
    abs_errors_32 = []
    rel_errors_32 = []

    print(f"Testing floating point precision for a range of large numbers with a constant small number: {small_num}\n")

    for large_num in large_nums:
        # Step 2: Perform addition with different precisions
        large_num_16 = np.float16(large_num)
        small_num_16 = np.float16(small_num)
        result_16 = np.float16(large_num_16 + small_num_16)

        large_num_32 = np.float32(large_num)
        small_num_32 = np.float32(small_num)
        result_32 = np.float32(large_num_32 + small_num_32)

        large_num_64 = np.float64(large_num)
        small_num_64 = np.float64(small_num)
        result_64 = np.float64(large_num_64 + small_num_64)

        # Step 3: Calculate the absolute and relative errors
        abs_error_16_vs_64 = abs(result_16 - result_64)
        rel_error_16_vs_64 = abs_error_16_vs_64 / abs(result_64) if result_64 != 0 else 0

        abs_error_32_vs_64 = abs(result_32 - result_64)
        rel_error_32_vs_64 = abs_error_32_vs_64 / abs(result_64) if result_64 != 0 else 0

        # Store errors in the lists
        abs_errors_16.append(abs_error_16_vs_64)
        rel_errors_16.append(rel_error_16_vs_64)
        abs_errors_32.append(abs_error_32_vs_64)
        rel_errors_32.append(rel_error_32_vs_64)

    # Print results
    print(f"{'Large Number':>15} {'Abs Error (16-bit)':>20} {'Rel Error (16-bit)':>20} {'Abs Error (32-bit)':>20} {'Rel Error (32-bit)':>20}")
    print("-" * 95)
    
    for i in range(num_steps):
        print(f"{large_nums[i]:>15.1e} {abs_errors_16[i]:>20.1e} {rel_errors_16[i]:>20.1e} {abs_errors_32[i]:>20.1e} {rel_errors_32[i]:>20.1e}")

    # Plot the results
    plt.figure(figsize=(10, 6))

    # Plot absolute errors
    plt.subplot(2, 1, 1)
    plt.plot(large_nums, abs_errors_16, label="Abs Error (16-bit)", marker='o', linestyle='--')
    plt.plot(large_nums, abs_errors_32, label="Abs Error (32-bit)", marker='o', linestyle='-')
    plt.xscale('log')
    plt.yscale('log')
    plt.title("Absolute Errors in Floating Point Addition")
    plt.xlabel("Large Number (log scale)")
    plt.ylabel("Absolute Error (log scale)")
    plt.legend()

    # Plot relative errors
    plt.subplot(2, 1, 2)
    plt.plot(large_nums, rel_errors_16, label="Rel Error (16-bit)", marker='o', linestyle='--')
    plt.plot(large_nums, rel_errors_32, label="Rel Error (32-bit)", marker='o', linestyle='-')
    plt.xscale('log')
    plt.yscale('log')
    plt.title("Relative Errors in Floating Point Addition")
    plt.xlabel("Large Number (log scale)")
    plt.ylabel("Relative Error (log scale)")
    plt.legend()

    plt.tight_layout()
    plt.show()

# Step 5: Run the test with a range of large numbers and a constant small number
min_large_number = 1e3   # Start at 10,000
max_large_number = 1e16  # Go up to 10^16
small_number = 1e-13     # Keep the small number constant
num_steps = 30           # Number of steps in the range

test_floating_point_precision_range_with_plot(min_large_number, max_large_number, small_number, num_steps)
