import numpy as np
import matplotlib.pyplot as plt

def test_floating_point_precision_range_with_plot(min_large_num, max_large_num, small_num, num_steps):
    # Enhanced resolution in the range between 1e3 and 1e4, and standard resolution elsewhere
    large_nums_fine = np.logspace(np.log10(1e3), np.log10(1e4), num=int(num_steps * 2))
    large_nums_coarse = np.logspace(np.log10(1e4), np.log10(max_large_num), num=num_steps)
    large_nums = np.unique(np.concatenate((large_nums_fine, large_nums_coarse)))

    # Prepare lists for error tracking
    abs_errors_16_add = []
    rel_errors_16_add = []
    abs_errors_32_add = []
    rel_errors_32_add = []
    
    abs_errors_16_mul = []
    rel_errors_16_mul = []
    abs_errors_32_mul = []
    rel_errors_32_mul = []
    
    abs_errors_16_div = []
    rel_errors_16_div = []
    abs_errors_32_div = []
    rel_errors_32_div = []

    print(f"Testing floating point precision for a range of large numbers with a constant small number: {small_num}\n")

    # Data type limits for np.float16
    float16_max = np.finfo(np.float16).max
    float16_min = np.finfo(np.float16).tiny

    for large_num in large_nums:
        # Set up for different precision levels
        large_num_16 = np.float16(large_num)
        small_num_16 = np.float16(small_num)

        large_num_32 = np.float32(large_num)
        small_num_32 = np.float32(small_num)

        large_num_64 = np.float64(large_num)
        small_num_64 = np.float64(small_num)

        # Safe addition (example for 16-bit, similarly for 32-bit and 64-bit)
        result_16_add = np.float16(large_num_16 + small_num_16)
        result_32_add = np.float32(large_num_32 + small_num_32)
        result_64_add = np.float64(large_num_64 + small_num_64)

        # Safe multiplication
        product_16 = large_num_16 * small_num_16
        result_16_mul = np.float16(0) if (product_16 > float16_max or product_16 < float16_min) else np.float16(product_16)
        result_32_mul = np.float32(large_num_32 * small_num_32)
        result_64_mul = np.float64(large_num_64 * small_num_64)

        # Safe division
        result_16_div = np.inf if small_num_16 == 0 else np.float16(large_num_16 / small_num_16)
        result_32_div = np.float32(large_num_32 / small_num_32)
        result_64_div = np.float64(large_num_64 / small_num_64)

        # Error calculation for each operation and type
        abs_error_16_add = abs(result_16_add - result_64_add)
        rel_error_16_add = abs_error_16_add / abs(result_64_add) if result_64_add != 0 else np.inf

        abs_error_32_add = abs(result_32_add - result_64_add)
        rel_error_32_add = abs_error_32_add / abs(result_64_add) if result_64_add != 0 else np.inf

        abs_error_16_mul = abs(result_16_mul - result_64_mul)
        rel_error_16_mul = abs_error_16_mul / abs(result_64_mul) if result_64_mul != 0 else np.inf

        abs_error_32_mul = abs(result_32_mul - result_64_mul)
        rel_error_32_mul = abs_error_32_mul / abs(result_64_mul) if result_64_mul != 0 else np.inf

        abs_error_16_div = abs(result_16_div - result_64_div)
        rel_error_16_div = abs_error_16_div / abs(result_64_div) if result_64_div != 0 else np.inf

        abs_error_32_div = abs(result_32_div - result_64_div)
        rel_error_32_div = abs_error_32_div / abs(result_64_div) if result_64_div != 0 else np.inf

        # Store errors in lists
        abs_errors_16_add.append(abs_error_16_add)
        rel_errors_16_add.append(rel_error_16_add)
        abs_errors_32_add.append(abs_error_32_add)
        rel_errors_32_add.append(rel_error_32_add)
        
        abs_errors_16_mul.append(abs_error_16_mul)
        rel_errors_16_mul.append(rel_error_16_mul)
        abs_errors_32_mul.append(abs_error_32_mul)
        rel_errors_32_mul.append(rel_error_32_mul)
        
        abs_errors_16_div.append(abs_error_16_div)
        rel_errors_16_div.append(rel_error_16_div)
        abs_errors_32_div.append(abs_error_32_div)
        rel_errors_32_div.append(rel_error_32_div)



    # Plot individual results for each operation
    plt.figure(figsize=(12, 12))

    # Plot absolute errors for addition
    plt.subplot(3, 2, 1)
    plt.plot(large_nums, abs_errors_16_add, label="Abs Error (16-bit Addition)", marker='o', linestyle='--')
    plt.plot(large_nums, abs_errors_32_add, label="Abs Error (32-bit Addition)", marker='o', linestyle='-')
    plt.xscale('log')
    plt.yscale('log')
    plt.title("Absolute Errors in Addition")
    plt.xlabel("Large Number (log scale)")
    plt.ylabel("Absolute Error (log scale)")
    plt.legend()

    # Plot relative errors for addition
    plt.subplot(3, 2, 2)
    plt.plot(large_nums, rel_errors_16_add, label="Rel Error (16-bit Addition)", marker='o', linestyle='--')
    plt.plot(large_nums, rel_errors_32_add, label="Rel Error (32-bit Addition)", marker='o', linestyle='-')
    plt.xscale('log')
    plt.yscale('log')
    plt.title("Relative Errors in Addition")
    plt.xlabel("Large Number (log scale)")
    plt.ylabel("Relative Error (log scale)")
    plt.legend()

    # Plot absolute errors for multiplication
    plt.subplot(3, 2, 3)
    plt.plot(large_nums, abs_errors_16_mul, label="Abs Error (16-bit Multiplication)", marker='o', linestyle='--')
    plt.plot(large_nums, abs_errors_32_mul, label="Abs Error (32-bit Multiplication)", marker='o', linestyle='-')
    plt.xscale('log')
    plt.yscale('log')
    plt.title("Absolute Errors in Multiplication")
    plt.xlabel("Large Number (log scale)")
    plt.ylabel("Absolute Error (log scale)")
    plt.legend()

    # Plot relative errors for multiplication
    plt.subplot(3, 2, 4)
    plt.plot(large_nums, rel_errors_16_mul, label="Rel Error (16-bit Multiplication)", marker='o', linestyle='--')
    plt.plot(large_nums, rel_errors_32_mul, label="Rel Error (32-bit Multiplication)", marker='o', linestyle='-')
    plt.xscale('log')
    plt.yscale('log')
    plt.title("Relative Errors in Multiplication")
    plt.xlabel("Large Number (log scale)")
    plt.ylabel("Relative Error (log scale)")
    plt.legend()

    # Plot absolute errors for division
    plt.subplot(3, 2, 5)
    plt.plot(large_nums, abs_errors_16_div, label="Abs Error (16-bit Division)", marker='o', linestyle='--')
    plt.plot(large_nums, abs_errors_32_div, label="Abs Error (32-bit Division)", marker='o', linestyle='-')
    plt.xscale('log')
    plt.yscale('log')
    plt.title("Absolute Errors in Division")
    plt.xlabel("Large Number (log scale)")
    plt.ylabel("Absolute Error (log scale)")
    plt.legend()

    # Plot relative errors for division
    plt.subplot(3, 2, 6)
    plt.plot(large_nums, rel_errors_16_div, label="Rel Error (16-bit Division)", marker='o', linestyle='--')
    plt.plot(large_nums, rel_errors_32_div, label="Rel Error (32-bit Division)", marker='o', linestyle='-')
    plt.xscale('log')
    plt.yscale('log')
    plt.title("Relative Errors in Division")
    plt.xlabel("Large Number (log scale)")
    plt.ylabel("Relative Error (log scale)")
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Now create the combined plot for all operations
    plt.figure(figsize=(12, 8))

    # Plot combined absolute errors
    plt.subplot(2, 1, 1)
    plt.plot(large_nums, abs_errors_16_add, label="Abs Error (16-bit Addition)", marker='o', linestyle='--')
    plt.plot(large_nums, abs_errors_32_add, label="Abs Error (32-bit Addition)", marker='o', linestyle='-')
    plt.plot(large_nums, abs_errors_16_mul, label="Abs Error (16-bit Multiplication)", marker='x', linestyle='--')
    plt.plot(large_nums, abs_errors_32_mul, label="Abs Error (32-bit Multiplication)", marker='x', linestyle='-')
    plt.plot(large_nums, abs_errors_16_div, label="Abs Error (16-bit Division)", marker='s', linestyle='--')
    plt.plot(large_nums, abs_errors_32_div, label="Abs Error (32-bit Division)", marker='s', linestyle='-')
    plt.xscale('log')
    plt.yscale('log')
    plt.title("Absolute Errors for All Operations")
    plt.xlabel("Large Number (log scale)")
    plt.ylabel("Absolute Error (log scale)")
    plt.legend()

    # Plot combined relative errors
    plt.subplot(2, 1, 2)
    plt.plot(large_nums, rel_errors_16_add, label="Rel Error (16-bit Addition)", marker='o', linestyle='--')
    plt.plot(large_nums, rel_errors_32_add, label="Rel Error (32-bit Addition)", marker='o', linestyle='-')
    plt.plot(large_nums, rel_errors_16_mul, label="Rel Error (16-bit Multiplication)", marker='x', linestyle='--')
    plt.plot(large_nums, rel_errors_32_mul, label="Rel Error (32-bit Multiplication)", marker='x', linestyle='-')
    plt.plot(large_nums, rel_errors_16_div, label="Rel Error (16-bit Division)", marker='s', linestyle='--')
    plt.plot(large_nums, rel_errors_32_div, label="Rel Error (32-bit Division)", marker='s', linestyle='-')
    plt.xscale('log')
    plt.yscale('log')
    plt.title("Relative Errors for All Operations")
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
