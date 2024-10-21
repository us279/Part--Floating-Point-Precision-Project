import numpy as np
import matplotlib.pyplot as plt

def test_floating_point_precision_range_with_plot(min_large_num, max_large_num, small_num, num_steps):
    # Step size in log space to generate the range of large numbers
    large_nums = np.logspace(np.log10(min_large_num), np.log10(max_large_num), num=num_steps)

    # Lists to store errors for addition, multiplication, and division
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

    for large_num in large_nums:
        # Step 2: Perform addition, multiplication, and division with different precisions
        
        # ---- Addition ----
        large_num_16 = np.float16(large_num)
        small_num_16 = np.float16(small_num)
        result_16_add = np.float16(large_num_16 + small_num_16)

        large_num_32 = np.float32(large_num)
        small_num_32 = np.float32(small_num)
        result_32_add = np.float32(large_num_32 + small_num_32)

        large_num_64 = np.float64(large_num)
        small_num_64 = np.float64(small_num)
        result_64_add = np.float64(large_num_64 + small_num_64)

        # ---- Multiplication ----
        result_16_mul = np.float16(large_num_16 * small_num_16)
        result_32_mul = np.float32(large_num_32 * small_num_32)
        result_64_mul = np.float64(large_num_64 * small_num_64)

        # ---- Division ----
        result_16_div = np.float16(large_num_16 / small_num_16)
        result_32_div = np.float32(large_num_32 / small_num_32)
        result_64_div = np.float64(large_num_64 / small_num_64)

        # Step 3: Calculate the absolute and relative errors for each operation
        # Addition
        abs_error_16_add = abs(result_16_add - result_64_add)
        rel_error_16_add = abs_error_16_add / abs(result_64_add) if result_64_add != 0 else 0

        abs_error_32_add = abs(result_32_add - result_64_add)
        rel_error_32_add = abs_error_32_add / abs(result_64_add) if result_64_add != 0 else 0

        # Multiplication
        abs_error_16_mul = abs(result_16_mul - result_64_mul)
        rel_error_16_mul = abs_error_16_mul / abs(result_64_mul) if result_64_mul != 0 else 0

        abs_error_32_mul = abs(result_32_mul - result_64_mul)
        rel_error_32_mul = abs_error_32_mul / abs(result_64_mul) if result_64_mul != 0 else 0

        # Division
        abs_error_16_div = abs(result_16_div - result_64_div)
        rel_error_16_div = abs_error_16_div / abs(result_64_div) if result_64_div != 0 else 0

        abs_error_32_div = abs(result_32_div - result_64_div)
        rel_error_32_div = abs_error_32_div / abs(result_64_div) if result_64_div != 0 else 0

        # Store errors in the lists
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

    # Print results for addition
    print(f"{'Large Number':>15} {'Abs Error (16-bit)':>20} {'Rel Error (16-bit)':>20} {'Abs Error (32-bit)':>20} {'Rel Error (32-bit)':>20}")
    print("-" * 95)
    for i in range(num_steps):
        print(f"{large_nums[i]:>15.1e} {abs_errors_16_add[i]:>20.1e} {rel_errors_16_add[i]:>20.1e} {abs_errors_32_add[i]:>20.1e} {rel_errors_32_add[i]:>20.1e}")

    # Plot the results
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

# Step 5: Run the test with a range of large numbers and a constant small number
min_large_number = 1e3   # Start at 10
