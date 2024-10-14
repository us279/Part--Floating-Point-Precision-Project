import numpy as np

def test_floating_point_precision_with_formats(large_num, small_num):
    print(f"Testing precision for large number: {large_num} and small number: {small_num}\\n")

    # Step 2: Perform addition with different precisions
    large_num_16 = np.float16(large_num)
    small_num_16 = np.float16(small_num)
    result_16 = large_num_16 + small_num_16

    large_num_32 = np.float32(large_num)
    small_num_32 = np.float32(small_num)
    result_32 = large_num_32 + small_num_32

    large_num_64 = np.float64(large_num)
    small_num_64 = np.float64(small_num)
    result_64 = large_num_64 + small_num_64

    # Step 3: Calculate the absolute and relative errors between precisions
    abs_error_16_vs_64 = abs(result_16 - result_64)
    rel_error_16_vs_64 = abs_error_16_vs_64 / abs(result_64) if result_64 != 0 else 0

    abs_error_32_vs_64 = abs(result_32 - result_64)
    rel_error_32_vs_64 = abs_error_32_vs_64 / abs(result_64) if result_64 != 0 else 0

    # Step 4: Output the results and errors
    print(f"Result with float16 precision: {result_16}")
    print(f"Result with float32 precision: {result_32}")
    print(f"Result with float64 precision: {result_64}")
    print(f"Absolute error (float16 vs float64): {abs_error_16_vs_64:.1e}")
    print(f"Relative error (float16 vs float64): {rel_error_16_vs_64:.1e}")
    print(f"Absolute error (float32 vs float64): {abs_error_32_vs_64:.1e}")
    print(f"Relative error (float32 vs float64): {rel_error_32_vs_64:.1e}\\n")

# Step 5: Run the test with a large number and a small number
large_number = 1e16
small_number = 1e-10

test_floating_point_precision_with_formats(large_number, small_number)
