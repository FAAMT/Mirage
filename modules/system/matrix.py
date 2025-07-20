#!/usr/bin/env python3
import numpy as np

matrix1 = np.array([-1, 127, 1, 124, -3, 127, 1, 127, 0, 126, 2, 127, 3, 127, 1, 127, 6, 120, 6, 127, 0, 127, -1, 127, -2, 127, 3, 125, -5, 120, -11, 126, -7, 127, 2, 127, 4, 125, -1, 127, -1, 127, -6, 127, 4, 127, -3, 124, -11, 124, 2, 127, 0, 120, 3, 127, -2, 127, 3, 127, 5, 127, -10, 125, -7, 127, -4, 124, -3, 125, 1, 126, -7, 127, 1, 127, 0, 123, -2, 127, -3, 127, -8, 119, -2, 123, -2, 127, 0, 120, 3, 126, -4, 125, 3, 122, -2, 121, -3, 127, 2, 120, 1, 127, -8, 124, -5, 127, 5, 127, 5, 119, 2, 119, -5, 122, -5, 122, 2, 125, 0, 127, -3, 127, -5, 127, 5, 124, -4, 125, 0, 123, 7, 123, 1, 125, 1, 127, -3, 127, 4, 127, 0, 127, 3, 127, -1, 122, -2, 127, 11, 127, -2, 127, 7, 127, 2, 127, -1, 123, -6, 127, 1, 127, -3, 127, -4, 126, 3, 127, -1, 127, 0, 127, 0, 126, 0, 123, 6, 127, 5, 127, 0, 125, 6, 119, -8, 126, -1, 127, 2, 127, -2, 127, 7, 127, 3, 127, -4, 126, 2, 127, 1, 127, -4, 127, 20, 123, 0, 126, -1, 127, -2, 127, 1, 127, 3, 127, 10, 123, 0, 127, 7, 127, 7, 124, -3, 127, 1, 127, -2, 123, 7, 125, 6, 127, 5, 127, -14, 119, 6, 127, -5, 116, -6, 127, 0, 127, 5, 127, -6, 127, -4, 127, 0, 127])
matrix2 = np.array([3, 4, -6, -2, 4, 5, -3, 9, -12, 1, -6, 0, 1, 3, -5, 6, 3, -6, 3, -1, 3, 3, 0, 0, -5, -12, -6, 5, 12, -5, 1, 6, -2, 6, 0, -7, 2, -13, 4, 8, 6, -3, 6, -1, -10, 8, 8, 7, 2, 3, 3, 6, -1, 5, -3, 0, 4, 1, 2, 2, 6, 8, -1, 3, 0, 15, -8, 6, -14, 0, -1, 0, -10, 7, -3, 4, -1, 11, 5, -9, 3, -1, 0, 11, 3, 18, 1, -10, -11, 6, 15, -5, 6, 5, 12, 0, -3, -2, -17, -2, -4, 2, 7, -2, 0, -6, 9, -4, -4, -7, 3, -18, -10, 3, -5, -6, 3, 1, 0, 9, 4, -10, -5, -3, 3, 4, -7, 3, -12, 11, -5, -9, -7, 3, -3, 9, 7, 7, 3, 9, 6, -2, -6, 17, -2, 4, -5, -18, 9, 21, -1, -17, -7, -4, -1, 1, -4, 2, 3, -2, -12, 11, 3, 1, -6, -2, 1, 5, 2, 19, 15, -2, 16, -4, 2, 10, -6, 9, -6, 4, 3, 12, 17, -14, -7, 3, 3, 4, 14, 8, 6, 6, 1, 12, 4, 16, 3, -6, 0, 23, 3, 16, -4, -3, -5, 11, 3, -4, -4, -3, -3, 6, -12, -2, 13, -11, 8, 12, 2, 0, -9, 3, -1, -2, 9, 19, -13, 2, 17, 8, 6, 4, 16, 3, 12, 3, -14, 0, -3, 4, -6, 2, 3, 8, -5, 3, 1, 0, 5, 6, 0, 2, -3, -6, 0, -2])

# Check if the matrices have the same length
if len(matrix1) != len(matrix2):
    print("Error: Matrices must have the same length for dot product calculation.")
else:
    D_MODEL = len(matrix1) # Number of elements in the matrices

    print("Starting detailed MAC calculation:")
    current_sum_val = 0
    mac_details_batch = []
    BATCH_SIZE = 5
    terms_for_current_batch_str = []

    for i in range(D_MODEL):
        val1 = matrix1[i]
        val2 = matrix2[i]

        # In standard Python/NumPy with these integer inputs,
        # 'term' will be calculated using sufficient precision (e.g., int64)
        # so overflow is not expected here for these specific values.
        term = val1 * val2
        
        current_sum_val += term

        mac_details_batch.append(f"    Term {i}: {val1} * {val2} = {term}")
        terms_for_current_batch_str.append(f"{val1}*{val2}")

        if (i + 1) % BATCH_SIZE == 0 or (i + 1) == D_MODEL:
            print(f"  Processing elements {i - len(mac_details_batch) + 1} to {i}:")
            print(f"    Values being multiplied in this batch: matrix1[{i - len(mac_details_batch) + 1}:{i+1}] and matrix2[{i - len(mac_details_batch) + 1}:{i+1}]")
            print(f"    Calculation for this batch: sum({', '.join(terms_for_current_batch_str)})")
            for detail_line in mac_details_batch:
                print(detail_line)
            print(f"    Current cumulative sum after {i + 1} elements: {current_sum_val}")
            print("-" * 30) # Separator for batches
            mac_details_batch = [] # Reset for the next batch
            terms_for_current_batch_str = [] # Reset for the next batch

    print(f"\nFinal manually calculated dot product: {current_sum_val}")

    # For verification, calculate dot product using np.dot and compare
    dot_product_numpy = np.dot(matrix1, matrix2)
    print(f"Dot Product using np.dot for verification: {dot_product_numpy}")

    if current_sum_val == dot_product_numpy:
        print("Manual calculation matches np.dot result.")
    else:
        print("Error: Manual calculation does NOT match np.dot result!")