import numpy as np

Q = np.array([3920, 9, -2909, -2152, 323, 1842, 2785, 1558, 4430, 3954, -1436, 2130,
               -775, 206, 543, 75, -365, -554, -887, 323, -3923, 327, 2240, -3972,
               -3954, 78, 108, 2948, 1266, 1000, -2441, 277, 3078, 1805, 115, -951,
               2681, 67, 2478, -401, 3204, -3240, -4, -349, -761, -82, 1246, -4026,
               4495, 80, 191, 6231, 3401, 106, 230, -993, -822, 96, 849, 761,
               4255, -1375, 938, 473])

K = np.array([-1845, 907, 3141, 3470, -779, -3949, -302, 800, 1123, 4765, 6561, -1667,
               -6467, -1155, 3109, 649, -1583, 774, 2011, -2872, 2196, 941, -143, 5609,
               -501, 5205, 423, -236, -2098, -2728, 124, -93, 649, 2873, -2951, -5055,
               -2879, -2556, 2027, -175, -2758, 1491, -1389, -1879, -277, -2816, 1214, -3980,
               -4690, -2437, -1494, -1088, -1378, -1017, -459, -1519, -3032, -79, 3713, -259,
               -2305, -2576, 4885, 1809])

# Right-shift by 8 (equivalent to floor division by 256)
Q_shifted = Q // 256
K_shifted = K // 256

# Initialize dot product
dot_product = 0
accum_product = 0
p0 = 0

# Process in chunks of 5 elements
for i in range(0, len(Q_shifted) - 5, 5):
    dot_product = 0
    chunk_q = Q_shifted[i:i+5]
    chunk_k = K_shifted[i:i+5]
    dot_product += np.dot(chunk_q, chunk_k)
    accum_product += dot_product
    print(f"MAC [{i}-{i+5}] {dot_product}")

    if i == 0:
        p0 = dot_product

# Process the remaining 4 elements
if len(Q_shifted) % 5 != 0:
    dot_product = 0
    remaining_q = Q_shifted[-4:]
    remaining_k = K_shifted[-4:]
    dot_product += np.dot(remaining_q, remaining_k)
    accum_product += dot_product
    print(f"MAC [{60}-{63}] {dot_product + p0}")

print(f"MAC Total -> {accum_product}")  # Output: -1037