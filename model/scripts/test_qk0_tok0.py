import numpy as np

Q = np.array([519, -2984, 1971, -4209, -79, 334, 420, 476, 560, 1641, -2631, 1602, -938, -1781, -2766, -2001, -2820, 1001, 1700, -187, 215, -1171, -778, -754, -2399, 2688, -853, 1176, -225, 2281, -2917, -1703, 1580, -1014, -1138, 1137, 943, -1882, 3121, -703, 3209, -4662, -433, -130, -251, -2711, -1353, -1547, 1099, -471, 1293, 2395, 1597, 4042, -2915, -1239, -3384, -1274, 3589, -772, 3918, -343, 2410, 119])

K = np.array([-65, -1549, 2235, 466, -609, -3572, -2249, 220, 1854, 1637, 6056, -3966, 1858, -487, 3308, 926, 257, -2680, -1075, -938, 1279, -506, -2398, 1932, -1220, 1272, -3135, -456, -548, -2112, -469, -363, 5141, -31, 4100, -4359, -2378, -2984, -3133, -1693, 887, -167, -1426, -1497, 92, -2979, 3602, -523, -1083, 831, -240, 1446, -3140, -929, -743, -2387, -1223, 2997, 232, -3683, -3133, -1159, -1058, 4035])

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