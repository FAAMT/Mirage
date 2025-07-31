import numpy as np

# Define Q and K matrices (both heads)
Q = np.array([
    [519, -2984, 1971, -4209, -79, 334, 420, 476, 560, 1641, -2631, 1602, -938, -1781, -2766, -2001, -2820, 1001, 1700, -187, 215, -1171, -778, -754, -2399, 2688, -853, 1176, -225, 2281, -2917, -1703, 1580, -1014, -1138, 1137, 943, -1882, 3121, -703, 3209, -4662, -433, -130, -251, -2711, -1353, -1547, 1099, -471, 1293, 2395, 1597, 4042, -2915, -1239, -3384, -1274, 3589, -772, 3918, -343, 2410, 119],
    [3920, 9, -2909, -2152, 323, 1842, 2785, 1558, 4430, 3954, -1436, 2130, -775, 206, 543, 75, -365, -554, -887, 323, -3923, 327, 2240, -3972, -3954, 78, 108, 2948, 1266, 1000, -2441, 277, 3078, 1805, 115, -951, 2681, 67, 2478, -401, 3204, -3240, -4, -349, -761, -82, 1246, -4026, 4495, 80, 191, 6231, 3401, 106, 230, -993, -822, 96, 849, 761, 4255, -1375, 938, 473]
])

K = np.array([
    [-65, -1549, 2235, 466, -609, -3572, -2249, 220, 1854, 1637, 6056, -3966, 1858, -487, 3308, 926, 257, -2680, -1075, -938, 1279, -506, -2398, 1932, -1220, 1272, -3135, -456, -548, -2112, -469, -363, 5141, -31, 4100, -4359, -2378, -2984, -3133, -1693, 887, -167, -1426, -1497, 92, -2979, 3602, -523, -1083, 831, -240, 1446, -3140, -929, -743, -2387, -1223, 2997, 232, -3683, -3133, -1159, -1058, 4035],
    [-1845, 907, 3141, 3470, -779, -3949, -302, 800, 1123, 4765, 6561, -1667, -6467, -1155, 3109, 649, -1583, 774, 2011, -2872, 2196, 941, -143, 5609, -501, 5205, 423, -236, -2098, -2728, 124, -93, 649, 2873, -2951, -5055, -2879, -2556, 2027, -175, -2758, 1491, -1389, -1879, -277, -2816, 1214, -3980, -4690, -2437, -1494, -1088, -1378, -1017, -459, -1519, -3032, -79, 3713, -259, -2305, -2576, 4885, 1809]
])

# Right-shift by 8 (equivalent to floor division by 256)
Q_shifted = Q // 256
K_shifted = K // 256

# Initialize accumulators
total_accum = 0
p0 = 0

print("=== Multi-Head Attention Calculation ===")
print("Processing Q[0] @ K[0].T:")

# Process Q[0] @ K[0].T
accum = 0
for i in range(0, len(Q_shifted[0]) - 4, 5):
    chunk_q = Q_shifted[0, i:i+5]
    chunk_k = K_shifted[0, i:i+5]
    dot = np.dot(chunk_q, chunk_k)
    accum += dot
    print(f"MAC Q0K0 [{i}-{i+4}] {dot} (Accum: {accum})")
    if i == 0:
        p0 = dot

# Remaining elements
if len(Q_shifted[0]) % 5 != 0:
    remaining_q = Q_shifted[0, -4:]
    remaining_k = K_shifted[0, -4:]
    dot = np.dot(remaining_q, remaining_k)
    accum += dot
    print(f"MAC Q0K0 [60-63] {dot + p0} (Accum: {accum})")

total_accum += accum
print(f"Q0K0 Total -> {accum}\n")

print("Processing Q[0] @ K[1].T:")
# Process Q[0] @ K[1].T
accum = 0
for i in range(0, len(Q_shifted[0]) - 4, 5):
    chunk_q = Q_shifted[0, i:i+5]
    chunk_k = K_shifted[1, i:i+5]
    dot = np.dot(chunk_q, chunk_k)
    accum += dot
    print(f"MAC Q0K1 [{i}-{i+4}] {dot} (Accum: {accum})")

# Remaining elements
if len(Q_shifted[0]) % 5 != 0:
    remaining_q = Q_shifted[0, -4:]
    remaining_k = K_shifted[1, -4:]
    dot = np.dot(remaining_q, remaining_k)
    accum += dot
    print(f"MAC Q0K1 [60-63] {dot + p0} (Accum: {accum})")

total_accum += accum
print(f"Q0K1 Total -> {accum}\n")

print("Processing Q[1] @ K[0].T:")
# Process Q[1] @ K[0].T
accum = 0
for i in range(0, len(Q_shifted[1]) - 4, 5):
    chunk_q = Q_shifted[1, i:i+5]
    chunk_k = K_shifted[0, i:i+5]
    dot = np.dot(chunk_q, chunk_k)
    accum += dot
    print(f"MAC Q1K0 [{i}-{i+4}] {dot} (Accum: {accum})")

# Remaining elements
if len(Q_shifted[1]) % 5 != 0:
    remaining_q = Q_shifted[1, -4:]
    remaining_k = K_shifted[0, -4:]
    dot = np.dot(remaining_q, remaining_k)
    accum += dot
    print(f"MAC Q1K0 [60-63] {dot + p0} (Accum: {accum})")

total_accum += accum
print(f"Q1K0 Total -> {accum}\n")

print("Processing Q[1] @ K[1].T:")
# Process Q[1] @ K[1].T
accum = 0
for i in range(0, len(Q_shifted[1]) - 4, 5):
    chunk_q = Q_shifted[1, i:i+5]
    chunk_k = K_shifted[1, i:i+5]
    dot = np.dot(chunk_q, chunk_k)
    accum += dot
    print(f"MAC Q1K1 [{i}-{i+4}] {dot} (Accum: {accum})")

# Remaining elements
if len(Q_shifted[1]) % 5 != 0:
    remaining_q = Q_shifted[1, -4:]
    remaining_k = K_shifted[1, -4:]
    dot = np.dot(remaining_q, remaining_k)
    accum += dot
    print(f"MAC Q1K1 [60-63] {dot + p0} (Accum: {accum})")

total_accum += accum
print(f"Q1K1 Total -> {accum}\n")

print(f"=== Final Attention Scores ===")
print(f"Total Accumulated: {total_accum}")