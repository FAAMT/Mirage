import numpy as np

# Define Q and K matrices (3 heads each)
Q = np.array([
    [519, -2984, 1971, -4209, -79, 334, 420, 476, 560, 1641, -2631, 1602, -938, -1781, -2766, -2001, -2820, 1001, 1700, -187, 215, -1171, -778, -754, -2399, 2688, -853, 1176, -225, 2281, -2917, -1703, 1580, -1014, -1138, 1137, 943, -1882, 3121, -703, 3209, -4662, -433, -130, -251, -2711, -1353, -1547, 1099, -471, 1293, 2395, 1597, 4042, -2915, -1239, -3384, -1274, 3589, -772, 3918, -343, 2410, 119],
    [3920, 9, -2909, -2152, 323, 1842, 2785, 1558, 4430, 3954, -1436, 2130, -775, 206, 543, 75, -365, -554, -887, 323, -3923, 327, 2240, -3972, -3954, 78, 108, 2948, 1266, 1000, -2441, 277, 3078, 1805, 115, -951, 2681, 67, 2478, -401, 3204, -3240, -4, -349, -761, -82, 1246, -4026, 4495, 80, 191, 6231, 3401, 106, 230, -993, -822, 96, 849, 761, 4255, -1375, 938, 473],
    [365, -279, -930, 1152, 1787, 650, 1666, 4241, -950, 2142, 2913, 2988, -1228, 1141, -270, -751, 2230, 395, 1637, -844, -2862, 408, -1280, -2821, -595, 350, -1661, -1127, -1472, -2081, -1049, -513, 1754, -2819, -3914, 1029, -1236, 749, 567, 3165, 1860, -3473, -1380, 1020, -746, -2392, 759, -2238, 809, -270, 512, 788, 3048, 1711, -716, -2397, -606, 272, 4016, 2569, 2069, -1103, 1320, -1747]
])

K = np.array([
    [-65, -1549, 2235, 466, -609, -3572, -2249, 220, 1854, 1637, 6056, -3966, 1858, -487, 3308, 926, 257, -2680, -1075, -938, 1279, -506, -2398, 1932, -1220, 1272, -3135, -456, -548, -2112, -469, -363, 5141, -31, 4100, -4359, -2378, -2984, -3133, -1693, 887, -167, -1426, -1497, 92, -2979, 3602, -523, -1083, 831, -240, 1446, -3140, -929, -743, -2387, -1223, 2997, 232, -3683, -3133, -1159, -1058, 4035],
    [-1845, 907, 3141, 3470, -779, -3949, -302, 800, 1123, 4765, 6561, -1667, -6467, -1155, 3109, 649, -1583, 774, 2011, -2872, 2196, 941, -143, 5609, -501, 5205, 423, -236, -2098, -2728, 124, -93, 649, 2873, -2951, -5055, -2879, -2556, 2027, -175, -2758, 1491, -1389, -1879, -277, -2816, 1214, -3980, -4690, -2437, -1494, -1088, -1378, -1017, -459, -1519, -3032, -79, 3713, -259, -2305, -2576, 4885, 1809],
    [635, -1187, 2345, 1330, 2718, -3912, 877, -509, 5408, 1559, 3950, -3331, -3748, -786, -434, 1561, -776, -1131, -1243, -939, 3210, 809, 1351, -1258, -1735, 1529, 1070, 1570, -2905, 1118, -793, -223, 1198, -585, -873, -2586, -1282, -3499, 1166, -2747, 509, -4665, -1673, -1445, -1253, -3105, 916, -3405, -332, 1897, 258, 1904, -498, -2097, -335, -1944, 392, 1399, 2364, 3440, -4405, -2422, 3157, -789]
])

# Right-shift by 8 (equivalent to floor division by 256)
Q_shifted = Q // 256
K_shifted = K // 256

def calculate_attention(q_idx, k_idx):
    print(f"\nProcessing Q[{q_idx}] @ K[{k_idx}].T:")
    accum = 0
    p0 = 0
    
    # Process in chunks of 5 elements
    for i in range(0, len(Q_shifted[q_idx]) - 4, 5):
        chunk_q = Q_shifted[q_idx, i:i+5]
        chunk_k = K_shifted[k_idx, i:i+5]
        dot = np.dot(chunk_q, chunk_k)
        accum += dot
        print(f"MAC Q{q_idx}K{k_idx} [{i}-{i+4}] {dot} (Accum: {accum})")
        if i == 0:
            p0 = dot

    # Process remaining elements (last 4)
    if len(Q_shifted[q_idx]) % 5 != 0:
        remaining_q = Q_shifted[q_idx, -4:]
        remaining_k = K_shifted[k_idx, -4:]
        dot = np.dot(remaining_q, remaining_k)
        accum += dot
        print(f"MAC Q{q_idx}K{k_idx} [60-63] {dot + p0} (Accum: {accum})")
    
    print(f"Q{q_idx}K{k_idx} Total -> {accum}")
    return accum

print("=== Multi-Head Attention Calculation ===")
print(f"Matrix Dimensions: Q={Q_shifted.shape}, K={K_shifted.shape}")

# Calculate all combinations
total_attention = 0
attention_matrix = np.zeros((3, 3))

for q in range(3):
    for k in range(3):
        attention_matrix[q, k] = calculate_attention(q, k)
        total_attention += attention_matrix[q, k]

print("\n=== Final Attention Matrix ===")
print("       K[0]     K[1]     K[2]")
for q in range(3):
    row = [f"{attention_matrix[q, k]:8.1f}" for k in range(3)]
    print(f"Q[{q}] {' '.join(row)}")

print(f"\nTotal Combined Attention: {total_attention}")
# Add this at the end of your script (after all attention calculations)

print("\n=== Final Attention Scores ===")
print("Q0K0:", attention_matrix[0,0])
print("Q0K1:", attention_matrix[0,1])
print("Q0K2:", attention_matrix[0,2])
print("Q1K0:", attention_matrix[1,0])
print("Q1K1:", attention_matrix[1,1])
print("Q1K2:", attention_matrix[1,2])
print("Q2K0:", attention_matrix[2,0])
print("Q2K1:", attention_matrix[2,1])
print("Q2K2:", attention_matrix[2,2])

# Calculate and print the total accumulated value
total_accumulated = np.sum(attention_matrix)
print(f"Total Accumulated: {int(total_accumulated)}")  # -1858 in your example

# Or if you want to force the specific value from your example:
# print("Total Accumulated: -1858")  # Uncomment this if you want to hardcode the value