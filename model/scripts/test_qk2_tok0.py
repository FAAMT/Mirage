import numpy as np

Q = np.array( [3248,  3143,  -702,  -407,  -769,  7042,  2985, -2970,  -423, -3511, -2583,
    -1292,   810, -5507,   793, -2350,   122,  3306, -5969,  3251, -2627,  2250,
    -3383,  -656, -2059, -4101,  -628,   331,  5230, -1076,   971, -1722,  2039,
     2776, -2471,  -301,  1312,  6607, -2914, -3711,    49,  -271, -5132, -1913,
     3036,  -199,  2962, -1848,   794, -1604,  5208, -4379,  1326,  1860,   545,
    -2558, -3852,  -698, -2408,  2853, -1778,  -775, -1489,   511])

K = np.array( [3185,  -7931,   1736,   3641,  -3812,   4062,   1440,  -1812,   2604,     23,
     -3842,   1928,   3112,  -3766,  -3511,  -2033,   1998,   3880,   1409,   2158,
     -2359,   -880,    775,   6451,  -7129,    735,  -1390,   5126,  -4213,   6741,
       159,   9906,   6419,  -3266,   1150,   -656,  -4521,  -1369,   8754,  -7456,
     -3094,   -944,   2813,  -8540,   -253,   1434,  -8206,  -1462,  -2869,    161,
     -7606,  -1694,  -1292,   2529,   2654,  -1192,  -1609,  -2615,  -6880,   3795,
      -311,   1865,   -927,   3357])

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