import numpy as np

Q = np.array([-4886,  -642,  -709,  -498,  2563, -2714,  -601, 1097, -6474,   575,  4192,
    1502,  5474, -3098,  1595, -1891, -2593,  2482, -4065,  -623,  8110,    81,
   -1803,  2877,  1051,  4409,  -777, -5716,   439,  1464,  1468,   383, -2530,
    -380, -1876, -5295,  1758,  5776,   -23,  -204, -7419,  4902, -2060, -5354,
    -540,  -516,  3970,   752,  -510,  -955,   483,  5229,   484,  1464,  1206,
     926,  5961,  1557, -3357,  2820,  -712, -2413,  3191,  5])

K = np.array([
    3238,  1561,  -815,   950, -1954, -2690,   270,  2192,  3944, -1372, -2595,
     -49,  2334,   178, -7085,  2353,  1072,   704,  1778, -1742,  1468, -4031,
   -1914,  -112,  2916, -1162,  3286,  3552,  -974,  2241,  2778, -4627,   516,
    4812, -5804,  2756, -2945, -2568, -1778,   307,  6547,  1636,     7,  2280,
    7304, -2475,  5555,  4129, -1280,  4716,  -343,   720, -4579, -5722,   759,
   -4921,  -374,  -211, -2683,  6002,  1561,  4096,   882,  2332])

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