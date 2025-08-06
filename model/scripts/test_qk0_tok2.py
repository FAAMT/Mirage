import numpy as np

Q = np.array([365,  -279,  -930,  1152,  1787,   650,  1666,  4241,  -950,  2142,  2913,  2988,
    -1228,  1141,  -270,  -751,  2230,   395,  1637,  -844, -2862,   408, -1280, -2821,
    -595,   350, -1661, -1127, -1472, -2081, -1049,  -513,  1754, -2819, -3914,  1029,
    -1236,   749,   567,  3165,  1860, -3473, -1380,  1020,  -746, -2392,   759, -2238,
    809,  -270,   512,   788,  3048,  1711,  -716, -2397,  -606,   272,  4016,  2569,
    2069, -1103,  1320, -1747])

K = np.array([635, -1187,  2345,  1330,  2718, -3912,   877,  -509,  5408,  1559,  3950, -3331,
    -3748,  -786,  -434,  1561,  -776, -1131, -1243,  -939,  3210,   809,  1351, -1258,
    -1735,  1529,  1070,  1570, -2905,  1118,  -793,  -223,  1198,  -585,  -873, -2586,
    -1282, -3499,  1166, -2747,   509, -4665, -1673, -1445, -1253, -3105,   916, -3405,
    -332,  1897,   258,  1904,  -498, -2097,  -335, -1944,   392,  1399,  2364,  3440,
    -4405, -2422,  3157,  -789])

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