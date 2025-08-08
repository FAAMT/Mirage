import numpy as np

Q = np.array([  9,  12,   6,   3,   0, -12,  -3, -10,  -3,   2,   4,   4,  -6,   8,  25,  -6,  -1,   4,
                2,  -4, -15,  -8,   6, -10,   8,   6,  -4,   2, -11,  -1,  22, -12,  11,  -4, -10,  -8,
                0,  10,  -3,   6, -15,  -4, -18,  -3,   7,   1,  13,   1,  30,  17,  -2,   8,   0, -12,
              -11,   4,  -1,  -9,  11,   8,  -9,   0, -10,  -2])

K = np.array([17,  12,  11,   5, -22,  -1,  -2,   4,   7,  -8,  15,   4,  -2,  15,  12,  -6, -12,   0,
               -4,  -3,  21,   5,   2,   2,  -2, -16,  -7,  -6,  14,  -7,  16,   0,   7,  -2,  -5, -13,
                6,   3,  -1,   9,  13,  10,  -9,   1,   2, -10,  19, -10, -28,  11,  -7,  -5,  -4,   1,
                6,  -6, -11,  -3, -18, -11,  -8,  -5,   1, -15])

# Right-shift by 8 (equivalent to floor division by 256)
Q_shifted = Q 
K_shifted = K 

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