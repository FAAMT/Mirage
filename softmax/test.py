#!/usr/bin/env python3

values = [
    3200, -25216,   9344, -24192,  5248,  -160,  -9088,  12672,
    -33664,  15744, -35712,      0,  4096,  -8192,   8192, -16384,
    12288, -20480,  16384, -24576, 20480, -28672,  24576, -32768,
    28672, -36864,  32768, -40960, 36864, -45056,  40960, -49152,
    45056, -53248,  49152, -57344, 53248, -61440,  57344, -65536,
    61440, -81920,  65536, -65536, 69632, -81920,  73728, -87040,
    77824, -92160,  81920, -97280, 86016, -102400, 90112, -107520,
    94208, -112640,  98304, -117760,102400, -122880,106496, -128000
]

softmax_0_0 = 8192

def verify_first_softmax():
    s_q8 = softmax_0_0 >> 8
    cumulative_sum = 0
    print("Index | V Value | V>>11 | In |  Product | Running Sum")
    print("----------------------------------------------")
    
    # First calculate up to the last 10 values
    for i, v in enumerate(values[:64]):  # First 64 elements only
        v_q8 = v >> 11
        product = s_q8 * v_q8
        cumulative_sum += product
        if i >= 54:  # Only print the last 10 values
            print(f"{i:5} | {v:7} | {v_q8:5} |  {s_q8} | {product:7} | {cumulative_sum:7}")
    
    print("\nFinal accumulated sum:", cumulative_sum)

verify_first_softmax()