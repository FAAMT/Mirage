#!/usr/bin/env python3
# verify_attention.py

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

# Replicate 4× to make a flat 256-entry V matrix
v_matrix = values * 4

softmax_q015 = [
    8192,  8192, 8191, 8191,
    8191,  8192, 8191, 8191,
    8191,  8191, 8191, 8192,
    8191,  8191, 8191, 8192
]

def per_score_vs_V(softmax, V):
    # Initialize 4x4 output matrix
    dot_products = [[0]*4 for _ in range(4)]
    
    for block in range(4):  # Process all 4 blocks
        blk_scores = softmax[4*block : 4*block+4]
        
        print(f"\n=== Processing block {block} (softmax elements {4*block} to {4*block+3}) ===")
        
        for s_idx, s in enumerate(blk_scores):
            s_q8 = s >> 8  # shift right by 8
            row = []
            
            # Process V in 64-element chunks
            for v in V[:64]:  # Only process first 64 elements of V
                v_q8 = v >> 11
                row.append(s_q8 * v_q8)
            
            # Calculate cumulative sums every 5 elements
            cumulative_sums = []
            current_sum = 0
            for i in range(0, len(row), 5):
                chunk = row[i:i+5]
                current_sum += sum(chunk)
                cumulative_sums.append(current_sum)
            
            # Store the final dot product (sum of all 64 elements)
            total_sum = sum(row)
            dot_products[block][s_idx] = total_sum
            
            # Print results in compact format
            print(f"\nsoftmax_{block}_{s_idx} (value: {s})")
            print("Cumulative sums:", " | ".join(f"{sum_val:8}" for sum_val in cumulative_sums))
            print(f"Final sum: {total_sum}")
    
    # Return the 4x4 dot product matrix
    return dot_products

# Run the function
dot_product_matrix = per_score_vs_V(softmax_q015, v_matrix)

# Print the final 4x4 matrix
print("\n=== Final 4x4 Dot Product Matrix ===")
for i, row in enumerate(dot_product_matrix):
    print(f"Block {i}:", " ".join(f"{val:8}" for val in row))