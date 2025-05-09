# Original input data
original_data = [
    3200, -25216, 9344, -24192, 5248, -160, -9088, 12672,
    -33664, 15744, -35712, 0, 4096, -8192, 8192, -16384,
    12288, -20480, 16384, -24576, 20480, -28672, 24576, -32768,
    28672, -36864, 32768, -40960, 36864, -45056, 40960, -49152,
    45056, -53248, 49152, -57344, 53248, -61440, 57344, -65536,
    61440, -81920, 65536, -65536, 69632, -81920, 73728, -87040,
    77824, -92160, 81920, -97280, 86016, -102400, 90112, -107520,
    94208, -112640, 98304, -117760, 102400, -122880, 106496, -128000,
]

# Quantization scale
input_scale = 4
weight_scale = 8

# Quantize and clip
def clip_to_int8(x):
    return max(-128, min(127, x))

input_a = [clip_to_int8((x // input_scale) >> 8) for x in original_data]
input_b = [clip_to_int8((x // weight_scale) >> 8) for x in original_data]

input_a_blocks = []
input_b_blocks = []

for proj_iter in range(4):
    a_block = []
    b_block = []
    for x in original_data:
        q_val = clip_to_int8(((x // input_scale) + 1000 * proj_iter) >> 8)
        k_val = clip_to_int8(((x // weight_scale) + 100 * proj_iter) >> 8)
        a_block.append(q_val)
        b_block.append(k_val)
    input_a_blocks.append(a_block)
    input_b_blocks.append(b_block)

    for x in original_data:
        q_val = clip_to_int8(((x // input_scale) + 1000 * proj_iter) >> 8)
        k_val = clip_to_int8(((x // weight_scale) + 100 * proj_iter) >> 8)
        input_a.append(q_val)
        input_b.append(k_val)

# Constants
macs_per_row = 5
rows_total = 12  # 4 passes × 3 rows = 12
rows_per_pass = 3
num_outputs = 48

# Compute all 12 output dot products
print("Final Output Dot Products (Q0 × Kᵀ):\n")
final_outputs = [0] * num_outputs

for q_proj in range(4):
    for mat_idx in range(4):  # 12 scores per Q
        for local_output in range(3):
            output_idx = q_proj * 12 + mat_idx * 3 + local_output
            row_idx = local_output

            dot_sum = 0
            print(f"▶ Q{q_proj} Output {output_idx}:")

            for pass_id in range(4):
                mac_row = row_idx
                global_row = pass_id * 3 + mac_row
                row_dot = 0

                for i in range(macs_per_row):
                    idx = global_row * macs_per_row + i
                    a = input_a_blocks[q_proj][idx]
                    b = input_b_blocks[mat_idx][idx]
                    prod = a * b
                    row_dot += prod
                    print(f"  Pass {pass_id}, idx {idx}: a={a}, b={b}, product={prod}")

                print(f"  → Dot from Pass {pass_id}, Row {global_row}: {row_dot}")
                dot_sum += row_dot

            final_outputs[output_idx] = dot_sum
            print(f"✅ Output[{output_idx}] = {dot_sum}\n")

print("\n==============================")
print("Summed Output per Matrix Mult (3 rows each):\n")

for q_proj in range(4):  # Q₀ to Q₃
    print(f"\nQ{q_proj} Scores:")
    for mat_idx in range(4):  # 12 scores per Q, grouped by 3
        base = q_proj * 12 + mat_idx * 3
        partials = final_outputs[base:base + 3]
        mat_sum = sum(partials)
        print(f"  MatrixMult {mat_idx}: Outputs[{base}-{base+2}] = {partials} → Sum = {mat_sum}")


print("Done.")

