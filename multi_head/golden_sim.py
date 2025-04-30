# Final Golden Verification for Q0 to Q63 Projection

TILE_W = 5
TILE_H = 12
DATA_WIDTH = 8
NUM_TILES = 52

def to_signed8(val):
    if val >= 128:
        return val - 256
    else:
        return val

# Static 256-entry token vector
static_token_vector = [(i * 2 + 3) % 128 for i in range(256)]

def generate_tile_inputs(tile_index, tile_h_pass):
    base_idx = tile_index * TILE_W

    input_a = []
    for j in range(TILE_W):
        if base_idx + j < 256:
            input_a.append(static_token_vector[base_idx + j])
        else:
            input_a.append(0)  # If we run out of valid tokens, fill with zero (just like in VHDL last tile)

    # Weights generation (same)
    input_b = []
    for i in range(TILE_H):
        for j in range(TILE_W):
            addr = (tile_h_pass + i) * 256 + tile_index * TILE_W + j
            val = ((addr % 256) * 59 + (addr // 256) * 37 + 13) % 256
            input_b.append(to_signed8(val))

    return input_a, input_b



def simulate_q_projections():
    tile_h_pass = 0
    tile_index = 0

    golden_results = [0] * 64

    for neuron_base in range(0, 64, 12):
        accumulators = [0] * 12  # 12 outputs per height slice

        for tile_pass in range(NUM_TILES):
            input_a, input_b = generate_tile_inputs(tile_index, tile_h_pass)

            for neuron_idx in range(12):
                if neuron_base + neuron_idx >= 64:
                    break

                partial_sum = 0
                if tile_pass < NUM_TILES - 1:
                    valid_elements = TILE_W
                else:
                    valid_elements = 256 - TILE_W * (NUM_TILES - 1)

                for j in range(valid_elements):
                    weight = input_b[neuron_idx * TILE_W + j]
                    partial_sum += weight * input_a[j]

                accumulators[neuron_idx] += partial_sum

            tile_index += 1

        for idx, acc in enumerate(accumulators):
            if neuron_base + idx < 64:
                golden_results[neuron_base + idx] = acc

        tile_index = 0
        tile_h_pass += 12

    return golden_results

def load_vhdl_output(filename="q_output.txt"):
    with open(filename, "r") as f:
        lines = f.readlines()
    return [int(line.strip()) for line in lines]

def compare_outputs(golden, vhdl_out):
    print(f"{'Neuron':>6} {'Golden':>12} {'VHDL':>12} {'Diff':>10}")
    print("-" * 50)
    mismatch = 0
    for i in range(len(golden)):
        diff = vhdl_out[i] - golden[i]
        marker = " <-- MISMATCH" if diff != 0 else ""
        print(f"{i:6d} {golden[i]:12d} {vhdl_out[i]:12d} {diff:10d}{marker}")
        if diff != 0:
            mismatch += 1
    print("-" * 50)
    if mismatch == 0:
        print("✅ All outputs match perfectly!")
    else:
        print(f"❌ {mismatch} mismatches found!")

def main():
    golden_q = simulate_q_projections()
    vhdl_q = load_vhdl_output()

    compare_outputs(golden_q, vhdl_q)

if __name__ == "__main__":
    main()
