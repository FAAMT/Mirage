# Golden Verification for Q, K, V Projections

TILE_W = 5
TILE_H = 12
NUM_TILES = 52
PROJECTIONS = ["Q", "K", "V"]

def to_signed8(val):
    return val - 256 if val >= 128 else val

# Token vector logic (matches TB)
static_token_vector = [(i * 2 + 3) % 128 for i in range(256)]

def generate_tile_inputs(tile_index):
    base_idx = tile_index * TILE_W
    input_a = [(static_token_vector[base_idx + j] if base_idx + j < 256 else 0) for j in range(TILE_W)]
    return input_a

def generate_weights(proj_type, tile_index, tile_h_pass):
    weights = []
    offset_mul = {"Q": 1, "K": 3, "V": 5}[proj_type]
    for i in range(TILE_H):
        for j in range(TILE_W):
            addr = (tile_h_pass + i) * 256 + tile_index * TILE_W + j
            val = ((addr * offset_mul % 256) * 59 + (addr // 256) * 37 + 13) % 256
            weights.append(to_signed8(val))
    return weights

def simulate_projection(proj_type):
    tile_h_pass = 0
    tile_index = 0
    result = [0] * 64

    for neuron_base in range(0, 64, 12):
        accumulators = [0] * 12

        for tile_pass in range(NUM_TILES):
            input_a = generate_tile_inputs(tile_index)
            input_b = generate_weights(proj_type, tile_index, tile_h_pass)

            for neuron_idx in range(12):
                if neuron_base + neuron_idx >= 64:
                    break
                psum = 0
                valid_elements = TILE_W if tile_pass < NUM_TILES - 1 else 256 - TILE_W * (NUM_TILES - 1)
                for j in range(valid_elements):
                    psum += input_b[neuron_idx * TILE_W + j] * input_a[j]
                accumulators[neuron_idx] += psum

            tile_index += 1

        for i, acc in enumerate(accumulators):
            if neuron_base + i < 64:
                result[neuron_base + i] = acc

        tile_index = 0
        tile_h_pass += 12

    return result

def load_vhdl_output(filename="q_output.txt"):
    with open(filename, "r") as f:
        values = [int(line.strip()) for line in f.readlines()]
    return {
        "Q": values[0:64],
        "K": values[64+8:128+8],
        "V": values[128+16:192+16]
    }

def compare(golden, vhdl, label):
    print(f"\n=== Comparing {label} Projection ===")
    print(f"{'Idx':>4} {'Golden':>12} {'VHDL':>12} {'Diff':>10}")
    print("-" * 45)
    errors = 0
    for i, (g, v) in enumerate(zip(golden, vhdl)):
        diff = v - g
        mark = " <-- MISMATCH" if diff != 0 else ""
        print(f"{i:4d} {g:12d} {v:12d} {diff:10d}{mark}")
        if diff != 0:
            errors += 1
    print("-" * 45)
    print("✅ Match!" if errors == 0 else f"❌ {errors} mismatches")

def main():
    vhdl_outputs = load_vhdl_output()
    for proj in PROJECTIONS:
        golden = simulate_projection(proj)
        compare(golden, vhdl_outputs[proj], proj)

if __name__ == "__main__":
    main()
