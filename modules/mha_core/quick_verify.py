# Golden Verification for Full Q0 to Q63 MAC Calculation

TILE_W = 5
TILE_H = 12
DATA_W = 8
NUM_TILES = 52

def to_signed8(val):
    if val >= 128:
        return val - 256
    else:
        return val

def generate_tile_inputs(tile_index, tile_h_pass):
    input_a = [((tile_index * 10) + (i * 2) + 3) % 128 for i in range(TILE_W)]

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

    for neuron_base in range(0, 64, 12):
        print(f"\n=== Q{neuron_base} to Q{min(neuron_base+11, 63)} MAC Calculation ===\n")

        accumulators = [0] * 12  # 12 outputs per height slice

        for tile_pass in range(NUM_TILES):
            input_a, input_b = generate_tile_inputs(tile_index, tile_h_pass)

            print(f"Tile {tile_pass}: Input A = {input_a}")

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
                    product = weight * input_a[j]
                    partial_sum += product
                    print(f"  Q{neuron_base + neuron_idx} Weight[{j}] = {weight}, Input_A[{j}] = {input_a[j]}, Product = {product}")

                accumulators[neuron_idx] += partial_sum
                print(f"Tile {tile_pass} Partial Sum for Q{neuron_base + neuron_idx}: {partial_sum}")

            tile_index += 1

        # After 52 tiles
        for idx, acc in enumerate(accumulators):
            if neuron_base + idx < 64:
                print(f"\nFinal Accumulated Q{neuron_base + idx}: {acc} (decimal), {hex(acc)} (hex)")

        tile_index = 0
        tile_h_pass += 12

if __name__ == "__main__":
    simulate_q_projections()