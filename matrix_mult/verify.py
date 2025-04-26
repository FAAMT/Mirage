TILE_W = 5
TILE_H = 12
NUM_TILES = 52

def generate_tile_inputs(tile_index):
    # Generate input_a (token embedding slice)
    input_a = [(j * 2 + tile_index * 7 + 3) % 128 for j in range(TILE_W)]
    # Generate input_b (12x5 weights per tile)
    input_b = [(i * 59 + tile_index * 17 + 13) % 128 for i in range(TILE_H * TILE_W)]
    return input_a, input_b

def main():
    q_totals = [0] * TILE_H  # Q[0] through Q[11]

    for t in range(NUM_TILES):
        input_a, input_b = generate_tile_inputs(t)
        for i in range(TILE_H):
            q_tile = sum(input_b[i * TILE_W + j] * input_a[j] for j in range(TILE_W))
            q_totals[i] += q_tile

    print("Final accumulated Q[0–11] values after 52 tiles:")
    for i, q in enumerate(q_totals):
        print(f"Q[{i:2d}] = {q:6d} (hex: {hex(q)})")

if __name__ == "__main__":
    main()
