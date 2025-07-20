bram_scores = [ 90309, 92904, 94026, 95644,
               97897, 101568, 103297, 105943,
               103228, 107576, 110291, 113653,
               106156, 111153, 114637, 118448]

def compute_taylor_q345(scores):
    table = []

    print("\n========== Starting Taylor Softmax Computation ==========\n")
    for block in range(4):
        start_idx = 4 * block
        group = scores[start_idx: start_idx + 4]
        max_val = max(group)
        taylor_group = []

        print(f"\n--- Q{block} ---")
        print(f"Scores: {group}")
        print(f"Max score in block: {max_val}")

        for i, score in enumerate(group):
            z = score - max_val
            z_q15 = z
            z_sq = int(z_q15 * z_q15)
            z_cu = int(z_sq * z_q15)

            term1 = z_q15 << 30
            term2 = int(z_sq / 2) << 15
            term3 = ((z_cu * 21846)) 
            taylor = 1 + term1 + term2 + term3

            print(f"\nToken {i}:")
            print(f"  Raw Score     = {score}")
            print(f"  z = score - max = {z}")
            print(f"  z_q15 (>>8)    = {z_q15}")
            print(f"  z^2            = {z_sq}")
            print(f"  z^3            = {z_cu}")
            print(f"  Term1 (z<<30)  = {term1}")
            print(f"  Term2          = {term2}")
            print(f"  Term3 (approx) = {term3}")
            print(f"  Taylor Approx  = {taylor}")

            taylor_group.append(taylor)

            table.append([
                f"Q{block}", i, score, z, z_q15,
                term1, term2, term3, taylor
            ])

        taylor_sum = sum(taylor_group)
        print(f"\nTotal Taylor Sum: {taylor_sum}")
        
        softmax_fp = []
        for i in range(4):
            softmax_q015 = (taylor_group[i] / taylor_sum)  # Convert to Q0.15
            softmax_float = 1 - softmax_q015  # Convert to float

            print(f"Softmax[{i}] Q0.15  = {int(softmax_q015)}")
            print(f"Softmax[{i}] Float  = {softmax_float:.6f}")

            table[4 * block + i].append(taylor_sum)
            table[4 * block + i].append(int(softmax_q015))
            softmax_fp.append(round(softmax_float, 6))

        print(f"\nFinal Softmax Vector (Float): Q{block} → {softmax_fp}")
        print("-" * 60)

    print("\n========== Computation Complete ==========\n")
    return table

compute_taylor_q345(bram_scores)