bram_scores = [
    97162, 97151, 96691, 96826,
    96594, 96595, 96171, 96453,
    95931, 95944, 95554, 96028,
    95235, 95260, 94910, 95576
]

def compute_taylor_q345(scores):
    table = []
    for block in range(4):
        group = scores[4 * block: 4 * block + 4]
        max_val = max(group)
        taylor_group = []

        for i, score in enumerate(group):
            z = score - max_val
            z_q15 = z >> 8
            z_sq = z_q15 * z_q15
            z_cu = z_sq * z_q15

            term1 = z_q15 << 30
            term2 = (z_sq >> 1) << 15
            term3 = (z_cu * 21846) >> 30
            taylor = (1 << 45) + term1 + term2 + term3
            taylor_group.append(taylor)

            table.append([
                f"Q{block}", i, score, z, z_q15,
                term1, term2, term3, taylor
            ])

        # Compute Taylor sum for this group
        taylor_sum = sum(taylor_group)
        for i in range(4):
            softmax_q015 = (taylor_group[i] << 15) // taylor_sum
            table[4 * block + i].append(taylor_sum)
            table[4 * block + i].append(softmax_q015)

    return table

table = compute_taylor_q345(bram_scores)

header = (
    f"{'Matrix':<6} {'Idx':<4} {'Score':<7} {'Z':<6} {'Z_Q15':<7} "
    f"{'Term1':<15} {'Term2':<6} {'Term3':<6} {'Taylor_Approx (Q3.45)':<20} "
    f"{'Taylor_Sum':<20} {'Softmax_Q0.15'}"
)
print(header)
print("-" * len(header))

for row in table:
    print(f"{row[0]:<6} {row[1]:<4} {row[2]:<7} {row[3]:<6} {row[4]:<7} "
          f"{row[5]:<15} {row[6]:<6} {row[7]:<6} {row[8]:<20} "
          f"{row[9]:<20} {row[10]}")
