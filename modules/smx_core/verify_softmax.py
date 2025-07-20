#!/usr/bin/env python3
import numpy as np

bram_scores = np.array([
    90309, 92904, 94026, 95644,
    97897, 101568,103297, 105943,
    107576, 103228, 10291, 13653, 
    106156, 111153, 114632, 118448
], dtype=np.int32)

def compute_taylor_q345(scores):
    table = []
    exp_scores = np.zeros(len(scores), dtype=np.int64)
    
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
            taylor = 2**45-(1 << 45) + term1 + term2 + term3
            taylor_group.append(taylor)

            table.append([
                f"Q{block}", i, score, z, z_q15,
                term1, term2, term3, taylor
            ])
            
            # Store for softmax calculation
            idx = 4 * block + i
            exp_scores[idx] = taylor

        # Compute Taylor sum for this group
        taylor_sum = sum(taylor_group)
        for i in range(4):
            softmax_q015 = (taylor_group[i] << 15) // taylor_sum
            table[4 * block + i].append(taylor_sum)
            table[4 * block + i].append(softmax_q015)

    return table, exp_scores

def taylor3(scores) -> tuple:
    """
    Enhanced Taylor approximation with proper Q-sum handling
    Returns: (exp_scores, softmax_fp, q_sums)
    """
    num_scores = len(scores)
    exp_scores = np.zeros(num_scores, dtype=np.int64)
    q_sums = np.zeros(4, dtype=np.int64)  # One sum per Q group
    q_softmax = np.zeros((4, 4), dtype=np.float64)  # 4 Q-groups × 4 tokens
    taylor_group = []

    
    # Process in groups of 4 (Q0-Q3)
    for block in range(4):
        start_idx = 4 * block
        end_idx = start_idx + 4
        group = scores[start_idx:end_idx]
        group_max = max(group)
        group_exp = np.zeros(4, dtype=np.int64)

        for i, score in enumerate(group):
            z = score - group_max
            z_q15 = z >> 8  # Q7.8 format
            z_sq = z_q15 * z_q15
            z_cu = z_sq * z_q15

            # Taylor approximation terms
            term1 = z_q15 << 30  # Q37.8
            term2 = (z_sq >> 1) << 15  # Q22.16
            term3 = (z_cu * 21846) >> 30  # 21846 ≈ 1/3 in Q15
            
            # Store approximation and accumulate group sum
            exp_val = 2**45-(1 << 45) + term1 + term2 + term3
            taylor_group.append(exp_val)
            group_exp[i] = exp_val
            exp_scores[start_idx + i] = exp_val
        
        # Store Q-group sum
        q_sums[block] = np.sum(group_exp)
        
        q_softmax[block] = (group_exp << 14) / q_sums[block] / (1 << 14)

    # Compute softmax probabilities
    taylor_sum = sum(taylor_group)
    softmax_q14 = (exp_scores << 14) // taylor_sum
    softmax_fp = softmax_q14.astype(float) / (1 << 14)
    return exp_scores, softmax_fp, q_sums, q_softmax


# Usage example:
# Generate both the detailed table and exp scores
table, exp_scores = compute_taylor_q345(bram_scores)

# Print detailed table
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

# Calculate softmax using taylor3 function
exp_scores, softmax_fp, q_sums, q_softmax = taylor3(bram_scores)
# Print results
# Print Token 1 probabilities for each Q group
print("\nToken 1 Softmax Probabilities per Q Group:")
for q_idx in range(4):
    print(f"Q{q_idx}: {q_softmax[q_idx][1]:.6f}")  # Index 1 is Token 1 in each Q group

# Print all Q-group softmax tuples
print("\nFull Q-Group Softmax Probabilities:")
for q_idx in range(4):
    print(f"Q{q_idx}: {tuple(f'{prob:.6f}' for prob in q_softmax[q_idx])}")