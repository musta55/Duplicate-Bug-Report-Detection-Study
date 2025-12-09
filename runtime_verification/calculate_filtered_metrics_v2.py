import pandas as pd
import numpy as np

def calculate_metrics_from_result_csv(csv_path):
    print(f"Loading results from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Group by query
    queries = df['query'].unique()
    print(f"Found {len(queries)} unique queries in results.")
    
    reciprocal_ranks = []
    average_precisions = []
    hits_at_k = {k: 0 for k in range(1, 11)}
    
    for qid in queries:
        # Get rows for this query, sorted by rank
        q_df = df[df['query'] == qid].sort_values('rank')
        
        # Extract ground truth indicators
        # c_is_gt is 1 if it's a duplicate, 0 otherwise
        gt_indicators = q_df['c_is_gt'].tolist()
        
        # --- MRR ---
        first_correct_rank = -1
        for i, is_gt in enumerate(gt_indicators):
            if is_gt == 1:
                first_correct_rank = i + 1
                break
        
        if first_correct_rank != -1:
            reciprocal_ranks.append(1.0 / first_correct_rank)
        else:
            reciprocal_ranks.append(0.0)
            
        # --- MAP ---
        precision_scores = []
        correct_hits = 0
        for i, is_gt in enumerate(gt_indicators):
            if is_gt == 1:
                correct_hits += 1
                precision_scores.append(correct_hits / (i + 1))
        
        if precision_scores:
            average_precisions.append(np.mean(precision_scores))
        else:
            average_precisions.append(0.0)
            
        # --- HITS@k ---
        for k in range(1, 11):
            # Check if any of the top k are GT
            if any(gt_indicators[:k]): # 1 is truthy
                hits_at_k[k] += 1

    # Calculate averages
    mrr = np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0
    map_score = np.mean(average_precisions) if average_precisions else 0.0
    
    # HITS scores are normalized by number of queries
    n_queries = len(queries)
    hits_scores = {k: (v / n_queries) if n_queries > 0 else 0.0 for k, v in hits_at_k.items()}
    
    print("\n" + "="*70)
    print("RETRIEVAL RESULTS (FILTERED)")
    print("="*70)
    print(f"Evaluated Queries: {n_queries}")
    print(f"MRR: {mrr:.4f}")
    print(f"MAP: {map_score:.4f}")
    print("HITS@k:")
    for k in sorted(hits_scores.keys()):
        print(f"  HITS@{k:2d}: {hits_scores[k]:.4f}")

if __name__ == "__main__":
    calculate_metrics_from_result_csv('output/semcluster_similarity_matrix_FILTERED_NORMALIZED.csv')
