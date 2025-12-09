import pandas as pd
import numpy as np

def calculate_metrics(df):
    # Ensure sorted by rank
    df = df.sort_values(['query', 'rank'])
    
    mrr_sum = 0
    hits1_count = 0
    hits5_count = 0
    hits10_count = 0
    ap_sum = 0
    
    queries = df['query'].unique()
    n_queries = len(queries)
    
    for q in queries:
        q_df = df[df['query'] == q]
        
        # Ground truth items
        gts = q_df[q_df['c_is_gt'] == 1]
        
        if len(gts) == 0:
            # Should not happen if dataset is correct, but handle it
            continue
            
        # Ranks of ground truth items (1-based)
        gt_ranks = gts['rank'].values
        
        # MRR: 1 / min_rank
        if len(gt_ranks) > 0:
            mrr_sum += 1.0 / np.min(gt_ranks)
            
        # HITS@k
        if np.any(gt_ranks <= 1):
            hits1_count += 1
        if np.any(gt_ranks <= 5):
            hits5_count += 1
        if np.any(gt_ranks <= 10):
            hits10_count += 1
            
        # MAP (Average Precision)
        # AP = sum(P@k * rel(k)) / number_of_relevant_items
        # Since we have ranks, we can iterate through them
        relevant_ranks = sorted(gt_ranks)
        p_sum = 0
        for i, r in enumerate(relevant_ranks):
            # Precision at rank r is (i+1) / r
            p_sum += (i + 1) / r
        
        if len(relevant_ranks) > 0:
            ap_sum += p_sum / len(relevant_ranks)
            
    return {
        'MRR': mrr_sum / n_queries,
        'MAP': ap_sum / n_queries,
        'HITS@1': hits1_count / n_queries,
        'HITS@5': hits5_count / n_queries,
        'HITS@10': hits10_count / n_queries
    }

print("Calculating metrics for FILTERED dataset...")
df_filtered = pd.read_csv('semcluster_similarity_matrix_FILTERED_weighted.csv')
metrics_filtered = calculate_metrics(df_filtered)
print("FILTERED Metrics:", metrics_filtered)

print("\nCalculating metrics for FULL dataset...")
df_full = pd.read_csv('semcluster_similarity_matrix_FULL_weighted.csv')
metrics_full = calculate_metrics(df_full)
print("FULL Metrics:", metrics_full)
