import pandas as pd
import numpy as np
import sys
import os

def apply_weights(csv_path, weights):
    print(f"Processing {csv_path} with weights {weights}...")
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return None

    df = pd.read_csv(csv_path)
    
    w_bb, w_rs, w_sf, w_cf = weights
    
    # Ensure columns are numeric
    for col in ['BB', 'RS', 'SF', 'CF']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            print(f"Warning: Column {col} missing in {csv_path}, assuming 0.")
            df[col] = np.nan
    
    has_bb = df['BB'].notna().astype(float)
    has_rs = df['RS'].notna().astype(float)
    has_sf = df['SF'].notna().astype(float)
    has_cf = df['CF'].notna().astype(float)
    
    bb = df['BB'].fillna(0)
    rs = df['RS'].fillna(0)
    sf = df['SF'].fillna(0)
    cf = df['CF'].fillna(0)
    
    numerator = (bb * w_bb) + (rs * w_rs) + (sf * w_sf) + (cf * w_cf)
    denominator = (has_bb * w_bb) + (has_rs * w_rs) + (has_sf * w_sf) + (has_cf * w_cf)
    
    # Avoid division by zero
    denominator = denominator.replace(0, 1.0)
    
    new_scores = numerator / denominator
    
    df['score'] = new_scores
    
    # Re-rank
    print("Re-ranking...")
    
    # Sort by query and score (ascending distance)
    df = df.sort_values(['query', 'score'], ascending=[True, True])
    
    # Assign new ranks
    df['rank'] = df.groupby('query').cumcount() + 1
    
    # Save updated CSV
    output_path = csv_path.replace('.csv', '_weighted.csv')
    df.to_csv(output_path, index=False)
    print(f"Saved weighted results to {output_path}")
    
    return df

def calculate_metrics(df):
    print("Calculating metrics...")
    
    reciprocal_ranks = []
    average_precisions = []
    hits_at_k = {1: 0, 5: 0, 10: 0}
    
    grouped = df.groupby('query')
    n_queries = 0
    
    for qid, group in grouped:
        # Check if there is any ground truth in this group
        if group['c_is_gt'].sum() == 0:
            continue
            
        n_queries += 1
        
        # Get the ranks of ground truth items
        gt_ranks = group[group['c_is_gt'] == 1]['rank'].values
        
        # MRR (1 / first_rank)
        if len(gt_ranks) > 0:
            first_rank = gt_ranks.min()
            reciprocal_ranks.append(1.0 / first_rank)
        else:
            reciprocal_ranks.append(0.0)
            
        # MAP
        precisions = []
        for i, rank in enumerate(sorted(gt_ranks)):
            precisions.append((i + 1) / rank)
            
        if precisions:
            average_precisions.append(np.mean(precisions))
        else:
            average_precisions.append(0.0)
            
        # Hits@K
        min_rank = gt_ranks.min() if len(gt_ranks) > 0 else float('inf')
        
        for k in hits_at_k:
            if min_rank <= k:
                hits_at_k[k] += 1
                
    mrr = np.mean(reciprocal_ranks) if reciprocal_ranks else 0
    map_score = np.mean(average_precisions) if average_precisions else 0
    
    print(f"Evaluated on {n_queries} queries with ground truth.")
    print(f"MRR: {mrr:.4f}")
    print(f"MAP: {map_score:.4f}")
    for k in sorted(hits_at_k.keys()):
        print(f"Hits@{k}: {hits_at_k[k]/n_queries:.4f}")

if __name__ == "__main__":
    # Weights: BB, RS, SF, CF
    weights = (8.0, 0.5, 0.7, 0.3)
    
    files = [
        'semcluster_similarity_matrix_FILTERED.csv',
        'semcluster_similarity_matrix_FULL.csv'
    ]
    
    for f in files:
        try:
            df = apply_weights(f, weights)
            if df is not None:
                calculate_metrics(df)
            print("-" * 40)
        except Exception as e:
            print(f"Error processing {f}: {e}")
