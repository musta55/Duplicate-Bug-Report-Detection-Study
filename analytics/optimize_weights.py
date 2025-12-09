import pandas as pd
import numpy as np
from tqdm import tqdm
import itertools
import sys

def calculate_metrics(df, weights):
    # weights: (w_bb, w_rs, w_sf, w_cf)
    w_bb, w_rs, w_sf, w_cf = weights
    
    # Create mask for presence (1.0 if present, 0.0 if NaN)
    has_bb = df['BB'].notna().astype(float)
    has_rs = df['RS'].notna().astype(float)
    has_sf = df['SF'].notna().astype(float)
    has_cf = df['CF'].notna().astype(float)
    
    # Fill NaNs with 0 for calculation
    bb = df['BB'].fillna(0)
    rs = df['RS'].fillna(0)
    sf = df['SF'].fillna(0)
    cf = df['CF'].fillna(0)
    
    # Calculate Weighted Average Distance
    numerator = (bb * w_bb) + (rs * w_rs) + (sf * w_sf) + (cf * w_cf)
    denominator = (has_bb * w_bb) + (has_rs * w_rs) + (has_sf * w_sf) + (has_cf * w_cf)
    
    # Avoid division by zero
    denominator = denominator.replace(0, 1.0)
    
    # New Score (Distance)
    new_scores = numerator / denominator
    
    # Assign to temp dataframe
    # We use a lightweight approach to avoid copying the whole DF repeatedly if possible
    # But for safety, let's just assign the column
    df['new_score'] = new_scores
    
    # Calculate Metrics
    hits10 = 0
    mrr_sum = 0
    n_queries = 0
    
    # Group by query
    # We assume the DF is sorted by query, but groupby handles it
    grouped = df.groupby('query')
    
    for qid, group in grouped:
        # Sort by new_score (ascending distance)
        # We only need the 'c_is_gt' column sorted by 'new_score'
        sorted_gt = group.sort_values('new_score', ascending=True)['c_is_gt'].values
        
        # Find indices where c_is_gt == 1
        gt_indices = np.where(sorted_gt == 1)[0]
        
        if len(gt_indices) > 0:
            first_rank = gt_indices[0] + 1 # 1-based rank
            
            mrr_sum += 1.0 / first_rank
            if first_rank <= 10:
                hits10 += 1
        
        n_queries += 1
        
    if n_queries == 0: return 0, 0
    
    return mrr_sum / n_queries, hits10 / n_queries

def optimize(csv_path):
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    
    print(f"Loaded {len(df)} rows. Optimizing weights...")
    
    # Define search grid
    # Refined grid based on previous best (4.0, 1.0, 0.5, 0.5)
    bb_vals = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]
    rs_vals = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
    sf_vals = [0.1, 0.3, 0.5, 0.7, 0.9]
    cf_vals = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    combinations = list(itertools.product(bb_vals, rs_vals, sf_vals, cf_vals))
    # Filter out all-zeros
    combinations = [c for c in combinations if sum(c) > 0]
    
    print(f"Testing {len(combinations)} combinations...")
    
    best_hits = -1
    best_mrr = -1
    best_weights = None
    
    # Baseline (Equal weights)
    base_mrr, base_hits = calculate_metrics(df.copy(), (1, 1, 1, 1))
    print(f"Baseline (1,1,1,1) -> MRR: {base_mrr:.4f}, Hits@10: {base_hits:.4f}")
    
    for w in tqdm(combinations):
        # Pass a copy or modify in place? 
        # calculate_metrics modifies 'new_score' column. 
        # It's safer to just overwrite the column in the same DF, 
        # as long as we don't depend on previous 'new_score'.
        mrr, hits = calculate_metrics(df, w)
        
        if hits > best_hits:
            best_hits = hits
            best_mrr = mrr
            best_weights = w
        elif hits == best_hits and mrr > best_mrr:
            best_mrr = mrr
            best_weights = w
            
    print("\n" + "="*30)
    print(f"OPTIMIZATION RESULTS for {csv_path}")
    print("="*30)
    print(f"Best Weights (BB, RS, SF, CF): {best_weights}")
    print(f"Best MRR:     {best_mrr:.4f} (Baseline: {base_mrr:.4f})")
    print(f"Best Hits@10: {best_hits:.4f} (Baseline: {base_hits:.4f})")
    print("="*30)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 optimize_weights.py <csv_file>")
        sys.exit(1)
    
    optimize(sys.argv[1])
