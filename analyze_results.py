import pandas as pd
import numpy as np

def analyze_results(csv_path):
    print(f"Analyzing {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Basic counts
    n_rows = len(df)
    n_queries = df['query'].nunique()
    n_projects = df['Project'].nunique()
    
    print(f"Total rows: {n_rows}")
    print(f"Unique queries: {n_queries}")
    print(f"Unique projects: {n_projects}")
    print(f"Projects: {df['Project'].unique()}")
    
    # Check for duplicates
    duplicates = df.duplicated(subset=['query', 'corpus'])
    n_duplicates = duplicates.sum()
    print(f"Duplicate query-corpus pairs: {n_duplicates}")
    
    if n_duplicates > 0:
        print("Dropping duplicates for analysis...")
        df = df.drop_duplicates(subset=['query', 'corpus'])
        
        # Re-calculate ranks after dropping duplicates
        print("Re-calculating ranks...")
        df['rank'] = df.groupby('query')['score'].rank(method='first', ascending=True)
    
    # Score statistics
    print("\nScore Statistics:")
    print(df['score'].describe())
    
    # Ground Truth Analysis
    gt_rows = df[df['c_is_gt'] == 1]
    print(f"\nGround Truth Matches Found: {len(gt_rows)}")
    
    if len(gt_rows) > 0:
        print("Rank statistics for Ground Truth matches:")
        print(gt_rows['rank'].describe())
        
        # Hits at K
        hits_1 = len(gt_rows[gt_rows['rank'] <= 1])
        hits_5 = len(gt_rows[gt_rows['rank'] <= 5])
        hits_10 = len(gt_rows[gt_rows['rank'] <= 10])
        
        # Note: This is raw count of found GTs, not per-query metrics yet
        print(f"GT matches at Rank 1: {hits_1}")
        print(f"GT matches in Top 5: {hits_5}")
        print(f"GT matches in Top 10: {hits_10}")
        
        # Per-query analysis
        # For each query, what is the best rank of a GT item?
        best_ranks = gt_rows.groupby('query')['rank'].min()
        
        q_hits_1 = (best_ranks <= 1).sum()
        q_hits_5 = (best_ranks <= 5).sum()
        q_hits_10 = (best_ranks <= 10).sum()
        
        print(f"\nQueries with at least one GT match in Top 1 (HITS@1): {q_hits_1}/{n_queries} ({q_hits_1/n_queries:.4f})")
        print(f"Queries with at least one GT match in Top 5 (HITS@5): {q_hits_5}/{n_queries} ({q_hits_5/n_queries:.4f})")
        print(f"Queries with at least one GT match in Top 10 (HITS@10): {q_hits_10}/{n_queries} ({q_hits_10/n_queries:.4f})")

    else:
        print("No ground truth matches found in the results.")

    # Per Project Breakdown
    print("\nPer-Project Performance (HITS@10):")
    for project, group in df.groupby('Project'):
        p_queries = group['query'].nunique()
        p_gt = group[group['c_is_gt'] == 1]
        if len(p_gt) > 0:
            p_best_ranks = p_gt.groupby('query')['rank'].min()
            p_hits_10 = (p_best_ranks <= 10).sum()
            print(f"  {project}: {p_hits_10}/{p_queries} ({p_hits_10/p_queries:.4f})")
        else:
            print(f"  {project}: 0/{p_queries} (0.0000)")

if __name__ == "__main__":
    analyze_results("semcluster_similarity_matrix_FILTERED.csv")
