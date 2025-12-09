import pandas as pd
import numpy as np

def calculate_breakdown(result_csv, gt_csv):
    print(f"Analyzing breakdown for {result_csv}...")
    
    # Load Results
    df = pd.read_csv(result_csv)
    
    # Load GT to check for images
    gt = pd.read_csv(gt_csv)
    # Create map: query_id -> has_image (bool)
    # Ensure query_id is int or string matching df
    gt['query'] = gt['query'].astype(str)
    df['query'] = df['query'].astype(str)
    
    # Map query to image status
    # We need to handle potential duplicates in GT csv (one row per query usually)
    query_image_map = dict(zip(gt['query'], gt['query_has_image']))
    
    # Add has_image column to df
    df['has_image'] = df['query'].map(query_image_map)
    
    # Split
    df_img = df[df['has_image'] == True]
    df_no_img = df[df['has_image'] == False]
    
    def get_metrics(sub_df, name):
        if sub_df.empty:
            print(f"{name}: Empty")
            return
            
        # Group by query
        grouped = sub_df.groupby('query')
        n_queries = 0
        mrr_sum = 0
        map_sum = 0
        hits10 = 0
        
        for qid, group in grouped:
            if group['c_is_gt'].sum() == 0: continue
            n_queries += 1
            
            # Get ranks of GT
            gt_ranks = group[group['c_is_gt'] == 1]['rank'].values
            if len(gt_ranks) == 0: continue
            
            first_rank = gt_ranks.min()
            mrr_sum += 1.0 / first_rank
            
            if first_rank <= 10:
                hits10 += 1
                
            # MAP (simplified for single GT usually, but handles multiple)
            precisions = []
            for i, r in enumerate(sorted(gt_ranks)):
                precisions.append((i+1)/r)
            map_sum += np.mean(precisions)
            
        mrr = mrr_sum / n_queries if n_queries > 0 else 0
        map_score = map_sum / n_queries if n_queries > 0 else 0
        h10 = hits10 / n_queries if n_queries > 0 else 0
        
        print(f"{name} (N={n_queries}): MRR={mrr:.4f}, MAP={map_score:.4f}, Hits@10={h10:.4f}")
        
    get_metrics(df_img, "With Images")
    get_metrics(df_no_img, "Without Images")

if __name__ == "__main__":
    calculate_breakdown(
        'semcluster_similarity_matrix_FULL_weighted.csv',
        'Overall - FULL_trimmed_year_1_corpus_with_gt.csv'
    )
