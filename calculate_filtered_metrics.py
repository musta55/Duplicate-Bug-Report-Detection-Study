import pandas as pd
import numpy as np
import sys

def parse_id_list(id_string):
    """Parse '[id1| id2| id3]' format into list of integers"""
    if pd.isna(id_string) or id_string == '[]':
        return []
    cleaned = id_string.strip('[]').strip()
    if not cleaned:
        return []
    ids = [int(x.strip()) for x in cleaned.split('|') if x.strip()]
    return ids

def create_ground_truth_dataframe(sample_df):
    """Create ground truth DataFrame for evaluation"""
    rows = []
    for idx, query_row in sample_df.iterrows():
        query_id = query_row['query']
        # Use ground_truth_issues_with_images for FILTERED
        if 'ground_truth_issues_with_images' in query_row and pd.notna(query_row.get('ground_truth_issues_with_images')):
            gt_ids = parse_id_list(query_row['ground_truth_issues_with_images'])
        else:
            gt_ids = parse_id_list(query_row['ground_truth'])
        duplicate_group = [query_id] + gt_ids
        cluster_label = min(duplicate_group)
        
        for rid in duplicate_group:
            rows.append({'id': rid, 'group': cluster_label})
    
    gt_df = pd.DataFrame(rows)
    print(f"\nGround truth: {len(gt_df)} reports in {gt_df['group'].nunique()} duplicate groups")
    return gt_df

def calculate_retrieval_metrics(similarity_df, ground_truth_df, k_values=range(1, 11)):
    """
    Calculates MRR, MAP, and HITS@k for the given similarity matrix.
    """
    # Ensure IDs are integers for comparison
    ground_truth_df['id'] = ground_truth_df['id'].astype(int)
    report_ids = ground_truth_df['id'].unique()
    ground_truth = {group: list(ids) for group, ids in ground_truth_df.groupby('group')['id']}

    reciprocal_ranks = []
    average_precisions = []
    hits_at_k = {k: 0 for k in k_values}

    # Ensure index is string for lookup
    similarity_df['index'] = similarity_df['index'].astype(str)
    
    # Pre-process columns to handle "Repo:ID" format
    # We need to map "Repo:ID" columns to just ID integers for this calculation
    # OR map the ground truth IDs to "Repo:ID" format.
    # The similarity matrix has columns like "Repo:ID".
    # The ground truth DF has integer IDs.
    # We need to know the Repo for each ID in ground truth to match correctly.
    
    # Let's inspect the similarity matrix columns first
    sim_cols = list(similarity_df.columns)
    # print(f"Similarity columns sample: {sim_cols[:5]}")
    
    # If columns are "Repo:ID", we need to handle that.
    # However, calculate_retrieval_metrics in main.py assumes integer IDs in columns.
    # Let's adapt the logic here.
    
    processed_queries = 0
    
    for query_id in report_ids:
        # Get true duplicates for the current query
        true_duplicates = []
        for group_ids in ground_truth.values():
            if query_id in group_ids:
                true_duplicates = [pid for pid in group_ids if pid != query_id]
                break

        if not true_duplicates:
            continue

        # Find the row for this query
        # The index might be "Repo:ID" or just "ID"
        # We need to find the row where the ID part matches query_id
        
        # Try exact match first (if index is just ID)
        if str(query_id) in similarity_df['index'].values:
            row = similarity_df[similarity_df['index'] == str(query_id)]
        else:
            # Try to find "Repo:ID"
            # This is slow, but robust
            matches = [idx for idx in similarity_df['index'] if str(idx).endswith(f":{query_id}")]
            if matches:
                row = similarity_df[similarity_df['index'] == matches[0]]
            else:
                # print(f"Query {query_id} not found in similarity matrix")
                continue
        
        processed_queries += 1
        
        # Get ranked list
        # Columns are also "Repo:ID" or "ID"
        # We need to extract the ID part and sort
        
        scores = []
        for col in similarity_df.columns:
            if col == 'index':
                continue
            
            try:
                val = float(row[col].iloc[0])
            except:
                continue
                
            # Extract ID from col
            if ':' in str(col):
                try:
                    col_id = int(str(col).split(':')[1])
                except:
                    continue
            else:
                try:
                    col_id = int(col)
                except:
                    continue
            
            if col_id != query_id:
                scores.append((col_id, val))
        
        # Sort by distance (ascending)
        scores.sort(key=lambda x: x[1])
        ranked_ids = [x[0] for x in scores]

        # --- MRR Calculation ---
        first_correct_rank = -1
        for i, pred_id in enumerate(ranked_ids):
            if pred_id in true_duplicates:
                first_correct_rank = i + 1
                break
        if first_correct_rank != -1:
            reciprocal_ranks.append(1 / first_correct_rank)
        else:
            reciprocal_ranks.append(0)

        # --- MAP Calculation ---
        precision_scores = []
        correct_hits = 0
        for i, pred_id in enumerate(ranked_ids):
            if pred_id in true_duplicates:
                correct_hits += 1
                precision_scores.append(correct_hits / (i + 1))

        if precision_scores:
            average_precisions.append(np.mean(precision_scores))
        else:
            average_precisions.append(0)

        # --- HITS@k Calculation ---
        for k in k_values:
            top_k = ranked_ids[:k]
            if any(pred_id in true_duplicates for pred_id in top_k):
                hits_at_k[k] += 1

    print(f"Processed {processed_queries} queries out of {len(report_ids)}")

    mrr = np.mean(reciprocal_ranks) if reciprocal_ranks else 0
    map_score = np.mean(average_precisions) if average_precisions else 0

    # Filter for queries that had duplicates to get a fair HITS score
    total_queries_with_duplicates = sum(
        1 for query_id in report_ids if any(query_id in g and len(g) > 1 for g in ground_truth.values()))
    
    # Adjust total queries to only those we actually processed (found in matrix)
    # This is important if the matrix is partial
    # total_queries_with_duplicates = processed_queries 
    
    hits_scores = {k: (v / processed_queries) if processed_queries > 0 else 0 for k, v in
                   hits_at_k.items()}

    return mrr, map_score, hits_scores

def main():
    print("Loading similarity matrix...")
    sim_df = pd.read_csv('semcluster_similarity_matrix_FILTERED.csv')
    print(f"Loaded matrix with {len(sim_df)} rows and {len(sim_df.columns)} columns")
    
    print("Loading ground truth...")
    gt_csv = 'Overall - FILTERED_trimmed_year_1_corpus_with_gt.csv'
    sample_df = pd.read_csv(gt_csv)
    
    # Filter for queries with images as per run_parquet_evaluation.py
    df_with_images = sample_df[sample_df['query_has_image'] == True].copy()
    df_with_images['gt_size'] = df_with_images['ground_truth_size']
    df_filtered = df_with_images[df_with_images['gt_size'] >= 1] # min_duplicates=1
    
    gt_df = create_ground_truth_dataframe(df_filtered)
    
    print("Calculating metrics...")
    mrr, map_score, hits_scores = calculate_retrieval_metrics(sim_df, gt_df)
    
    print("\n" + "="*70)
    print("RETRIEVAL RESULTS (FILTERED)")
    print("="*70)
    print(f"MRR: {mrr:.4f}")
    print(f"MAP: {map_score:.4f}")
    print("HITS@k:")
    for k in sorted(hits_scores.keys()):
        print(f"  HITS@{k:2d}: {hits_scores[k]:.4f}")

if __name__ == "__main__":
    main()
