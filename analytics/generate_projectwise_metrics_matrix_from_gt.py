import pandas as pd
import sys
import os
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from core.metrics import Metrics

# File paths
SIM_MATRIX_FULL = 'output/semcluster_similarity_matrix_FULL.csv'
GT_FULL = 'Dataset/Overall - FULL_trimmed_year_1_corpus_with_gt.csv'
OUTPUT_FULL = 'output/Projectwise_Retrieval_Full.csv'

SIM_MATRIX_FILTERED = 'output/semcluster_similarity_matrix_FILTERED.csv'
GT_FILTERED = 'Dataset/Overall - FILTERED_trimmed_year_1_corpus_with_gt.csv'
OUTPUT_FILTERED = 'output/Projectwise_Retrieval_Filtered.csv'

def calculate_projectwise_metrics(sim_matrix_csv, gt_csv, output_csv, is_full_dataset=False):
    """
    Compare two retrieval strategies:
    For FILTERED dataset (100% images):
      1. Text Only (BB + RS): Use only text components for all queries
      2. Text + Image (BB + RS + SF + CF): Use all components for all queries
    
    For FULL dataset (~13% images):
      1. Text Only (BB + RS): Use only text components for ALL 2323 queries
      2. Text + Image (BB + RS + SF + CF): Use all components for ONLY ~300 queries with images
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    sim_df = pd.read_csv(sim_matrix_csv)
    gt_df = pd.read_csv(gt_csv)
    metrics = Metrics()

    print(f"\nProcessing: {sim_matrix_csv}")
    print(f"Total rows in similarity matrix: {len(sim_df)}")
    
    # Identify queries with images from ground truth
    gt_df['query_project'] = gt_df['query'].astype(str) + '_' + gt_df['Repository_Name']
    queries_with_images_gt = set(gt_df[gt_df['query_has_image'] == True]['query'])
    
    print(f"Queries marked with images in GT: {len(queries_with_images_gt)}")
    print(f"Total queries in GT: {len(gt_df)}")
    
    # Recompute scores for both strategies
    def calc_text_only_score(row):
        """Calculate score using only BB and RS"""
        components = []
        if pd.notna(row['BB']) and row['BB'] != '':
            components.append(float(row['BB']))
        if pd.notna(row['RS']) and row['RS'] != '':
            components.append(float(row['RS']))
        return np.mean(components) if components else np.nan
    
    def calc_text_img_score(row):
        """Calculate score using BB, RS, SF, CF"""
        components = []
        if pd.notna(row['BB']) and row['BB'] != '':
            components.append(float(row['BB']))
        if pd.notna(row['RS']) and row['RS'] != '':
            components.append(float(row['RS']))
        if pd.notna(row['SF']) and row['SF'] != '':
            components.append(float(row['SF']))
        if pd.notna(row['CF']) and row['CF'] != '':
            components.append(float(row['CF']))
        return np.mean(components) if components else np.nan
    
    # Calculate both scores
    sim_df['score_text_only'] = sim_df.apply(calc_text_only_score, axis=1)
    sim_df['score_text_img'] = sim_df.apply(calc_text_img_score, axis=1)
    
    # Re-rank for each query using both strategies
    def rerank_group(group, score_col):
        """Re-rank a group based on the specified score column"""
        group = group.copy()
        group = group.dropna(subset=[score_col])
        group = group.sort_values(by=score_col)
        group['rank'] = range(1, len(group) + 1)
        return group
    
    # Create two separate dataframes with re-ranked results
    sim_df_text_only = sim_df.groupby(['Project', 'query'], group_keys=False).apply(
        lambda x: rerank_group(x, 'score_text_only')
    ).reset_index(drop=True)
    
    sim_df_text_img = sim_df.groupby(['Project', 'query'], group_keys=False).apply(
        lambda x: rerank_group(x, 'score_text_img')
    ).reset_index(drop=True)
    
    print(f"Text Only rankings: {len(sim_df_text_only)} rows")
    print(f"Text + Image rankings: {len(sim_df_text_img)} rows")
    
    # --- Global metrics ---
    # For Text Only: ALL queries
    text_only_global = sim_df_text_only.copy()
    text_only_global['q_id'] = text_only_global['query'].astype(str) + '_' + text_only_global['Project']
    
    # For Text+Image: ONLY queries with images (for FULL dataset), or ALL (for FILTERED)
    if is_full_dataset:
        # FULL: Only use queries that have images
        text_img_global = sim_df_text_img[sim_df_text_img['query'].isin(queries_with_images_gt)].copy()
    else:
        # FILTERED: All queries have images
        text_img_global = sim_df_text_img.copy()
    
    text_img_global['q_id'] = text_img_global['query'].astype(str) + '_' + text_img_global['Project']
    
    global_mrr_text = metrics.computeMRR(text_only_global, 'rank')
    global_mrr_img = metrics.computeMRR(text_img_global, 'rank')
    global_map_text = metrics.computeMAP(text_only_global, 'rank')
    global_map_img = metrics.computeMAP(text_img_global, 'rank')
    global_recall10_text = metrics.computeRecall_K(text_only_global, 10, 'rank')
    global_recall10_img = metrics.computeRecall_K(text_img_global, 10, 'rank')

    print("\nGLOBAL METRICS COMPARISON:")
    print(f"Text Only (BB+RS) - {len(text_only_global['q_id'].unique())} queries:")
    print(f"  MRR={global_mrr_text:.4f}, MAP={global_map_text:.4f}, HITS@10={global_recall10_text*100:.2f}%")
    print(f"Text+Image (BB+RS+SF+CF) - {len(text_img_global['q_id'].unique())} queries:")
    print(f"  MRR={global_mrr_img:.4f}, MAP={global_map_img:.4f}, HITS@10={global_recall10_img*100:.2f}%")
    
    projects = sim_df['Project'].unique()
    results = []

    for project in projects:
        # Get data for this project using both strategies
        project_text_only = sim_df_text_only[sim_df_text_only['Project'] == project].copy()
        project_text_img_all = sim_df_text_img[sim_df_text_img['Project'] == project].copy()
        
        # For FULL dataset: only use queries with images for Text+Image
        if is_full_dataset:
            project_text_img = project_text_img_all[project_text_img_all['query'].isin(queries_with_images_gt)].copy()
        else:
            project_text_img = project_text_img_all.copy()
        
        queries_text_only = project_text_only['query'].unique()
        queries_text_img = project_text_img['query'].unique()
        
        # Prepare for metrics
        project_text_only = project_text_only.rename(columns={'query': 'q_id'})
        project_text_img = project_text_img.rename(columns={'query': 'q_id'})

        # Calculate metrics for both strategies
        mrr_text = metrics.computeMRR(project_text_only, 'rank') if not project_text_only.empty else 0.0
        mrr_img = metrics.computeMRR(project_text_img, 'rank') if not project_text_img.empty else 0.0
        
        map_text = metrics.computeMAP(project_text_only, 'rank') if not project_text_only.empty else 0.0
        map_img = metrics.computeMAP(project_text_img, 'rank') if not project_text_img.empty else 0.0
        
        recall1_text = metrics.computeRecall_K(project_text_only, 1, 'rank') if not project_text_only.empty else 0.0
        recall1_img = metrics.computeRecall_K(project_text_img, 1, 'rank') if not project_text_img.empty else 0.0
        
        recall5_text = metrics.computeRecall_K(project_text_only, 5, 'rank') if not project_text_only.empty else 0.0
        recall5_img = metrics.computeRecall_K(project_text_img, 5, 'rank') if not project_text_img.empty else 0.0
        
        recall10_text = metrics.computeRecall_K(project_text_only, 10, 'rank') if not project_text_only.empty else 0.0
        recall10_img = metrics.computeRecall_K(project_text_img, 10, 'rank') if not project_text_img.empty else 0.0

        # Calculate differences and RI
        mrr_diff = mrr_img - mrr_text
        mrr_ri = ((mrr_img - mrr_text) / mrr_text * 100) if mrr_text > 0 else 0.0
        
        map_diff = map_img - map_text
        map_ri = ((map_img - map_text) / map_text * 100) if map_text > 0 else 0.0
        
        recall1_diff = recall1_img - recall1_text
        recall5_diff = recall5_img - recall5_text
        recall10_diff = recall10_img - recall10_text

        results.append({
            'Repository': project,
            '#queries_text_only': len(queries_text_only),
            '#queries_text+img': len(queries_text_img),
            'MRR_Text Only': f"{mrr_text:.4f}",
            'MRR_Text+Image': f"{mrr_img:.4f}",
            'MRR_Diff': f"{mrr_diff:.4f}",
            'MRR_RI%': f"{mrr_ri:.2f}",
            'MAP_Text Only': f"{map_text:.4f}",
            'MAP_Text+Image': f"{map_img:.4f}",
            'MAP_Diff': f"{map_diff:.4f}",
            'MAP_RI%': f"{map_ri:.2f}",
            'Recall@1_Text Only': f"{recall1_text:.4f}",
            'Recall@1_Text+Image': f"{recall1_img:.4f}",
            'Recall@1_Diff': f"{recall1_diff:.4f}",
            'Recall@5_Text Only': f"{recall5_text:.4f}",
            'Recall@5_Text+Image': f"{recall5_img:.4f}",
            'Recall@5_Diff': f"{recall5_diff:.4f}",
            'Recall@10_Text Only': f"{recall10_text:.4f}",
            'Recall@10_Text+Image': f"{recall10_img:.4f}",
            'Recall@10_Diff': f"{recall10_diff:.4f}",
        })

    # Custom column order and header
    columns = [
        'Repository', '#queries_text_only', '#queries_text+img',
        'MRR_Text Only', 'MRR_Text+Image', 'MRR_Diff', 'MRR_RI%',
        'MAP_Text Only', 'MAP_Text+Image', 'MAP_Diff', 'MAP_RI%',
        'Recall@1_Text Only', 'Recall@1_Text+Image', 'Recall@1_Diff',
        'Recall@5_Text Only', 'Recall@5_Text+Image', 'Recall@5_Diff',
        'Recall@10_Text Only', 'Recall@10_Text+Image', 'Recall@10_Diff',
    ]
    df_out = pd.DataFrame(results)[columns]

    # Compute overall (weighted by #queries) and average (unweighted mean) rows
    metric_cols = [
        'MRR_Text Only', 'MRR_Text+Image', 'MRR_Diff',
        'MAP_Text Only', 'MAP_Text+Image', 'MAP_Diff',
        'Recall@1_Text Only', 'Recall@1_Text+Image', 'Recall@1_Diff',
        'Recall@5_Text Only', 'Recall@5_Text+Image', 'Recall@5_Diff',
        'Recall@10_Text Only', 'Recall@10_Text+Image', 'Recall@10_Diff',
    ]
    
    # Convert metric columns to float (except RI%, which we'll handle separately)
    df_metrics = df_out.copy()
    for col in metric_cols:
        df_metrics[col] = df_metrics[col].astype(float)
    
    # Weighted overall (by #queries for each strategy)
    total_queries_text = df_out['#queries_text_only'].astype(int).sum()
    total_queries_img = df_out['#queries_text+img'].astype(int).sum()
    
    overall = {'Repository': 'Overall', '#queries_text_only': total_queries_text, '#queries_text+img': total_queries_img}
    
    # For text-only metrics, weight by text_only queries
    for col in ['MRR_Text Only', 'MAP_Text Only', 'Recall@1_Text Only', 'Recall@5_Text Only', 'Recall@10_Text Only']:
        val = (df_metrics[col] * df_out['#queries_text_only'].astype(int)).sum() / total_queries_text if total_queries_text else 0.0
        overall[col] = f"{val:.4f}"
    
    # For text+image metrics, weight by text+img queries
    for col in ['MRR_Text+Image', 'MAP_Text+Image', 'Recall@1_Text+Image', 'Recall@5_Text+Image', 'Recall@10_Text+Image']:
        val = (df_metrics[col] * df_out['#queries_text+img'].astype(int)).sum() / total_queries_img if total_queries_img else 0.0
        overall[col] = f"{val:.4f}"
    
    # Diff columns
    for col in ['MRR_Diff', 'MAP_Diff', 'Recall@1_Diff', 'Recall@5_Diff', 'Recall@10_Diff']:
        text_col = col.replace('_Diff', '_Text Only')
        img_col = col.replace('_Diff', '_Text+Image')
        diff = float(overall[img_col]) - float(overall[text_col])
        overall[col] = f"{diff:.4f}"
    
    # Calculate RI% for overall
    text_mrr = float(overall['MRR_Text Only'])
    img_mrr = float(overall['MRR_Text+Image'])
    overall['MRR_RI%'] = f"{((img_mrr - text_mrr) / text_mrr * 100) if text_mrr > 0 else 0.0:.2f}"
    
    text_map = float(overall['MAP_Text Only'])
    img_map = float(overall['MAP_Text+Image'])
    overall['MAP_RI%'] = f"{((img_map - text_map) / text_map * 100) if text_map > 0 else 0.0:.2f}"
    
    # Unweighted average
    avg = {'Repository': 'Average', '#queries_text_only': total_queries_text, '#queries_text+img': total_queries_img}
    for col in metric_cols:
        val = df_metrics[col].mean()
        avg[col] = f"{val:.4f}"
    
    # Calculate RI% for average
    text_mrr_avg = df_metrics['MRR_Text Only'].mean()
    img_mrr_avg = df_metrics['MRR_Text+Image'].mean()
    avg['MRR_RI%'] = f"{((img_mrr_avg - text_mrr_avg) / text_mrr_avg * 100) if text_mrr_avg > 0 else 0.0:.2f}"
    
    text_map_avg = df_metrics['MAP_Text Only'].mean()
    img_map_avg = df_metrics['MAP_Text+Image'].mean()
    avg['MAP_RI%'] = f"{((img_map_avg - text_map_avg) / text_map_avg * 100) if text_map_avg > 0 else 0.0:.2f}"

    # Append rows
    df_out = pd.concat([df_out, pd.DataFrame([overall, avg])], ignore_index=True)
    df_out.to_csv(output_csv, index=False)
    print(f"\nProjectwise metrics matrix saved to {output_csv}")

if __name__ == "__main__":
    print("=" * 80)
    print("TEXT-ONLY vs TEXT+IMAGE RETRIEVAL COMPARISON")
    print("=" * 80)
    
    print("\n" + "=" * 80)
    print("PROCESSING FILTERED DATASET")
    print("All 125 queries have images (100%)")
    print("Strategy 1: Text Only (BB + RS) for all 125 queries")
    print("Strategy 2: Text + Image (BB + RS + SF + CF) for all 125 queries")
    print("=" * 80)
    calculate_projectwise_metrics(SIM_MATRIX_FILTERED, GT_FILTERED, OUTPUT_FILTERED, is_full_dataset=False)
    
    print("\n" + "=" * 80)
    print("PROCESSING FULL DATASET")
    print("2323 total queries: ~304 with images (13%), ~2019 without (87%)")
    print("Strategy 1: Text Only (BB + RS) for ALL 2323 queries")
    print("Strategy 2: Text + Image (BB + RS + SF + CF) for ONLY ~304 queries with images")
    print("=" * 80)
    calculate_projectwise_metrics(SIM_MATRIX_FULL, GT_FULL, OUTPUT_FULL, is_full_dataset=True)
    
    print("\n" + "=" * 80)
    print("COMPLETED - Both datasets processed successfully")
    print("=" * 80)
    print("\nGenerated files:")
    print(f"  - {OUTPUT_FILTERED}")
    print(f"  - {OUTPUT_FULL}")
    print("=" * 80)

