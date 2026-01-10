#!/usr/bin/env python3
"""
Post-Processing for SemCluster Test Results

This module processes raw test results to calculate individual component scores
and generate comprehensive projectwise retrieval metrics.

Functions:
1. add_component_scores(): Calculate BB, RS, SF, CF distances from embeddings
   - BB: Euclidean distance on Bag of Words vectors
   - RS: DTW distance on Recurrence Sequence vectors
   - SF: APTED tree edit distance on UI structures
   - CF: Euclidean distance on VGG16 content features

2. generate_projectwise_metrics(): Calculate retrieval metrics per project
   - MRR (Mean Reciprocal Rank)
   - MAP (Mean Average Precision)
   - Recall@K (K=1, 5, 10)
   - Uses core.metrics.Metrics class (matching main project)

Input:
- test_output/semcluster_similarity_matrix_*.csv (from test_scenarios.py)
- test_output/embeddings/*.pkl (BB, RS, SF, CF embeddings)

Output:
- Updated similarity matrix CSV with component scores
- Projectwise_Retrieval_*.csv with complete metrics

Usage:
    python post_process_results.py
    # Or use the test pipeline script:
    ./test_pipeline_quick.sh

Author: SemCluster Team
Version: 2.0 (Multi-project support)
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.spatial.distance import euclidean
from apted import APTED

# Add project root
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def compute_dtw_distance(seq1, seq2):
    """Compute DTW distance between two sequences of vectors"""
    M, N = len(seq1), len(seq2)
    if M == 0 or N == 0:
        return 1.0
        
    cost = np.zeros((M, N))
    
    # Initialize
    cost[0, 0] = euclidean(seq1[0], seq2[0])
    for i in range(1, M):
        cost[i, 0] = cost[i-1, 0] + euclidean(seq1[i], seq2[0])
    for j in range(1, N):
        cost[0, j] = cost[0, j-1] + euclidean(seq1[0], seq2[j])
    
    # Fill matrix
    for i in range(1, M):
        for j in range(1, N):
            d = euclidean(seq1[i], seq2[j])
            cost[i, j] = d + min(cost[i-1, j], cost[i, j-1], cost[i-1, j-1])
    
    return cost[M-1, N-1] / (M + N)

def add_component_scores(dataset_name):
    """Add BB, RS, SF, CF scores to similarity matrix CSV"""
    
    test_output = project_root / 'test_output'
    embeddings_dir = test_output / 'embeddings'
    
    # Load pickle files - use old structure (text/structure/content)
    text_path = embeddings_dir / f'text_embeddings_{dataset_name.lower()}.pkl'
    struct_path = embeddings_dir / f'structure_embeddings_{dataset_name.lower()}.pkl'
    content_path = embeddings_dir / f'content_embeddings_{dataset_name.lower()}.pkl'
    
    if not text_path.exists():
        print(f"Warning: Could not find embeddings for {dataset_name}")
        return
    
    with open(text_path, 'rb') as f:
        text_data = pickle.load(f)
    with open(struct_path, 'rb') as f:
        sf_emb = pickle.load(f)
    with open(content_path, 'rb') as f:
        cf_emb = pickle.load(f)
    
    # Extract BB and RS from combined text embeddings
    bb_emb = {k: v['problem_vector'] for k, v in text_data.items()}
    rs_emb = {k: v['procedure_vectors'] for k, v in text_data.items()}
    
    # Load similarity matrix CSV
    csv_path = test_output / f'semcluster_similarity_matrix_{dataset_name}.csv'
    df = pd.read_csv(csv_path)
    
    print(f"Processing {dataset_name} similarity matrix ({len(df)} rows)...")
    
    # Compute component scores for each pair
    bb_scores = []
    rs_scores = []
    sf_scores = []
    cf_scores = []
    
    for idx, row in df.iterrows():
        q_id = row['query']
        c_id = row['corpus']
        
        # BB: Bag of Words distance
        if q_id in bb_emb and c_id in bb_emb:
            bb_dist = euclidean(bb_emb[q_id], bb_emb[c_id])
        else:
            bb_dist = None
        bb_scores.append(bb_dist)
        
        # RS: Recurrence Sequence (DTW)
        if q_id in rs_emb and c_id in rs_emb:
            rs_dist = compute_dtw_distance(rs_emb[q_id], rs_emb[c_id])
        else:
            rs_dist = None
        rs_scores.append(rs_dist)
        
        # SF: Structure Feature (APTED)
        if q_id in sf_emb and c_id in sf_emb:
            sf_dist = APTED(sf_emb[q_id], sf_emb[c_id]).compute_edit_distance()
        else:
            sf_dist = None
        sf_scores.append(sf_dist)
        
        # CF: Content Feature
        if q_id in cf_emb and c_id in cf_emb:
            # Average distance between all region pairs
            q_vecs = [r['vector'] for r in cf_emb[q_id]]
            c_vecs = [r['vector'] for r in cf_emb[c_id]]
            distances = []
            for qv in q_vecs:
                for cv in c_vecs:
                    distances.append(euclidean(qv, cv))
            cf_dist = np.mean(distances) if distances else None
        else:
            cf_dist = None
        cf_scores.append(cf_dist)
    
    # Update DataFrame
    df['BB'] = [round(x, 6) if x is not None else '' for x in bb_scores]
    df['RS'] = [round(x, 6) if x is not None else '' for x in rs_scores]
    df['SF'] = [round(x, 6) if x is not None else '' for x in sf_scores]
    df['CF'] = [round(x, 6) if x is not None else '' for x in cf_scores]
    
    # Recalculate combined score based on available components
    new_scores = []
    for idx in range(len(df)):
        components = []
        if bb_scores[idx] is not None:
            components.append(bb_scores[idx])
        if rs_scores[idx] is not None:
            components.append(rs_scores[idx])
        if sf_scores[idx] is not None:
            components.append(sf_scores[idx])
        if cf_scores[idx] is not None:
            components.append(cf_scores[idx])
        new_scores.append(round(np.mean(components), 6) if components else 0.0)
    
    df['score'] = new_scores
    
    # Re-rank within each query
    df = df.sort_values(['query', 'score']).reset_index(drop=True)
    df['rank'] = df.groupby('query').cumcount() + 1
    
    # Save updated CSV
    df.to_csv(csv_path, index=False)
    print(f"✓ Updated: {csv_path}")
    print(f"  Sample: BB={df.iloc[0]['BB']}, RS={df.iloc[0]['RS']}, "
          f"SF={df.iloc[0]['SF']}, CF={df.iloc[0]['CF']}")

def generate_projectwise_metrics(dataset_name):
    """Generate projectwise retrieval metrics CSV using core.metrics.Metrics"""
    
    test_output = project_root / 'test_output'
    csv_path = test_output / f'semcluster_similarity_matrix_{dataset_name}.csv'
    
    from core.metrics import Metrics
    metrics_calc = Metrics()
    
    sim_df = pd.read_csv(csv_path)
    
    print(f"\nGenerating projectwise metrics for {dataset_name}...")
    
    # Calculate score_text_only (BB + RS) and score_text_img (BB + RS + SF + CF)
    def calc_text_only_score(row):
        components = []
        if pd.notna(row['BB']) and row['BB'] != '':
            components.append(float(row['BB']))
        if pd.notna(row['RS']) and row['RS'] != '':
            components.append(float(row['RS']))
        return np.mean(components) if components else np.nan
    
    def calc_text_img_score(row):
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
    
    sim_df['score_text_only'] = sim_df.apply(calc_text_only_score, axis=1)
    sim_df['score_text_img'] = sim_df.apply(calc_text_img_score, axis=1)
    
    # Re-rank for each query using both strategies
    def rerank_group(group, score_col):
        group = group.copy()
        group = group.dropna(subset=[score_col])
        group = group.sort_values(by=score_col)
        group['rank_' + score_col] = range(1, len(group) + 1)
        return group
    
    sim_df_text_only = sim_df.groupby(['Project', 'query'], group_keys=False).apply(
        lambda x: rerank_group(x, 'score_text_only')
    ).reset_index(drop=True)
    
    sim_df_text_img = sim_df.groupby(['Project', 'query'], group_keys=False).apply(
        lambda x: rerank_group(x, 'score_text_img')
    ).reset_index(drop=True)
    
    projects = sim_df['Project'].unique()
    results = []
    
    for project in projects:
        project_text_only = sim_df_text_only[sim_df_text_only['Project'] == project].copy()
        project_text_img = sim_df_text_img[sim_df_text_img['Project'] == project].copy()
        
        queries_text_only = project_text_only['query'].unique()
        queries_with_images = project_text_img[(project_text_img['SF'] != '') & (project_text_img['CF'] != '')]['query'].nunique()
        
        # Prepare for metrics - add q_id column
        project_text_only['q_id'] = project_text_only['query'].astype(str) + '_' + project_text_only['Project']
        project_text_img['q_id'] = project_text_img['query'].astype(str) + '_' + project_text_img['Project']
        
        # Calculate metrics using core.metrics.Metrics
        mrr_text = metrics_calc.computeMRR(project_text_only, 'rank_score_text_only') if not project_text_only.empty else 0.0
        mrr_img = metrics_calc.computeMRR(project_text_img, 'rank_score_text_img') if not project_text_img.empty else 0.0
        
        map_text = metrics_calc.computeMAP(project_text_only, 'rank_score_text_only') if not project_text_only.empty else 0.0
        map_img = metrics_calc.computeMAP(project_text_img, 'rank_score_text_img') if not project_text_img.empty else 0.0
        
        recall1_text = metrics_calc.computeRecall_K(project_text_only, 1, 'rank_score_text_only') if not project_text_only.empty else 0.0
        recall1_img = metrics_calc.computeRecall_K(project_text_img, 1, 'rank_score_text_img') if not project_text_img.empty else 0.0
        
        recall5_text = metrics_calc.computeRecall_K(project_text_only, 5, 'rank_score_text_only') if not project_text_only.empty else 0.0
        recall5_img = metrics_calc.computeRecall_K(project_text_img, 5, 'rank_score_text_img') if not project_text_img.empty else 0.0
        
        recall10_text = metrics_calc.computeRecall_K(project_text_only, 10, 'rank_score_text_only') if not project_text_only.empty else 0.0
        recall10_img = metrics_calc.computeRecall_K(project_text_img, 10, 'rank_score_text_img') if not project_text_img.empty else 0.0
        
        # Calculate differences and RI
        mrr_diff = mrr_img - mrr_text
        mrr_ri = ((mrr_img - mrr_text) / mrr_text * 100) if mrr_text > 0 else 0.0
        
        map_diff = map_img - map_text
        map_ri = ((map_img - map_text) / map_text * 100) if map_text > 0 else 0.0
        
        results.append({
            'Repository': project,
            '#queries': len(queries_text_only),
            '#queries with image': queries_with_images,
            'MRR_Text Only': round(mrr_text, 4),
            'MRR_Text+Image': round(mrr_img, 4),
            'MRR_Diff': round(mrr_diff, 4),
            'MRR_RI': round(mrr_ri, 2),
            'MAP_Text Only': round(map_text, 4),
            'MAP_Text+Image': round(map_img, 4),
            'MAP_Diff': round(map_diff, 4),
            'MAP_RI': round(map_ri, 2),
            'Recall@1_Text Only': round(recall1_text, 4),
            'Recall@1_Text+Image': round(recall1_img, 4),
            'Recall@1_Diff': round(recall1_text - recall1_img, 4),
            'Recall@5_Text Only': round(recall5_text, 4),
            'Recall@5_Text+Image': round(recall5_img, 4),
            'Recall@5_Diff': round(recall5_text - recall5_img, 4),
            'Recall@10_Text Only': round(recall10_text, 4),
            'Recall@10_Text+Image': round(recall10_img, 4),
            'Recall@10_Diff': round(recall10_text - recall10_img, 4),
        })
    
    metrics_df = pd.DataFrame(results)
    metrics_path = test_output / f'Projectwise_Retrieval_{dataset_name}.csv'
    metrics_df.to_csv(metrics_path, index=False)
    print(f"✓ Generated: {metrics_path}")

if __name__ == '__main__':
    print("="*70)
    print("Post-Processing Test Results")
    print("="*70)
    
    for dataset in ['FILTERED', 'FULL']:
        csv_path = project_root / 'test_output' / f'semcluster_similarity_matrix_{dataset}.csv'
        if csv_path.exists():
            print(f"\n{dataset} Dataset:")
            add_component_scores(dataset)
            generate_projectwise_metrics(dataset)
        else:
            print(f"\nSkipping {dataset} (CSV not found)")
    
    print("\n" + "="*70)
    print("Post-processing complete!")
    print("="*70)
