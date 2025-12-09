#!/usr/bin/env python3
"""
SemCluster Evaluation using Pre-computed Embeddings
"""

import os
import sys
import pickle
import argparse
import pandas as pd
import numpy as np
from apted import APTED, helpers
import multiprocessing
from tqdm import tqdm

# Import SemCluster modules for some utilities
from core import cluster
from core.semcluster import calculate_retrieval_metrics, debug_retrieval

# Global variables for worker processes
text_emb = None
struct_emb = None
content_emb = None
stats = None

def parse_id_list(id_string):
    if pd.isna(id_string) or id_string == '[]': return []
    cleaned = id_string.strip('[]').strip()
    if not cleaned: return []
    return [int(x.strip()) for x in cleaned.split('|') if x.strip()]

def load_sample_queries(csv_path, n_queries=-1, require_images=True):
    df = pd.read_csv(csv_path)
    if require_images:
        df_with_images = df[df['query_has_image'] == True].copy()
        df_with_images['gt_size'] = df_with_images['ground_truth_size']
        df_filtered = df_with_images[df_with_images['gt_size'] >= 1]
    else:
        df['gt_size'] = df['ground_truth_size']
        df_filtered = df[df['gt_size'] >= 1]
    
    if n_queries != -1 and n_queries < len(df_filtered):
        return df_filtered.sample(n=n_queries, random_state=42)
    return df_filtered

def dtw_distance_vectors(vecs_a, vecs_b, min_step_dis=0, max_step_dis=0):
    M, N = len(vecs_a), len(vecs_b)
    if M == 0 or N == 0:
        return 1.0
        
    cost = np.zeros((M, N))
    step = np.zeros((M, N))
    
    def dist_fn(v1, v2):
        d = np.linalg.norm(v1 - v2)
        if max_step_dis > min_step_dis:
            return (d - min_step_dis) / (max_step_dis - min_step_dis)
        return d
        
    # Init 0,0
    d = dist_fn(vecs_a[0], vecs_b[0])
    cost[0, 0] = d
    step[0, 0] = 1
    
    # Init first col
    for i in range(1, M):
        d = dist_fn(vecs_a[i], vecs_b[0])
        cost[i, 0] = cost[i-1, 0] + d
        step[i, 0] = i + 1
        
    # Init first row
    for j in range(1, N):
        d = dist_fn(vecs_a[0], vecs_b[j])
        cost[0, j] = cost[0, j-1] + d
        step[0, j] = j + 1
        
    # Fill
    for i in range(1, M):
        for j in range(1, N):
            c_diag, s_diag = cost[i-1, j-1], step[i-1, j-1]
            c_left, s_left = cost[i, j-1], step[i, j-1]
            c_top, s_top = cost[i-1, j], step[i-1, j]
            
            if c_diag < c_left:
                mc, ms = c_diag, s_diag
            elif c_diag == c_left:
                mc, ms = c_diag, min(s_diag, s_left)
            else:
                mc, ms = c_left, s_left
                
            if mc < c_top:
                final_c, final_s = mc, ms
            elif mc == c_top:
                final_c, final_s = mc, min(ms, s_top)
            else:
                final_c, final_s = c_top, s_top
                
            d = dist_fn(vecs_a[i], vecs_b[j])
            cost[i, j] = final_c + d
            step[i, j] = final_s + 1
            
    return cost[-1, -1] / step[-1, -1]

def cal_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1+w1, x2+w2)
    yi2 = min(y1+h1, y2+h2)
    inter_w = max(0, xi2 - xi1)
    inter_h = max(0, yi2 - yi1)
    inter_area = inter_w * inter_h
    union_area = (w1 * h1) + (w2 * h2) - inter_area
    if union_area == 0: return 0
    return inter_area / union_area

def content_dist(widgets1, widgets2):
    if not widgets1 or not widgets2:
        return 1.0
        
    matches = []
    used_j = set()
    
    for i, w1 in enumerate(widgets1):
        best_iou = 0
        best_j = -1
        for j, w2 in enumerate(widgets2):
            if j in used_j: continue
            iou = cal_iou(w1['bbox'], w2['bbox'])
            if iou > 0.7 and iou > best_iou:
                best_iou = iou
                best_j = j
        
        if best_j != -1:
            matches.append((w1, widgets2[best_j]))
            used_j.add(best_j)
            
    if not matches:
        return 1.0
        
    dists = []
    for m1, m2 in matches:
        v1 = m1['vector'] / (np.linalg.norm(m1['vector']) + 1e-9)
        v2 = m2['vector'] / (np.linalg.norm(m2['vector']) + 1e-9)
        d = np.linalg.norm(v1 - v2)
        dists.append(d)
        
    return sum(dists) / len(dists)

def process_pair(pair):
    k1, k2 = pair
    
    # Parse keys
    r1, i1 = k1.split(':')
    r2, i2 = k2.split(':')
    key_tuple = (r1, int(i1), int(i2))
    rev_key_tuple = (r2, int(i2), int(i1))
    
    scores = []
    
    # Text
    if k1 in text_emb and k2 in text_emb:
        # Problem
        prob_dist = np.linalg.norm(text_emb[k1]['problem_vector'] - text_emb[k2]['problem_vector'])
        pd_norm = (prob_dist - stats['prob']['min']) / (stats['prob']['max'] - stats['prob']['min'])
        scores.append(pd_norm)
        
        # Procedure
        rd = dtw_distance_vectors(text_emb[k1]['procedure_vectors'], text_emb[k2]['procedure_vectors'],
                                stats['proc']['min'], stats['proc']['max'])
        scores.append(rd)
        
        # SemCluster Constraints
        if pd_norm < 0.3: # Must link check (partial)
            pass
            
    # Structure
    if k1 in struct_emb and k2 in struct_emb:
        try:
            apted = APTED(struct_emb[k1], struct_emb[k2])
            sd = apted.compute_edit_distance()
            sd_norm = (sd - stats['struct']['min']) / (stats['struct']['max'] - stats['struct']['min'])
            scores.append(sd_norm)
        except: pass
        
    # Content
    if k1 in content_emb and k2 in content_emb:
        cd = content_dist(content_emb[k1], content_emb[k2])
        cd_norm = (cd - stats['content']['min']) / (stats['content']['max'] - stats['content']['min'])
        scores.append(cd_norm)
        
    if scores:
        avg_score = sum(scores) / len(scores)
        # Map scores back to components for return
        # Order of appending: BB, RS, SF, CF (if available)
        # This is tricky because some might be missing.
        # Let's reconstruct explicitly.
        
        comps = {}
        
        # Text (BB, RS)
        if k1 in text_emb and k2 in text_emb:
            # Re-calculate or capture from above? 
            # Better to capture variables.
            # Let's refactor this function slightly to be cleaner.
            pass
            
    return None

def process_pair(pair):
    k1, k2 = pair
    
    # Parse keys
    r1, i1 = k1.split(':')
    r2, i2 = k2.split(':')
    key_tuple = (r1, int(i1), int(i2))
    rev_key_tuple = (r2, int(i2), int(i1))
    
    scores = []
    comps = {'BB': None, 'RS': None, 'SF': None, 'CF': None}
    
    # Text
    if k1 in text_emb and k2 in text_emb:
        # Problem
        prob_dist = np.linalg.norm(text_emb[k1]['problem_vector'] - text_emb[k2]['problem_vector'])
        pd_norm = (prob_dist - stats['prob']['min']) / (stats['prob']['max'] - stats['prob']['min'])
        scores.append(pd_norm)
        comps['BB'] = pd_norm
        
        # Procedure
        rd = dtw_distance_vectors(text_emb[k1]['procedure_vectors'], text_emb[k2]['procedure_vectors'],
                                stats['proc']['min'], stats['proc']['max'])
        scores.append(rd)
        comps['RS'] = rd
        
        # SemCluster Constraints
        if pd_norm < 0.3: # Must link check (partial)
            pass
            
    # Structure
    if k1 in struct_emb and k2 in struct_emb:
        try:
            apted = APTED(struct_emb[k1], struct_emb[k2])
            sd = apted.compute_edit_distance()
            sd_norm = (sd - stats['struct']['min']) / (stats['struct']['max'] - stats['struct']['min'])
            scores.append(sd_norm)
            comps['SF'] = sd_norm
        except: pass
        
    # Content
    if k1 in content_emb and k2 in content_emb:
        cd = content_dist(content_emb[k1], content_emb[k2])
        cd_norm = (cd - stats['content']['min']) / (stats['content']['max'] - stats['content']['min'])
        scores.append(cd_norm)
        comps['CF'] = cd_norm
        
    if scores:
        avg_score = sum(scores) / len(scores)
        return (key_tuple, rev_key_tuple, avg_score, comps)
    return None

def main():
    global text_emb, struct_emb, content_emb, stats
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='FILTERED', choices=['FILTERED', 'FULL'])
    args = parser.parse_args()
    
    print(f"Running evaluation from embeddings for {args.dataset} dataset...")
    
    # Load Embeddings
    print("Loading embeddings...")
    with open('embeddings/text_embeddings.pkl', 'rb') as f:
        text_emb = pickle.load(f)
    with open('embeddings/structure_embeddings.pkl', 'rb') as f:
        struct_emb = pickle.load(f)
    with open('embeddings/content_embeddings.pkl', 'rb') as f:
        content_emb = pickle.load(f)
    print("Embeddings loaded.")
    
    # Load GT
    if args.dataset == 'FILTERED':
        gt_csv = 'Overall - FILTERED_trimmed_year_1_corpus_with_gt.csv'
        require_images = True
    else:
        gt_csv = 'Overall - FULL_trimmed_year_1_corpus_with_gt.csv'
        require_images = False
        
    sample_df = load_sample_queries(gt_csv, require_images=require_images)
    print(f"Loaded {len(sample_df)} queries.")
    
    # Identify needed pairs
    query_to_valid_corpus = {}
    needed_pairs = set()
    
    for idx, row in sample_df.iterrows():
        qid = row['query']
        repo = row['Repository_Name']
        q_key = f"{repo}:{qid}"
        
        if require_images:
            corpus_ids = parse_id_list(row.get('corpus_issues_with_images', '[]'))
        else:
            corpus_ids = parse_id_list(row.get('corpus', '[]'))
            
        valid_set = set()
        for cid in corpus_ids:
            c_key = f"{repo}:{cid}"
            valid_set.add((repo, cid))
            needed_pairs.add(tuple(sorted((q_key, c_key))))
            
        query_to_valid_corpus[(repo, qid)] = valid_set
        
    print(f"Identified {len(needed_pairs)} unique pairs to compute.")
    
    # 1. Compute Global Min/Max for Normalization (Sampled)
    print("Computing normalization statistics...")
    stats = {
        'prob': {'min': 1e9, 'max': 0},
        'proc': {'min': 1e9, 'max': 0},
        'struct': {'min': 1e9, 'max': 0},
        'content': {'min': 1e9, 'max': 0}
    }
    
    # Improved Sampling for Stats
    # We need to ensure we sample enough pairs that HAVE images to get valid stats for struct/content
    import random
    
    # Sample keys from embeddings directly to ensure coverage
    text_keys = list(text_emb.keys())
    struct_keys = list(struct_emb.keys())
    content_keys = list(content_emb.keys())
    
    # Sample 2000 pairs for each modality
    n_samples = 2000
    
    # Text Stats
    for _ in range(n_samples):
        k1 = random.choice(text_keys)
        k2 = random.choice(text_keys)
        if k1 == k2: continue
        
        # Problem
        d = np.linalg.norm(text_emb[k1]['problem_vector'] - text_emb[k2]['problem_vector'])
        stats['prob']['min'] = min(stats['prob']['min'], d)
        stats['prob']['max'] = max(stats['prob']['max'], d)
        
        # Procedure
        if text_emb[k1]['procedure_vectors'] and text_emb[k2]['procedure_vectors']:
            v1 = text_emb[k1]['procedure_vectors'][0]
            v2 = text_emb[k2]['procedure_vectors'][0]
            d = np.linalg.norm(v1 - v2)
            stats['proc']['min'] = min(stats['proc']['min'], d)
            stats['proc']['max'] = max(stats['proc']['max'], d)
            
    # Structure Stats
    if struct_keys:
        for _ in range(n_samples):
            k1 = random.choice(struct_keys)
            k2 = random.choice(struct_keys)
            if k1 == k2: continue
            
            try:
                apted = APTED(struct_emb[k1], struct_emb[k2])
                d = apted.compute_edit_distance()
                stats['struct']['min'] = min(stats['struct']['min'], d)
                stats['struct']['max'] = max(stats['struct']['max'], d)
            except: pass
            
    # Content Stats
    if content_keys:
        for _ in range(n_samples):
            k1 = random.choice(content_keys)
            k2 = random.choice(content_keys)
            if k1 == k2: continue
            
            d = content_dist(content_emb[k1], content_emb[k2])
            stats['content']['min'] = min(stats['content']['min'], d)
            stats['content']['max'] = max(stats['content']['max'], d)
            
    print("Stats:", stats)
    
    # Ensure non-zero ranges
    for k in stats:
        if stats[k]['max'] == stats[k]['min']:
            stats[k]['max'] = stats[k]['min'] + 1.0
            
    # 2. Compute All Pairs (Parallel)
    print(f"Computing all pairs using {multiprocessing.cpu_count()} cores...")
    combined_pairs = {}
    
    pairs_list = list(needed_pairs)
    
    # Use multiprocessing pool
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        # Use imap_unordered for better responsiveness and tqdm
        results = list(tqdm(pool.imap_unordered(process_pair, pairs_list, chunksize=100), total=len(pairs_list)))
        
    for res in results:
        if res:
            key_tuple, rev_key_tuple, avg_score, comps = res
            combined_pairs[key_tuple] = (avg_score, comps)
            combined_pairs[rev_key_tuple] = (avg_score, comps)
            
    # 3. Generate Output CSV
    print("Generating output CSV...")
    rows = []
    
    # Ground Truth Map
    query_to_gt = {}
    for idx, row in sample_df.iterrows():
        qid = row['query']
        repo = row['Repository_Name']
        q_key = f"{repo}:{qid}"
        
        if require_images:
            gt_ids = parse_id_list(row.get('ground_truth_issues_with_images', '[]'))
        else:
            gt_ids = parse_id_list(row.get('ground_truth', '[]'))
            
        query_to_gt[q_key] = set(f"{repo}:{gid}" for gid in gt_ids)

    for idx, row in sample_df.iterrows():
        qid = row['query']
        repo = row['Repository_Name']
        q_key = f"{repo}:{qid}"
        
        valid_corpus = query_to_valid_corpus.get((repo, qid), set())
        gt_set = query_to_gt.get(q_key, set())
        
        similarities = []
        for c_repo, cid in valid_corpus:
            c_key = f"{c_repo}:{cid}"
            if c_key == q_key: continue
            
            dist_data = combined_pairs.get((repo, qid, cid))
            if dist_data:
                dist, comps = dist_data
            else:
                dist = 1.0
                comps = {'BB': None, 'RS': None, 'SF': None, 'CF': None}
            
            similarities.append((c_key, dist, comps))
            
        similarities.sort(key=lambda x: x[1])
        
        for rank, (c_key, dist, comps) in enumerate(similarities, 1):
            c_is_gt = 1 if c_key in gt_set else 0
            
            row_dict = {
                'Project': repo,
                'query': qid,
                'corpus': c_key,
                'score': round(dist, 6),
                'rank': rank,
                'c_is_gt': c_is_gt
            }
            
            # Add components
            for k in ['BB', 'RS', 'SF', 'CF']:
                val = comps.get(k)
                row_dict[k] = round(val, 6) if val is not None else ''
                
            rows.append(row_dict)
            
    out_df = pd.DataFrame(rows)
    out_path = f'semcluster_similarity_matrix_{args.dataset}.csv'
    out_df.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")
    
    # Calculate Metrics
    print("Calculating metrics...")
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runtime_verification'))
    from calculate_filtered_metrics_v2 import calculate_metrics_from_result_csv
    calculate_metrics_from_result_csv(out_path)
    # print("Metrics calculation skipped (module missing).")

if __name__ == '__main__':
    main()
