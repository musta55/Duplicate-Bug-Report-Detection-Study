import os
import sys
import pickle
import pandas as pd
import numpy as np
from apted import APTED
from sklearn.preprocessing import normalize

# Add current directory to path
sys.path.append(os.getcwd())

import cluster
from main import calculate_retrieval_metrics

# Configuration
EMBEDDING_DIR = 'embeddings/'
GT_CSV = 'Overall - FULL_trimmed_year_1_corpus_with_gt.csv'
OUTPUT_CSV = 'semcluster_similarity_matrix_FULL_retrieval.csv'

def load_embeddings():
    print("Loading embeddings...")
    with open(os.path.join(EMBEDDING_DIR, 'text_embeddings.pkl'), 'rb') as f:
        text_embeds = pickle.load(f)
    with open(os.path.join(EMBEDDING_DIR, 'structure_embeddings.pkl'), 'rb') as f:
        struct_embeds = pickle.load(f)
    with open(os.path.join(EMBEDDING_DIR, 'content_embeddings.pkl'), 'rb') as f:
        content_embeds = pickle.load(f)
    print(f"Loaded: Text={len(text_embeds)}, Struct={len(struct_embeds)}, Content={len(content_embeds)}")
    return text_embeds, struct_embeds, content_embeds

def parse_id_list(id_string):
    if pd.isna(id_string) or id_string == '[]': return []
    cleaned = id_string.strip('[]').strip()
    if not cleaned: return []
    return [int(x.strip()) for x in cleaned.split('|') if x.strip()]

def compute_text_distance(id1, id2, text_embeds):
    if id1 not in text_embeds or id2 not in text_embeds:
        return 1.0
        
    e1 = text_embeds[id1]
    e2 = text_embeds[id2]
    
    # Problem Distance (Euclidean)
    v1 = e1['problem_vector']
    v2 = e2['problem_vector']
    prob_dist = np.linalg.norm(v1 - v2)
    
    # Procedure Distance (DTW on vectors)
    vecs1 = e1.get('procedure_vectors', [])
    vecs2 = e2.get('procedure_vectors', [])
    
    proc_dist = dtw_distance_vectors(vecs1, vecs2)
    
    # Combine (Average? Or weighted?)
    # Original code: 
    # sp_dis = normalize(prob_dist)
    # sr_dis = normalize(proc_dist)
    # But here we return raw distances and normalize later globally.
    # Let's return a tuple or combined raw distance.
    # For simplicity, let's return average of raw distances.
    
    return (prob_dist + proc_dist) / 2.0

def dtw_distance_vectors(ts_a, ts_b):
    """DTW on list of vectors"""
    M, N = len(ts_a), len(ts_b)
    
    if M == 0 or N == 0:
        return 1.0
        
    cost = np.zeros((M, N))
    step = np.zeros((M, N))
    
    # Initialize [0,0]
    dis = np.linalg.norm(ts_a[0] - ts_b[0])
    cost[0, 0] = dis
    step[0, 0] = 1
    
    # First column
    for i in range(1, M):
        dis = np.linalg.norm(ts_a[i] - ts_b[0])
        cost[i, 0] = cost[i-1, 0] + dis
        step[i, 0] = step[i-1, 0] + 1
        
    # First row
    for j in range(1, N):
        dis = np.linalg.norm(ts_a[0] - ts_b[j])
        cost[0, j] = cost[0, j-1] + dis
        step[0, j] = step[0, j-1] + 1
        
    # Rest
    for i in range(1, M):
        for j in range(1, N):
            # Find min prev
            c_diag = cost[i-1, j-1]
            c_top = cost[i-1, j]
            c_left = cost[i, j-1]
            
            if c_diag <= c_top and c_diag <= c_left:
                min_cost = c_diag
                min_step = step[i-1, j-1]
            elif c_top <= c_left:
                min_cost = c_top
                min_step = step[i-1, j]
            else:
                min_cost = c_left
                min_step = step[i, j-1]
                
            dis = np.linalg.norm(ts_a[i] - ts_b[j])
            cost[i, j] = min_cost + dis
            step[i, j] = min_step + 1
            
    return cost[-1, -1] / step[-1, -1]


def compute_structure_distance(id1, id2, struct_embeds):
    if id1 not in struct_embeds or id2 not in struct_embeds:
        return 1.0 # Max distance
        
    t1 = struct_embeds[id1]
    t2 = struct_embeds[id2]
    
    try:
        apted = APTED(t1, t2)
        return apted.compute_edit_distance()
    except:
        return 1.0

def compute_content_distance(id1, id2, content_embeds):
    if id1 not in content_embeds or id2 not in content_embeds:
        return 1.0
        
    widgets1 = content_embeds[id1]
    widgets2 = content_embeds[id2]
    
    if not widgets1 or not widgets2:
        return 1.0
        
    # Match widgets by IoU
    # widgets is list of {'bbox': [x,y,w,h], 'vector': vec}
    
    matched_pairs = []
    used_j = set()
    
    for i, w1 in enumerate(widgets1):
        best_iou = 0
        best_j = -1
        
        for j, w2 in enumerate(widgets2):
            if j in used_j: continue
            
            # Calculate IoU
            iou = calculate_iou(w1['bbox'], w2['bbox'])
            if iou > 0.7 and iou > best_iou: # Threshold from original code
                best_iou = iou
                best_j = j
        
        if best_j != -1:
            matched_pairs.append((w1['vector'], widgets2[best_j]['vector']))
            used_j.add(best_j)
            
    if not matched_pairs:
        return 1.0
        
    # Compute distance for matched pairs
    dist_sum = 0
    for v1, v2 in matched_pairs:
        # Normalize? Original code does normalize=True
        v1n = v1 / (np.linalg.norm(v1) + 1e-9)
        v2n = v2 / (np.linalg.norm(v2) + 1e-9)
        dist_sum += np.linalg.norm(v1n - v2n)
        
    return dist_sum / len(matched_pairs)

def calculate_iou(box1, box2):
    # box: [x, y, w, h]
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1+w1, x2+w2)
    yi2 = min(y1+h1, y2+h2)
    
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0: return 0
    return inter_area / union_area

def main():
    text_embeds, struct_embeds, content_embeds = load_embeddings()
    
    print("Loading Ground Truth...")
    df = pd.read_csv(GT_CSV)
    
    # Identify pairs to compute
    pairs = []
    query_to_gt = {}
    
    print("Generating pairs...")
    for idx, row in df.iterrows():
        repo = row['Repository_Name']
        qid = row['query']
        q_key = f"{repo}:{qid}"
        
        gt_ids = parse_id_list(row['ground_truth'])
        corpus_ids = parse_id_list(row['corpus'])
        
        # Store GT for evaluation
        query_to_gt[q_key] = set(f"{repo}:{gid}" for gid in gt_ids)
        
        # Add pairs
        for cid in corpus_ids:
            c_key = f"{repo}:{cid}"
            pairs.append((q_key, c_key, repo, qid, cid))
            
    print(f"Total pairs to compute: {len(pairs)}")
    
    results = []
    
    # Compute distances
    # We need to normalize distances globally or per query?
    # Original code normalizes by Max Distance observed.
    # We should track max distances.
    
    max_text_dist = 0
    max_struct_dist = 0
    max_content_dist = 0
    
    raw_scores = []
    
    for i, (q_key, c_key, repo, qid, cid) in enumerate(pairs):
        if i % 1000 == 0:
            print(f"Processed {i}/{len(pairs)} pairs...")
            
        d_text = compute_text_distance(q_key, c_key, text_embeds)
        d_struct = compute_structure_distance(q_key, c_key, struct_embeds)
        d_content = compute_content_distance(q_key, c_key, content_embeds)
        
        max_text_dist = max(max_text_dist, d_text)
        max_struct_dist = max(max_struct_dist, d_struct)
        max_content_dist = max(max_content_dist, d_content)
        
        raw_scores.append({
            'q': q_key, 'c': c_key, 
            'repo': repo, 'qid': qid, 'cid': cid,
            'dt': d_text, 'ds': d_struct, 'dc': d_content
        })
        
    # Normalize and Combine
    print("Normalizing and combining scores...")
    final_rows = []
    
    for item in raw_scores:
        nt = item['dt'] / max_text_dist if max_text_dist > 0 else 0
        ns = item['ds'] / max_struct_dist if max_struct_dist > 0 else 0
        nc = item['dc'] / max_content_dist if max_content_dist > 0 else 0
        
        # Combination logic (Average)
        # Original code has complex logic with weights and thresholds.
        # Simplified average for now:
        score = (nt + ns + nc) / 3.0
        
        is_gt = 1 if item['c'] in query_to_gt.get(item['q'], set()) else 0
        
        final_rows.append({
            'Project': item['repo'],
            'query': item['qid'],
            'corpus': item['cid'], # Just ID or composite? Output format expects ID usually
            'score': round(score, 6),
            'c_is_gt': is_gt
        })
        
    # Save
    res_df = pd.DataFrame(final_rows)
    # Add rank
    res_df['rank'] = res_df.groupby(['Project', 'query'])['score'].rank(method='first', ascending=True)
    
    res_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved results to {OUTPUT_CSV}")

if __name__ == '__main__':
    main()
