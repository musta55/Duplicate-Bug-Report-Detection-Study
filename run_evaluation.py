#!/usr/bin/env python3
"""
Unified SemCluster Evaluation Script
Supports two modes:
1. Fast mode: Load pre-computed embeddings (if available)
2. Full mode: Generate embeddings from parquet data
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
from PIL import Image
from io import BytesIO
import shutil
import pyarrow.parquet as pq

# Add root directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import cluster
from image import image_main
from text import text_main
from core.semcluster import calculate_retrieval_metrics, debug_retrieval


def parse_id_list(id_string):
    """Parse '[id1| id2| id3]' format into list of integers"""
    if pd.isna(id_string) or id_string == '[]':
        return []
    cleaned = id_string.strip('[]').strip()
    if not cleaned:
        return []
    ids = [int(x.strip()) for x in cleaned.split('|') if x.strip()]
    return ids


def load_sample_queries(csv_path, n_queries=-1, min_duplicates=1, require_images=True):
    """Load sample queries from CSV
    
    Args:
        require_images: If True, only load queries with images (for FILTERED).
                       If False, load all queries (for FULL, handles text-only).
    """
    df = pd.read_csv(csv_path)
    
    # Filter based on image requirement
    if require_images:
        # FILTERED: queries with images and at least min_duplicates
        df_with_images = df[df['query_has_image'] == True].copy()
        df_with_images['gt_size'] = df_with_images['ground_truth_size']
        df_filtered = df_with_images[df_with_images['gt_size'] >= min_duplicates]
    else:
        # FULL: all queries (with or without images)
        df['gt_size'] = df['ground_truth_size']
        df_filtered = df[df['gt_size'] >= min_duplicates]
    
    # Sample queries (n_queries=-1 means all queries)
    if n_queries == -1 or n_queries > len(df_filtered):
        print(f"  Using all {len(df_filtered)} available queries")
        sample_df = df_filtered
    else:
        if len(df_filtered) < n_queries:
            print(f"  Warning: Only {len(df_filtered)} queries available (requested {n_queries})")
        sample_df = df_filtered.sample(n=n_queries, random_state=42)
    
    return sample_df


def check_embeddings_exist(dataset):
    """Check if pre-computed embeddings exist for the dataset"""
    pkl_files = [
        f'embeddings/text_embeddings_{dataset.lower()}.pkl',
        f'embeddings/structure_embeddings_{dataset.lower()}.pkl',
        f'embeddings/content_embeddings_{dataset.lower()}.pkl'
    ]
    
    all_exist = all(os.path.exists(f) for f in pkl_files)
    
    if all_exist:
        print(f"✓ Found pre-computed embeddings for {dataset} dataset:")
        for f in pkl_files:
            size_mb = os.path.getsize(f) / (1024 * 1024)
            print(f"  - {f} ({size_mb:.2f} MB)")
    else:
        print(f"✗ Pre-computed embeddings not found for {dataset} dataset")
        print(f"  Will generate embeddings from parquet data")
        
    return all_exist


def load_embeddings(dataset):
    """Load pre-computed embeddings"""
    print(f"\nLoading embeddings from pickle files...")
    
    with open(f'embeddings/text_embeddings_{dataset.lower()}.pkl', 'rb') as f:
        text_emb = pickle.load(f)
    with open(f'embeddings/structure_embeddings_{dataset.lower()}.pkl', 'rb') as f:
        struct_emb = pickle.load(f)
    with open(f'embeddings/content_embeddings_{dataset.lower()}.pkl', 'rb') as f:
        content_emb = pickle.load(f)
        
    print(f"  Text embeddings: {len(text_emb)} reports")
    print(f"  Structure embeddings: {len(struct_emb)} reports")
    print(f"  Content embeddings: {len(content_emb)} reports")
    
    return text_emb, struct_emb, content_emb


# ============================================================================
# FULL PIPELINE: Generate Embeddings from Parquet
# ============================================================================

def load_parquet_data(parquet_path, report_ids_by_repo):
    """Load specific bug reports from parquet file
    
    Args:
        parquet_path: Path to parquet file
        report_ids_by_repo: Dict mapping repo_name -> set of IDs
    """
    table = pq.read_table(parquet_path)
    df = table.to_pandas()

    dfs = []
    for repo_name, ids in report_ids_by_repo.items():
        repo_df = df[(df['repo_name'] == repo_name) & (df['id'].isin(ids))].copy()
        dfs.append(repo_df)
    
    if not dfs:
        print(f"  ✗ No reports found in parquet!")
        return pd.DataFrame()
    
    df_filtered = pd.concat(dfs, ignore_index=True)
    
    total_requested = sum(len(ids) for ids in report_ids_by_repo.values())
    total_found = len(df_filtered)
    with_images = len(df_filtered[df_filtered['valid_image'] == True])
    
    print(f"  Loaded {total_found}/{total_requested} reports from parquet")
    print(f"  Reports with valid images: {with_images}/{total_found}")
    
    if total_found < total_requested:
        missing_by_repo = {}
        for repo_name, requested_ids in report_ids_by_repo.items():
            found_ids = set(df_filtered[df_filtered['repo_name'] == repo_name]['id'].tolist())
            missing = requested_ids - found_ids
            if missing:
                missing_by_repo[repo_name] = missing
        
        if missing_by_repo:
            print(f"  ⚠ Missing reports:")
            for repo, missing_ids in missing_by_repo.items():
                print(f"    {repo}: {sorted(missing_ids)}")
    
    return df_filtered


def extract_images_from_parquet(parquet_df, output_dir):
    """Extract images from parquet binary data to disk"""
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    extracted_count = 0
    missing_count = 0
    error_count = 0
    
    id_to_seq = {}
    seq_to_id = {}
    reports_with_images = set()
    
    for seq_idx, (idx, row) in enumerate(parquet_df.iterrows()):
        report_id = row['id']
        repo_name = row['repo_name']
        composite_id = f"{repo_name}:{report_id}"
        
        id_to_seq[composite_id] = seq_idx
        seq_to_id[seq_idx] = composite_id
        
        image_data = row.get('image')
        
        if pd.isna(image_data) or image_data is None:
            missing_count += 1
            continue
        
        if isinstance(image_data, dict):
            image_bytes = image_data.get('bytes')
        else:
            image_bytes = image_data
        
        if image_bytes is None or len(image_bytes) == 0:
            missing_count += 1
            continue
        
        try:
            img = Image.open(BytesIO(image_bytes))
            img_path = os.path.join(output_dir, f"report_img_{seq_idx}.png")
            img.save(img_path)
            extracted_count += 1
            reports_with_images.add(seq_idx)
        except Exception as e:
            error_count += 1
    
    print(f"  Extracted {extracted_count} images, {missing_count} missing, {error_count} errors")
    return extracted_count, id_to_seq, seq_to_id, reports_with_images


def prepare_evaluation_csv(sample_df, parquet_df, img_dir, output_path, id_to_seq):
    """Create CSV file for SemCluster evaluation"""
    id_to_cluster = {}
    for idx, query_row in sample_df.iterrows():
        query_id = query_row['query']
        repo_name = query_row['Repository_Name']
        
        if 'ground_truth_issues_with_images' in query_row and pd.notna(query_row.get('ground_truth_issues_with_images')):
            gt_ids = parse_id_list(query_row['ground_truth_issues_with_images'])
        else:
            gt_ids = parse_id_list(query_row.get('ground_truth', '[]'))
            
        duplicate_group = [query_id] + gt_ids
        cluster_label = min(duplicate_group)
        for rid in duplicate_group:
            composite_id = f"{repo_name}:{rid}"
            id_to_cluster[composite_id] = cluster_label
    
    image_files = sorted([f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    content_map = {}
    for idx, row in parquet_df.iterrows():
        report_id = row['id']
        repo_name = row['repo_name']
        composite_id = f"{repo_name}:{report_id}"
        
        title = str(row.get('title', ''))
        description = str(row.get('description', ''))
        comments = str(row.get('comments', ''))
        full_text = f"{title} {description} {comments}".strip()
        content_map[composite_id] = full_text
    
    seq_to_id = {v: k for k, v in id_to_seq.items()}
    
    rows = []
    for img_file in image_files:
        base_name = img_file.rsplit('.', 1)[0]
        if '_' in base_name:
            seq_idx = int(base_name.split('_')[-1])
        else:
            seq_idx = int(base_name)
        
        composite_id = seq_to_id.get(seq_idx, str(seq_idx))
        cluster = id_to_cluster.get(composite_id, -1)
        description = content_map.get(composite_id, '').replace('\n', ' ').replace(',', ';')[:500]
        
        rows.append({
            'index': seq_idx,
            'description': description,
            'img_url': '',
            'appid': '',
            'useless': '',
            'tag': cluster
        })
    
    df_eval = pd.DataFrame(rows)
    df_eval.to_csv(output_path, index=False)
    
    print(f"  Created CSV with {len(df_eval)} rows ({len(image_files)} images, {len(id_to_cluster)} clusters)")
    return df_eval, id_to_cluster


def run_semcluster_pipeline(eval_csv_path, img_dir, xml_dir, reports_with_images, seq_to_id, parquet_df, sample_df, query_to_valid_corpus):
    """Run full SemCluster feature extraction pipeline"""
    print("\n" + "="*70)
    print("SEMCLUSTER FEATURE EXTRACTION")
    print("="*70)
    
    if not img_dir.endswith('/'):
        img_dir = img_dir + '/'
    if not xml_dir.endswith('/'):
        xml_dir = xml_dir + '/'
    
    n_with_images = len(reports_with_images)
    n_without_images = len(seq_to_id) - len(reports_with_images)
    
    # Image features
    print("\n[1/2] Extracting image features...")
    print(f"  Reports with images: {n_with_images}")
    print(f"  Reports without images: {n_without_images}")
    
    st_pairs = {}
    ct_pairs = {}

    if n_with_images > 0:
        np.seterr(all='raise')
        
        try:
            st_list, ct_list, st_pairs, ct_pairs = image_main.image_main(
                eval_csv_path, img_dir, xml_dir
            )
            print(f"  ✓ Structure features: {len(st_list)-1} reports")
            print(f"  ✓ Content features: {len(ct_list)-1} reports")
        except Exception as e:
            print(f"  ✗ Image feature extraction failed: {e}")
            header = list(range(1, 513))
            header.insert(0, 'index')
            st_list = [header]
            ct_list = [header]
            st_pairs = {}
            ct_pairs = {}
        
        if n_without_images > 0:
            print(f"  → Adding zero features for {n_without_images} reports without images")
    else:
        print(f"  → No images available, creating zero features for all {len(seq_to_id)} reports")
        header = list(range(1, 513))
        header.insert(0, 'index')
        st_list = [header]
        ct_list = [header]
        st_pairs = {}
        ct_pairs = {}
    
    # Text features
    print("\n[2/2] Extracting text features...")
    print(f"  Processing text for all {len(seq_to_id)} reports...")
    
    all_reports_csv = eval_csv_path.replace('evaluation.csv', 'evaluation_all_reports.csv')
    
    rows = []
    for seq_idx in sorted(seq_to_id.keys()):
        composite_id = seq_to_id[seq_idx]
        if ':' in str(composite_id):
            repo_name, report_id = composite_id.split(':', 1)
            report_id = int(report_id)
        else:
            repo_name = 'Unknown'
            report_id = int(composite_id)
            
        report_row = parquet_df[(parquet_df['repo_name'] == repo_name) & (parquet_df['id'] == report_id)]
        if len(report_row) > 0:
            row = report_row.iloc[0]
            title = str(row.get('title', ''))
            description = str(row.get('description', ''))
            comments = str(row.get('comments', ''))
            full_text = f"{title} {description} {comments}".strip()
            cluster = -1
            
            rows.append({
                'index': seq_idx,
                'description': full_text,
                'img_url': '',
                'appid': '',
                'useless': '',
                'tag': cluster
            })
    
    pd.DataFrame(rows).to_csv(all_reports_csv, index=False)
    print(f"  ✓ Created text CSV with {len(rows)} reports")
    
    p_pairs = {}
    r_pairs = {}

    try:
        print(f"  → Running TextCNN on GPU...")
        p_list, r_list, p_pairs, r_pairs = text_main.text_main(
            all_reports_csv,
            sample_df,
            query_to_valid_corpus,
            seq_to_id,
            parquet_df
        )
        print(f"  ✓ TextCNN problem features: {len(p_list)-1} reports")
        print(f"  ✓ TextCNN reproduction features: {len(r_list)-1} reports")
    except Exception as e:
        print(f"  ✗ CRITICAL: TextCNN feature extraction failed: {e}")
        raise RuntimeError(f"TextCNN extraction failed - cannot proceed: {e}")
    
    return {
        'st_list': st_list,
        'ct_list': ct_list,
        'p_list': p_list,
        'r_list': r_list,
        'st_pairs': st_pairs,
        'ct_pairs': ct_pairs,
        'p_pairs': p_pairs,
        'r_pairs': r_pairs
    }


def save_embeddings_as_pickles(feature_data, seq_to_id, parquet_df, reports_with_images, dataset):
    """Convert feature_data to pickle format and save"""
    print(f"\nSaving embeddings to pickle files...")
    
    # Text embeddings
    text_emb = {}
    p_list = feature_data['p_list']
    r_list = feature_data['r_list']
    
    for seq_idx in seq_to_id.keys():
        composite_id = seq_to_id[seq_idx]
        
        # Find in p_list and r_list
        p_vec = None
        r_vecs = []
        
        for row in p_list[1:]:  # Skip header
            if int(row[0]) == seq_idx:
                p_vec = np.array(row[1:], dtype=np.float32)
                break
                
        for row in r_list[1:]:
            if int(row[0]) == seq_idx:
                r_vecs.append(np.array(row[1:], dtype=np.float32))
                
        if p_vec is not None:
            text_emb[composite_id] = {
                'problem_vector': p_vec,
                'procedure_vectors': r_vecs
            }
    
    # Structure embeddings (tree strings)
    struct_emb = {}
    st_pairs = feature_data.get('st_pairs', {})
    
    # Extract tree strings from XML files if available
    xml_dir = f'file/xml_file_parquet_{dataset.lower()}'
    if os.path.exists(xml_dir):
        for seq_idx in reports_with_images:
            composite_id = seq_to_id[seq_idx]
            xml_file = os.path.join(xml_dir, f'layout{seq_idx}.xml')
            if os.path.exists(xml_file):
                try:
                    tree = helpers.Tree.from_text(open(xml_file).read())
                    struct_emb[composite_id] = tree
                except:
                    pass
    
    # Content embeddings (widget lists)
    content_emb = {}
    ct_list = feature_data['ct_list']
    
    for seq_idx in reports_with_images:
        composite_id = seq_to_id[seq_idx]
        
        # Find in ct_list
        for row in ct_list[1:]:
            if int(row[0]) == seq_idx:
                vec = np.array(row[1:], dtype=np.float32)
                # Store as simple widget dict (bbox + vector)
                content_emb[composite_id] = [{
                    'bbox': [0, 0, 100, 100],  # Dummy bbox
                    'vector': vec
                }]
                break
    
    # Save pickle files
    os.makedirs('embeddings', exist_ok=True)
    
    with open(f'embeddings/text_embeddings_{dataset.lower()}.pkl', 'wb') as f:
        pickle.dump(text_emb, f)
    with open(f'embeddings/structure_embeddings_{dataset.lower()}.pkl', 'wb') as f:
        pickle.dump(struct_emb, f)
    with open(f'embeddings/content_embeddings_{dataset.lower()}.pkl', 'wb') as f:
        pickle.dump(content_emb, f)
        
    print(f"  ✓ Saved text_embeddings_{dataset.lower()}.pkl ({len(text_emb)} reports)")
    print(f"  ✓ Saved structure_embeddings_{dataset.lower()}.pkl ({len(struct_emb)} reports)")
    print(f"  ✓ Saved content_embeddings_{dataset.lower()}.pkl ({len(content_emb)} reports)")
    
    return text_emb, struct_emb, content_emb


# ============================================================================
# FAST PATH: Compute Similarities from Pre-computed Embeddings
# ============================================================================

def dtw_distance_vectors(vecs_a, vecs_b, min_step_dis=0, max_step_dis=0):
    """Dynamic Time Warping distance"""
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
        
    d = dist_fn(vecs_a[0], vecs_b[0])
    cost[0, 0] = d
    step[0, 0] = 1
    
    for i in range(1, M):
        d = dist_fn(vecs_a[i], vecs_b[0])
        cost[i, 0] = cost[i-1, 0] + d
        step[i, 0] = i + 1
        
    for j in range(1, N):
        d = dist_fn(vecs_a[0], vecs_b[j])
        cost[0, j] = cost[0, j-1] + d
        step[0, j] = j + 1
        
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
    """Calculate IoU between two bounding boxes"""
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
    """Content distance between widget lists"""
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


def compute_normalization_stats(text_emb, struct_emb, content_emb, n_samples=2000):
    """Compute global min/max statistics for normalization"""
    import random
    
    print("Computing normalization statistics...")
    stats = {
        'prob': {'min': 1e9, 'max': 0},
        'proc': {'min': 1e9, 'max': 0},
        'struct': {'min': 1e9, 'max': 0},
        'content': {'min': 1e9, 'max': 0}
    }
    
    # Text Stats
    text_keys = list(text_emb.keys())
    for _ in range(min(n_samples, len(text_keys) * len(text_keys) // 2)):
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
    struct_keys = list(struct_emb.keys())
    if struct_keys:
        for _ in range(min(n_samples, len(struct_keys) * len(struct_keys) // 2)):
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
    content_keys = list(content_emb.keys())
    if content_keys:
        for _ in range(min(n_samples, len(content_keys) * len(content_keys) // 2)):
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
            
    return stats


# Global variables for multiprocessing
_text_emb = None
_struct_emb = None
_content_emb = None
_stats = None

def init_worker(text_emb, struct_emb, content_emb, stats):
    """Initialize worker process with shared data"""
    global _text_emb, _struct_emb, _content_emb, _stats
    _text_emb = text_emb
    _struct_emb = struct_emb
    _content_emb = content_emb
    _stats = stats


def process_pair(pair):
    """Process a single pair and compute distances"""
    k1, k2 = pair
    
    r1, i1 = k1.split(':')
    r2, i2 = k2.split(':')
    key_tuple = (r1, int(i1), int(i2))
    rev_key_tuple = (r2, int(i2), int(i1))
    
    scores = []
    comps = {'BB': None, 'RS': None, 'SF': None, 'CF': None}
    
    # Text
    if k1 in _text_emb and k2 in _text_emb:
        # Problem (Bag of Words)
        prob_dist = np.linalg.norm(_text_emb[k1]['problem_vector'] - _text_emb[k2]['problem_vector'])
        pd_norm = (prob_dist - _stats['prob']['min']) / (_stats['prob']['max'] - _stats['prob']['min'])
        scores.append(pd_norm)
        comps['BB'] = pd_norm
        
        # Procedure (Recurrence Sequence)
        rd = dtw_distance_vectors(_text_emb[k1]['procedure_vectors'], _text_emb[k2]['procedure_vectors'],
                                _stats['proc']['min'], _stats['proc']['max'])
        scores.append(rd)
        comps['RS'] = rd
        
    # Structure
    if k1 in _struct_emb and k2 in _struct_emb:
        try:
            apted = APTED(_struct_emb[k1], _struct_emb[k2])
            sd = apted.compute_edit_distance()
            sd_norm = (sd - _stats['struct']['min']) / (_stats['struct']['max'] - _stats['struct']['min'])
            scores.append(sd_norm)
            comps['SF'] = sd_norm
        except: pass
        
    # Content
    if k1 in _content_emb and k2 in _content_emb:
        cd = content_dist(_content_emb[k1], _content_emb[k2])
        cd_norm = (cd - _stats['content']['min']) / (_stats['content']['max'] - _stats['content']['min'])
        scores.append(cd_norm)
        comps['CF'] = cd_norm
        
    if scores:
        avg_score = sum(scores) / len(scores)
        return (key_tuple, rev_key_tuple, avg_score, comps)
    return None


def compute_similarities_from_embeddings(text_emb, struct_emb, content_emb, sample_df, require_images):
    """Compute pairwise similarities using pre-loaded embeddings"""
    print("\n" + "="*70)
    print("COMPUTING SIMILARITIES FROM EMBEDDINGS")
    print("="*70)
    
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
    
    # Compute normalization statistics
    stats = compute_normalization_stats(text_emb, struct_emb, content_emb)
    
    # Compute all pairs in parallel
    print(f"\nComputing all pairs using {multiprocessing.cpu_count()} cores...")
    combined_pairs = {}
    
    pairs_list = list(needed_pairs)
    
    with multiprocessing.Pool(processes=multiprocessing.cpu_count(), 
                              initializer=init_worker,
                              initargs=(text_emb, struct_emb, content_emb, stats)) as pool:
        results = list(tqdm(pool.imap_unordered(process_pair, pairs_list, chunksize=100), 
                           total=len(pairs_list)))
        
    for res in results:
        if res:
            key_tuple, rev_key_tuple, avg_score, comps = res
            combined_pairs[key_tuple] = (avg_score, comps)
            combined_pairs[rev_key_tuple] = (avg_score, comps)
            
    return combined_pairs, query_to_valid_corpus


# ============================================================================
# OUTPUT GENERATION
# ============================================================================

def generate_similarity_csv(sample_df, combined_pairs, query_to_valid_corpus, output_path, require_images):
    """Generate similarity matrix CSV"""
    print("\nGenerating output CSV...")
    
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

    rows = []
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
    out_df.to_csv(output_path, index=False)
    
    print(f"✓ Similarity matrix CSV saved to: {output_path}")
    print(f"  Total query-corpus pairs: {len(out_df)}")
    print(f"  Unique queries: {out_df['query'].nunique()}")
    if len(out_df) > 0:
        gt_pairs = out_df[out_df['c_is_gt'] == 1]
        print(f"  Ground truth pairs: {len(gt_pairs)}")
    
    return out_df


def calculate_metrics(output_csv):
    """Calculate retrieval metrics from result CSV"""
    print("\n" + "="*70)
    print("CALCULATING RETRIEVAL METRICS")
    print("="*70)
    
    try:
        sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runtime_verification'))
        from calculate_filtered_metrics_v2 import calculate_metrics_from_result_csv
        calculate_metrics_from_result_csv(output_csv)
    except ImportError:
        print("  ⚠ Metrics calculation module not found (runtime_verification/calculate_filtered_metrics_v2.py)")
        print("  → Manual metrics calculation from CSV")


def main():
    parser = argparse.ArgumentParser(description='Unified SemCluster Evaluation')
    parser.add_argument('--dataset', type=str, default='FILTERED', 
                       choices=['FILTERED', 'FULL'],
                       help='Dataset to evaluate: FILTERED (images only) or FULL (all queries)')
    parser.add_argument('--n-queries', type=int, default=-1,
                       help='Number of queries to process (-1 for all available)')
    parser.add_argument('--force-rebuild', action='store_true',
                       help='Force rebuild embeddings even if they exist')
    args = parser.parse_args()
    
    print("="*70)
    print(f"UNIFIED SEMCLUSTER EVALUATION - {args.dataset} DATASET")
    print("="*70)
    
    # Configuration
    if args.dataset == 'FILTERED':
        ground_truth_csv = 'Dataset/Overall - FILTERED_trimmed_year_1_corpus_with_gt.csv'
        require_images = True
        output_suffix = 'FILTERED'
    else:
        ground_truth_csv = 'Dataset/Overall - FULL_trimmed_year_1_corpus_with_gt.csv'
        require_images = False
        output_suffix = 'FULL'
    
    output_csv = f'output/semcluster_similarity_matrix_{output_suffix}.csv'
    
    # Check if embeddings exist
    embeddings_exist = check_embeddings_exist(args.dataset) and not args.force_rebuild
    
    # Load queries
    print(f"\nLoading queries from {ground_truth_csv}...")
    sample_df = load_sample_queries(ground_truth_csv, n_queries=args.n_queries, require_images=require_images)
    print(f"  Selected {len(sample_df)} queries")
    
    if embeddings_exist:
        # ========== FAST PATH: Load embeddings ==========
        print("\n" + "="*70)
        print("MODE: FAST (Loading pre-computed embeddings)")
        print("="*70)
        
        text_emb, struct_emb, content_emb = load_embeddings(args.dataset)
        combined_pairs, query_to_valid_corpus = compute_similarities_from_embeddings(
            text_emb, struct_emb, content_emb, sample_df, require_images
        )
        
    else:
        # ========== FULL PATH: Generate embeddings ==========
        print("\n" + "="*70)
        print("MODE: FULL (Generating embeddings from parquet)")
        print("="*70)
        
        parquet_file = 'Dataset/bug_reports_with_images.parquet'
        img_dir = f'file/pic_file_parquet_{args.dataset.lower()}'
        xml_dir = f'file/xml_file_parquet_{args.dataset.lower()}'
        eval_csv = f'file/label_file_parquet/evaluation_{args.dataset.lower()}.csv'
        
        # Clean directories
        for d in [img_dir, xml_dir]:
            if os.path.exists(d):
                shutil.rmtree(d)
        
        os.makedirs(os.path.dirname(eval_csv), exist_ok=True)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(xml_dir, exist_ok=True)
        
        # Collect needed reports
        report_ids_by_repo = {}
        query_to_valid_corpus = {}
        
        for idx, row in sample_df.iterrows():
            qid = row['query']
            repo = row['Repository_Name']
            
            if require_images:
                gt_ids = parse_id_list(row.get('ground_truth_issues_with_images', '[]'))
                corpus_ids = parse_id_list(row.get('corpus_issues_with_images', '[]'))
            else:
                gt_ids = parse_id_list(row.get('ground_truth', '[]'))
                corpus_ids = parse_id_list(row.get('corpus', '[]'))
            
            all_ids = set([qid] + gt_ids + corpus_ids)
            
            if repo not in report_ids_by_repo:
                report_ids_by_repo[repo] = set()
            report_ids_by_repo[repo].update(all_ids)
            
            valid_set = set((repo, cid) for cid in corpus_ids)
            query_to_valid_corpus[(repo, qid)] = valid_set
        
        total_ids = sum(len(ids) for ids in report_ids_by_repo.items())
        print(f"  Total unique reports needed: {total_ids}")
        
        # Load parquet
        print(f"\nLoading bug reports from {parquet_file}...")
        parquet_df = load_parquet_data(parquet_file, report_ids_by_repo)
        
        # Extract images
        print(f"\nExtracting images to {img_dir}...")
        img_count, id_to_seq, seq_to_id, reports_with_images = extract_images_from_parquet(parquet_df, img_dir)
        
        # Prepare evaluation CSV
        print(f"\nCreating evaluation CSV at {eval_csv}...")
        eval_df, id_to_cluster = prepare_evaluation_csv(sample_df, parquet_df, img_dir, eval_csv, id_to_seq)
        
        # Run feature extraction
        print(f"\nRunning SemCluster feature extraction...")
        feature_data = run_semcluster_pipeline(
            eval_csv, img_dir, xml_dir, reports_with_images, seq_to_id, 
            parquet_df, sample_df, query_to_valid_corpus
        )
        
        # Save embeddings as pickle files
        text_emb, struct_emb, content_emb = save_embeddings_as_pickles(
            feature_data, seq_to_id, parquet_df, reports_with_images, args.dataset
        )
        
        # Compute similarities
        combined_pairs, query_to_valid_corpus = compute_similarities_from_embeddings(
            text_emb, struct_emb, content_emb, sample_df, require_images
        )
    
    # Generate output CSV
    print("\n" + "="*70)
    print("GENERATING OUTPUT")
    print("="*70)
    
    result_df = generate_similarity_csv(
        sample_df, combined_pairs, query_to_valid_corpus, output_csv, require_images
    )
    
    # Calculate metrics
    calculate_metrics(output_csv)
    
    # Final summary
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)
    print(f"Queries evaluated: {len(sample_df)}")
    print(f"Output file: {output_csv}")
    print(f"Total rows: {len(result_df)}")
    print("="*70)


if __name__ == '__main__':
    main()
