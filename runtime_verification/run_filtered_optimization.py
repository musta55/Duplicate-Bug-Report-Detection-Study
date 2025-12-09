#!/usr/bin/env python3

import os
import sys
import argparse
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from io import BytesIO
import pickle

# Add root directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import SemCluster modules
from core import cluster
import image.image_main as image_main
import text.text_main as text_main
from core.semcluster import calculate_retrieval_metrics, debug_retrieval

# Monkey patch image.vgg16.getdistance to use normalize=True by default
import image.vgg16
original_getdistance = image.vgg16.getdistance
def normalized_getdistance(widget_list, normalize=True):
    return original_getdistance(widget_list, normalize=True)
image.vgg16.getdistance = normalized_getdistance

def load_sample_queries(csv_path, n_queries=10, min_duplicates=2, require_images=True):
    """Load sample queries from CSV"""
    df = pd.read_csv(csv_path)
    
    if require_images:
        df_with_images = df[df['query_has_image'] == True].copy()
        df_with_images['gt_size'] = df_with_images['ground_truth_size']
        df_filtered = df_with_images[df_with_images['gt_size'] >= min_duplicates]
    else:
        df['gt_size'] = df['ground_truth_size']
        df_filtered = df[df['gt_size'] >= min_duplicates]
    
    if n_queries == -1 or n_queries > len(df_filtered):
        print(f"  Using all {len(df_filtered)} available queries")
        sample_df = df_filtered
    else:
        if len(df_filtered) < n_queries:
            print(f"  Warning: Only {len(df_filtered)} queries available (requested {n_queries})")
            n_queries = len(df_filtered)
        sample_df = df_filtered.sample(n=n_queries, random_state=42)
    
    return sample_df

def parse_id_list(id_string):
    if pd.isna(id_string) or id_string == '[]':
        return []
    cleaned = id_string.strip('[]').strip()
    if not cleaned:
        return []
    ids = [int(x.strip()) for x in cleaned.split('|') if x.strip()]
    return ids

def load_parquet_data(parquet_path, report_ids_by_repo):
    table = pq.read_table(parquet_path)
    df = table.to_pandas()
    
    dfs = []
    for repo_name, ids in report_ids_by_repo.items():
        repo_df = df[(df['repo_name'] == repo_name) & (df['id'].isin(ids))].copy()
        dfs.append(repo_df)
    
    if not dfs:
        return pd.DataFrame()
    
    df_filtered = pd.concat(dfs, ignore_index=True)
    return df_filtered

def extract_images_from_parquet_resume(parquet_df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    extracted_count = 0
    skipped_count = 0
    missing_count = 0
    error_count = 0
    
    id_to_seq = {}
    seq_to_id = {}
    reports_with_images = set()
    
    print(f"  Checking {len(parquet_df)} reports for images...")
    
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
            
        output_path = os.path.join(output_dir, f"report_img_{seq_idx}.png")
        if os.path.exists(output_path):
            extracted_count += 1
            skipped_count += 1
            reports_with_images.add(composite_id)
            continue
        
        try:
            img = Image.open(BytesIO(image_bytes))
            if img.mode == 'CMYK':
                img = img.convert('RGB')
            img.save(output_path, 'PNG')
            extracted_count += 1
            reports_with_images.add(composite_id)
            
        except Exception as e:
            error_count += 1
            if error_count <= 3:
                print(f"  Error extracting image for {composite_id}: {e}")
    
    print(f"  Total images: {extracted_count} (Skipped existing: {skipped_count}, New: {extracted_count - skipped_count})")
    return extracted_count, id_to_seq, seq_to_id, reports_with_images

def prepare_evaluation_csv(sample_df, parquet_df, img_dir, output_path, id_to_seq):
    id_to_cluster = {}
    for idx, query_row in sample_df.iterrows():
        query_id = query_row['query']
        repo_name = query_row['Repository_Name']
        
        if 'ground_truth_issues_with_images' in query_row and pd.notna(query_row.get('ground_truth_issues_with_images')):
            gt_ids = parse_id_list(query_row['ground_truth_issues_with_images'])
        else:
            gt_ids = parse_id_list(query_row['ground_truth'])
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
            parts = base_name.split('_')
            seq_idx = int(parts[-1])
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
    return df_eval, id_to_cluster

def run_semcluster_pipeline_optimized(eval_csv_path, img_dir, xml_dir, reports_with_images, seq_to_id, parquet_df, sample_df, query_to_valid_corpus, cache_file='feature_cache_filtered.pkl'):
    """Run SemCluster's feature extraction with caching"""
    print("\n" + "="*70)
    print("SEMCLUSTER FEATURE EXTRACTION (OPTIMIZED + CACHED)")
    print("="*70)
    
    if not img_dir.endswith('/'):
        img_dir = img_dir + '/'
    if not xml_dir.endswith('/'):
        xml_dir = xml_dir + '/'
    
    # Check cache
    st_pairs = {}
    ct_pairs = {}
    p_pairs = {}
    r_pairs = {}
    st_list = []
    ct_list = []
    p_list = []
    r_list = []
    
    cache_exists = False
    if os.path.exists(cache_file):
        print(f"  → Found cache file: {cache_file}")
        try:
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
                st_pairs = cache_data.get('st_pairs', {})
                p_pairs = cache_data.get('p_pairs', {})
                r_pairs = cache_data.get('r_pairs', {})
                st_list = cache_data.get('st_list', [])
                p_list = cache_data.get('p_list', [])
                r_list = cache_data.get('r_list', [])
                print(f"  → Loaded cached Structure ({len(st_pairs)} pairs) and Text ({len(p_pairs)} pairs) features")
                cache_exists = True
        except Exception as e:
            print(f"  ⚠ Error loading cache: {e}")
    
    n_with_images = len(reports_with_images)
    
    print("\n[1/2] Extracting image features...")
    
    # Always re-run image_main to get Content features (VGG16) with normalization
    # But we can inject the cached Structure features if available
    
    if n_with_images > 0:
        import numpy as np
        np.seterr(all='raise')
        
        # We need to run image_main, but we want to skip getSTdis if we have cached st_pairs
        # However, image_main calls both.
        # To properly optimize, we should call getCTdis directly if we have st_pairs.
        # But image_main also returns st_list/ct_list which are needed for clustering (though dummy values are used later).
        
        if cache_exists and st_pairs:
            print("  → Using cached Structure features, running Content features (Normalized VGG16)...")
            import image.content_feature
            ct_list, ct_pairs = image.content_feature.getCTdis(
                xml_dir, img_dir, eval_csv_path, sample_df, query_to_valid_corpus, seq_to_id, parquet_df
            )
            # st_list is already loaded from cache
        else:
            print("  → Computing Structure and Content features...")
            st_list, ct_list, st_pairs, ct_pairs = image_main.image_main(
                img_dir,
                eval_csv_path,
                xml_dir,
                sample_df,
                query_to_valid_corpus,
                seq_to_id,
                parquet_df
            )
    else:
        # No images
        header = list(range(1, 513))
        header.insert(0, 'index')
        st_list = [header]
        ct_list = [header]
        st_pairs = {}
        ct_pairs = {}
        for seq_idx in sorted(seq_to_id.keys()):
            zero_row = [seq_idx] + [0.0] * 512
            st_list.append(zero_row)
            ct_list.append(zero_row)

    print("\n[2/2] Extracting text features...")
    
    if cache_exists and p_pairs and r_pairs:
        print("  → Using cached Text features")
        # p_list, r_list already loaded
    else:
        print("  → Computing Text features...")
        all_reports_csv = eval_csv_path.replace('evaluation.csv', 'evaluation_all_reports.csv')
        rows = []
        for seq_idx in sorted(seq_to_id.keys()):
            composite_id = seq_to_id[seq_idx]
            if ':' in str(composite_id):
                repo_name, report_id_str = str(composite_id).split(':', 1)
                report_id = int(report_id_str)
                report_row = parquet_df[(parquet_df['id'] == report_id) & (parquet_df['repo_name'] == repo_name)]
            else:
                report_id = int(composite_id)
                report_row = parquet_df[parquet_df['id'] == report_id]
                
            if len(report_row) > 0:
                report_row = report_row.iloc[0]
                title = str(report_row.get('title', ''))
                description = str(report_row.get('description', ''))
                comments = str(report_row.get('comments', ''))
                full_text = f"Title:{title} Description:{description} Comments:{comments}"
                rows.append({
                    'index': seq_idx,
                    'description': full_text.replace('\n', ' ').replace(',', ';')[:1000]
                })
        
        pd.DataFrame(rows).to_csv(all_reports_csv, index=False)
        
        try:
            p_list, r_list, p_pairs, r_pairs = text_main.text_main(
                all_reports_csv,
                sample_df,
                query_to_valid_corpus,
                seq_to_id,
                parquet_df
            )
        except Exception as e:
            print(f"  ✗ CRITICAL: TextCNN feature extraction failed: {e}")
            raise RuntimeError(f"TextCNN extraction failed: {e}")

    # Save to cache (update with new ct_pairs if needed, though we don't cache ct_pairs usually as we are experimenting with it)
    # We cache ST, P, R pairs because they are stable. CT pairs change with VGG16 normalization.
    print(f"  → Updating cache file: {cache_file}")
    with open(cache_file, 'wb') as f:
        pickle.dump({
            'st_pairs': st_pairs,
            'p_pairs': p_pairs,
            'r_pairs': r_pairs,
            'st_list': st_list,
            'p_list': p_list,
            'r_list': r_list
        }, f)
    
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

def _combine_sparse_pair_distances(feature_data, reports_with_images):
    st_pairs = feature_data.get('st_pairs', {}) or {}
    ct_pairs = feature_data.get('ct_pairs', {}) or {}
    p_pairs = feature_data.get('p_pairs', {}) or {}
    r_pairs = feature_data.get('r_pairs', {}) or {}
    
    combined_pairs = {}
    all_keys = set()
    for pair_dict in (st_pairs, ct_pairs, p_pairs, r_pairs):
        all_keys.update(pair_dict.keys())
        
    for key in all_keys:
        repo_name, query_id, corpus_id = key
        
        st_val = st_pairs.get(key)
        ct_val = ct_pairs.get(key)
        p_val = p_pairs.get(key)
        r_val = r_pairs.get(key)
        
        if st_val is not None and p_val is not None:
            if st_val < 0.1 and p_val < 0.3:
                combined_pairs[key] = 0.0
                continue

        if ct_val is not None and r_val is not None:
            if ct_val > 0.7 and r_val > 0.8:
                combined_pairs[key] = 1.0
                continue

        contributions = []
        if query_id in reports_with_images:
            if st_val is not None:
                contributions.append(st_val)
            if ct_val is not None:
                contributions.append(ct_val)
        
        if p_val is not None:
            contributions.append(p_val)
        if r_val is not None:
            contributions.append(r_val)
            
        if contributions:
            combined_pairs[key] = round(sum(contributions) / len(contributions), 4)
            
    return combined_pairs

def avg_data_adaptive(feature_data, combined_pairs, seq_to_id, parquet_df):
    st_list = feature_data['st_list']
    header = list(st_list[0])
    seq_headers = [int(col) for col in header[1:]]
    
    repo_lookup = {}
    if parquet_df is not None and not parquet_df.empty:
        for _, row in parquet_df.iterrows():
            repo_name = row.get('repo_name', 'Unknown')
            if pd.isna(repo_name):
                repo_name = 'Unknown'
            repo_lookup[int(row['id'])] = str(repo_name)
    inferred_repo = {}
    for repo_name, query_id, _ in combined_pairs.keys():
        inferred_repo.setdefault(query_id, repo_name)
    
    def get_repo_name(report_id):
        repo_name = repo_lookup.get(report_id)
        if repo_name is None:
            repo_name = inferred_repo.get(report_id, 'Unknown')
        return repo_name
    
    avg_rows = [header]
    for seq_idx in seq_headers:
        row = [seq_idx]
        query_composite_id = seq_to_id.get(seq_idx, seq_idx)
        
        if isinstance(query_composite_id, str) and ':' in query_composite_id:
            q_repo, q_id_str = query_composite_id.split(':', 1)
            try:
                q_real_id = int(q_id_str)
            except ValueError:
                q_real_id = q_id_str
        else:
            q_repo = get_repo_name(query_composite_id)
            q_real_id = query_composite_id

        for target_seq in seq_headers:
            corpus_composite_id = seq_to_id.get(target_seq, target_seq)
            
            if isinstance(corpus_composite_id, str) and ':' in corpus_composite_id:
                c_repo, c_id_str = corpus_composite_id.split(':', 1)
                try:
                    c_real_id = int(c_id_str)
                except ValueError:
                    c_real_id = c_id_str
            else:
                c_real_id = corpus_composite_id

            if query_composite_id == corpus_composite_id:
                value = 0.0
            else:
                key_int = (q_repo, q_real_id, c_real_id)
                value = combined_pairs.get(key_int)
                
                if value is None:
                    key_comp = (q_repo, query_composite_id, corpus_composite_id)
                    value = combined_pairs.get(key_comp, np.nan)
            
            row.append(value)
        avg_rows.append(row)
    return avg_rows

def convert_to_similarity_matrix(feature_data, metric_name, seq_to_id, reports_with_images, parquet_df):
    combined_pairs = _combine_sparse_pair_distances(feature_data, reports_with_images)
    avg_list = avg_data_adaptive(feature_data, combined_pairs, seq_to_id, parquet_df)
    
    similarity_df = pd.DataFrame(data=avg_list[1:], columns=avg_list[0])
    
    new_columns = ['index']
    for col in similarity_df.columns[1:]:
        seq_idx = int(col)
        real_id = seq_to_id.get(seq_idx, str(seq_idx))
        new_columns.append(real_id)
    
    similarity_df.columns = new_columns
    
    similarity_df['index'] = similarity_df['index'].apply(lambda x: seq_to_id.get(int(x), str(x))).astype(str)
    renamed_columns = ['index'] + [str(col) for col in similarity_df.columns[1:]]
    similarity_df.columns = renamed_columns
    
    return similarity_df, combined_pairs

def create_ground_truth_dataframe(sample_df):
    rows = []
    for idx, query_row in sample_df.iterrows():
        query_id = query_row['query']
        repo_name = query_row['Repository_Name']
        if 'ground_truth_issues_with_images' in query_row and pd.notna(query_row.get('ground_truth_issues_with_images')):
            gt_ids = parse_id_list(query_row['ground_truth_issues_with_images'])
        else:
            gt_ids = parse_id_list(query_row['ground_truth'])
        duplicate_group = [query_id] + gt_ids
        cluster_label = min(duplicate_group)
        
        for rid in duplicate_group:
            composite_id = f"{repo_name}:{rid}"
            rows.append({'id': composite_id, 'group': cluster_label})
    
    gt_df = pd.DataFrame(rows)
    return gt_df

def generate_similarity_csv(similarity_df, ground_truth_df, parquet_df, output_path, query_to_repo, query_to_valid_corpus, top_k=None, combined_pairs=None):
    query_to_gt = {}
    for idx, query_row in ground_truth_df.iterrows():
        group_id = query_row['group']
        group_members = ground_truth_df[ground_truth_df['group'] == group_id]['id'].tolist()
        for qid in group_members:
            if qid not in query_to_gt:
                query_to_gt[qid] = set()
            query_to_gt[qid].update([m for m in group_members if m != qid])
    
    rows = []
    combined_pairs = combined_pairs or {}
    similarity_df = similarity_df.copy()
    similarity_df['index'] = similarity_df['index'].astype(str)
    similarity_df.columns = ['index'] + [str(col) for col in similarity_df.columns[1:]]
    similarity_lookup = similarity_df.set_index('index') if not similarity_df.empty else similarity_df
    
    for query_id in sorted(query_to_repo.keys()):
        query_project = query_to_repo.get(query_id, 'Unknown')
        query_index_key = f"{query_project}:{query_id}"

        if similarity_df.empty or query_index_key not in similarity_lookup.index:
            if str(query_id) in similarity_lookup.index:
                query_index_key = str(query_id)
            else:
                continue

        gt_set = query_to_gt.get(query_index_key, set())
        repo_query_key = (query_project, query_id)
        valid_corpus_tuples = query_to_valid_corpus.get(repo_query_key, set())

        query_row = None
        if not similarity_df.empty:
            try:
                query_row = similarity_lookup.loc[query_index_key]
                if isinstance(query_row, pd.DataFrame):
                    query_row = query_row.iloc[0]
            except KeyError:
                continue

        similarities = []
        for col in similarity_df.columns:
            if col == 'index':
                continue

            corpus_composite_id = str(col)
            try:
                if ':' in corpus_composite_id:
                    col_repo, col_id_str = corpus_composite_id.split(':', 1)
                    corpus_id = int(col_id_str)
                else:
                    col_repo = query_project
                    corpus_id = int(col)
            except ValueError:
                continue

            if corpus_id == query_id and col_repo == query_project:
                continue

            if (col_repo, corpus_id) not in valid_corpus_tuples:
                continue

            key = (query_project, query_id, corpus_id)
            distance = combined_pairs.get(key)
            if distance is None and query_row is not None:
                distance_value = query_row.get(corpus_composite_id)
                if isinstance(distance_value, pd.Series):
                    distance_value = distance_value.iloc[0]
                if pd.notna(distance_value):
                    distance = float(distance_value)
            if distance is None:
                continue
            similarities.append((corpus_composite_id, float(distance)))

        similarities.sort(key=lambda x: x[1], reverse=False)

        if top_k is not None:
            top_similarities = similarities[:top_k]
        else:
            top_similarities = similarities

        for rank, (corpus_composite_id, distance) in enumerate(top_similarities, start=1):
            c_is_gt = 1 if corpus_composite_id in gt_set else 0
            rows.append({
                'Project': query_project,
                'query': query_id,
                'corpus': corpus_composite_id,
                'score': round(distance, 6),
                'rank': rank,
                'c_is_gt': c_is_gt
            })
    
    result_columns = ['Project', 'query', 'corpus', 'score', 'rank', 'c_is_gt']
    result_df = pd.DataFrame(rows, columns=result_columns)
    result_df.to_csv(output_path, index=False)
    
    print(f"\n✓ Similarity matrix CSV saved to: {output_path}")
    return result_df

def run_clustering_and_evaluate(similarity_df, ground_truth_df, eval_csv_path):
    print("\n" + "="*70)
    print("CLUSTERING & EVALUATION")
    print("="*70)
    
    st_data = pd.DataFrame(data=[[0]], columns=['index'])
    ct_data = pd.DataFrame(data=[[0]], columns=['index'])
    p_data = pd.DataFrame(data=[[0]], columns=['index'])
    r_data = pd.DataFrame(data=[[0]], columns=['index'])
    
    print("\nRunning SemCluster clustering (K=2, max_iter=50)...")
    try:
        cluster_result = cluster.semi(
            eval_csv_path, 
            2,
            50,
            similarity_df,
            st_data, ct_data, p_data, r_data
        )
        print(f"  ✓ Clustering complete: {cluster_result}")
    except Exception as e:
        print(f"  ✗ Clustering failed: {e}")
        cluster_result = None
    
    print("\n" + "-"*70)
    print("RETRIEVAL METRICS")
    print("-"*70)
    
    mrr, map_score, hits_dict = calculate_retrieval_metrics(
        similarity_df, 
        ground_truth_df, 
        k_values=range(1, 11)
    )
    
    print(f"\nMean Reciprocal Rank (MRR): {mrr:.4f}")
    print(f"Mean Average Precision (MAP): {map_score:.4f}")
    print("\nHITS@k:")
    for k in sorted(hits_dict.keys()):
        print(f"  HITS@{k:2d}: {hits_dict[k]:.4f}")
    
    return cluster_result, mrr, map_score, hits_dict

def main():
    parser = argparse.ArgumentParser(description='SemCluster Evaluation (OPTIMIZED FILTERED)')
    parser.add_argument('--n-queries', type=int, default=-1,
                       help='Number of queries to process (-1 for all available)')
    args = parser.parse_args()
    
    print("="*70)
    print(f"SEMCLUSTER OPTIMIZED EVALUATION - FILTERED DATASET")
    print("="*70)
    
    ground_truth_csv = 'Dataset/Overall - FILTERED_trimmed_year_1_corpus_with_gt.csv'
    output_suffix = 'FILTERED_NORMALIZED'
    require_images = True
    
    parquet_file = 'Dataset/bug_reports_with_images.parquet'
    n_queries = args.n_queries
    
    img_dir = f'file/pic_file_parquet_filtered'
    xml_dir = f'file/xml_file_parquet_filtered'
    eval_csv = f'file/label_file_parquet/evaluation_filtered.csv'
    similarity_csv_path = f'output/semcluster_similarity_matrix_{output_suffix}.csv'
    cache_file = 'feature_cache_filtered.pkl'
    
    os.makedirs(os.path.dirname(eval_csv), exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(xml_dir, exist_ok=True)
    
    print(f"\n[STEP 1] Loading queries from {ground_truth_csv}...")
    sample_df = load_sample_queries(ground_truth_csv, n_queries=n_queries, min_duplicates=1, require_images=require_images)
    print(f"  Selected {len(sample_df)} queries")
    
    report_ids_by_repo = {}
    query_to_valid_corpus = {}

    for idx, row in sample_df.iterrows():
        query_id = row['query']
        repo_name = row['Repository_Name']

        if repo_name not in report_ids_by_repo:
            report_ids_by_repo[repo_name] = set()
        report_ids_by_repo[repo_name].add(query_id)

        gt_ids = parse_id_list(row.get('ground_truth_issues_with_images', '[]'))
        report_ids_by_repo[repo_name].update(gt_ids)

        corpus_ids = parse_id_list(row.get('corpus_issues_with_images', '[]'))
        report_ids_by_repo[repo_name].update(corpus_ids)

        query_key = (repo_name, query_id)
        query_to_valid_corpus[query_key] = set((repo_name, cid) for cid in corpus_ids)
    
    print(f"\n[STEP 2] Loading bug reports from {parquet_file}...")
    parquet_df = load_parquet_data(parquet_file, report_ids_by_repo)
    
    print(f"\n[STEP 3] Extracting images to {img_dir} (RESUME MODE)...")
    img_count, id_to_seq, seq_to_id, reports_with_images = extract_images_from_parquet_resume(parquet_df, img_dir)
    
    if img_count == 0:
        print("\n✗ No images extracted! Cannot proceed.")
        return
    
    print(f"\n[STEP 4] Creating evaluation CSV at {eval_csv}...")
    eval_df, id_to_cluster = prepare_evaluation_csv(sample_df, parquet_df, img_dir, eval_csv, id_to_seq)
    
    print(f"\n[STEP 5] Running SemCluster feature extraction (OPTIMIZED)...")
    feature_data = run_semcluster_pipeline_optimized(
        eval_csv,
        img_dir,
        xml_dir,
        reports_with_images,
        seq_to_id,
        parquet_df,
        sample_df,
        query_to_valid_corpus,
        cache_file=cache_file
    )
    
    required_keys = ('st_list', 'ct_list', 'p_list', 'r_list')
    if feature_data is None or any(feature_data.get(key) is None for key in required_keys):
        print("\n✗ Feature extraction failed! Cannot proceed to evaluation.")
        return
    
    print(f"\n[STEP 6] Creating similarity matrix...")
    similarity_df, combined_pairs = convert_to_similarity_matrix(
        feature_data,
        "Combined",
        seq_to_id,
        reports_with_images,
        parquet_df
    )
    
    print(f"\n[STEP 7] Preparing ground truth...")
    ground_truth_df = create_ground_truth_dataframe(sample_df)
    
    print(f"\n[STEP 8] Generating similarity matrix CSV...")
    query_to_repo = dict(zip(sample_df['query'], sample_df['Repository_Name']))
    
    similarity_result_df = generate_similarity_csv(
        similarity_df, 
        ground_truth_df, 
        parquet_df,
        similarity_csv_path,
        query_to_repo,
        query_to_valid_corpus,
        top_k=None,
        combined_pairs=combined_pairs
    )
    
    print(f"\n[STEP 9] Calculating retrieval metrics...")
    cluster_result, mrr, map_score, hits_dict = run_clustering_and_evaluate(
        similarity_df, 
        ground_truth_df, 
        eval_csv
    )
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)

if __name__ == '__main__':
    main()
