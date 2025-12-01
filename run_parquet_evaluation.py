#!/usr/bin/env python3
"""
SemCluster Evaluation using Parquet Dataset

This script:
1. Loads queries from FILTERED/FULL CSV with ground truth
2. Extracts images and text from parquet file
3. Runs SemCluster's feature extraction and clustering
4. Evaluates duplicate detection performance (MRR, MAP, HITS@k)

Supports:
- FILTERED dataset: Queries with images only (92 reports)
- FULL dataset: All queries including text-only (2323 reports)
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from io import BytesIO

# Import SemCluster modules
import cluster
import image.image_main as image_main
import text.text_main as text_main
from main import calculate_retrieval_metrics, debug_retrieval


def load_sample_queries(csv_path, n_queries=10, min_duplicates=2, require_images=True):
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
            n_queries = len(df_filtered)
        sample_df = df_filtered.sample(n=n_queries, random_state=42)
    
    return sample_df


def parse_id_list(id_string):
    """Parse '[id1| id2| id3]' format into list of integers"""
    if pd.isna(id_string) or id_string == '[]':
        return []
    cleaned = id_string.strip('[]').strip()
    if not cleaned:
        return []
    ids = [int(x.strip()) for x in cleaned.split('|') if x.strip()]
    return ids


def load_parquet_data(parquet_path, report_ids_by_repo):
    """Load specific bug reports from parquet file
    
    Args:
        parquet_path: Path to parquet file
        report_ids_by_repo: Dict mapping repo_name -> set of IDs, e.g. {'Aegis': {450, 772, 1085}, ...}
    """
    table = pq.read_table(parquet_path)
    df = table.to_pandas()
    
    # Load reports by filtering on BOTH repo_name AND id to avoid ID collisions
    dfs = []
    for repo_name, ids in report_ids_by_repo.items():
        # Filter by repository name AND IDs
        repo_df = df[(df['repo_name'] == repo_name) & (df['id'].isin(ids))].copy()
        dfs.append(repo_df)
    
    if not dfs:
        print(f"  ✗ No reports found in parquet!")
        return pd.DataFrame()
    
    df_filtered = pd.concat(dfs, ignore_index=True)
    
    # Report statistics
    total_requested = sum(len(ids) for ids in report_ids_by_repo.values())
    total_found = len(df_filtered)
    with_images = len(df_filtered[df_filtered['valid_image'] == True])
    
    print(f"  Loaded {total_found}/{total_requested} reports from parquet")
    print(f"  Reports with valid images: {with_images}/{total_found}")
    
    if total_found < total_requested:
        # Report which IDs are missing
        missing_by_repo = {}
        for repo_name, requested_ids in report_ids_by_repo.items():
            found_ids = set(df_filtered[df_filtered['repo_name'] == repo_name]['id'])
            missing = requested_ids - found_ids
            if missing:
                missing_by_repo[repo_name] = missing
        
        if missing_by_repo:
            print(f"  ⚠ Missing reports:")
            for repo, missing_ids in missing_by_repo.items():
                print(f"    {repo}: {sorted(list(missing_ids)[:10])}{'...' if len(missing_ids) > 10 else ''}")
    
    return df_filtered


def extract_images_from_parquet(parquet_df, output_dir):
    """Extract images from parquet binary data to disk"""
    # Clean directory first to avoid mixing with previous runs
    if os.path.exists(output_dir):
        import shutil
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    extracted_count = 0
    missing_count = 0
    error_count = 0
    
    # Create sequential index mapping
    id_to_seq = {}
    seq_to_id = {}
    reports_with_images = set()  # Track which reports have images
    
    for seq_idx, (idx, row) in enumerate(parquet_df.iterrows()):
        report_id = row['id']
        repo_name = row['repo_name']
        # Use composite key to avoid ID collisions across projects
        composite_id = f"{repo_name}:{report_id}"
        
        id_to_seq[composite_id] = seq_idx
        seq_to_id[seq_idx] = composite_id
        
        image_data = row.get('image')
        
        if pd.isna(image_data) or image_data is None:
            missing_count += 1
            continue
        
        # Image is stored as dict with 'bytes' and 'path' fields
        if isinstance(image_data, dict):
            image_bytes = image_data.get('bytes')
        else:
            image_bytes = image_data
        
        if image_bytes is None or len(image_bytes) == 0:
            missing_count += 1
            continue
        
        try:
            # Load image from binary data
            img = Image.open(BytesIO(image_bytes))
            
            # Convert CMYK to RGB if needed
            if img.mode == 'CMYK':
                img = img.convert('RGB')
            
            # Save with SEQUENTIAL index for SemCluster compatibility
            # Format: report_img_SEQ.png where SEQ is 0-30
            output_path = os.path.join(output_dir, f"report_img_{seq_idx}.png")
            img.save(output_path, 'PNG')
            extracted_count += 1
            reports_with_images.add(composite_id)  # Track this report has an image
            
        except Exception as e:
            error_count += 1
            if error_count <= 3:  # Only print first few errors
                print(f"  Error extracting image for {composite_id}: {e}")
    
    print(f"  Extracted {extracted_count} images, {missing_count} missing, {error_count} errors")
    return extracted_count, id_to_seq, seq_to_id, reports_with_images


def prepare_evaluation_csv(sample_df, parquet_df, img_dir, output_path, id_to_seq):
    """
    Create CSV file for SemCluster evaluation.
    Format: index,description,img_url,appid,useless,tag
    One row per IMAGE FILE (critical for structure_feature.py)
    Uses sequential indices for CSV index column
    """
    # Build ID-to-cluster mapping from ground truth
    # Use ground_truth_issues_with_images for FILTERED, ground_truth for FULL
    id_to_cluster = {}
    for idx, query_row in sample_df.iterrows():
        query_id = query_row['query']
        repo_name = query_row['Repository_Name']
        
        # Check which column exists - FILTERED has ground_truth_issues_with_images
        if 'ground_truth_issues_with_images' in query_row and pd.notna(query_row.get('ground_truth_issues_with_images')):
            gt_ids = parse_id_list(query_row['ground_truth_issues_with_images'])
        else:
            gt_ids = parse_id_list(query_row['ground_truth'])
        duplicate_group = [query_id] + gt_ids
        cluster_label = min(duplicate_group)  # Use smallest ID as cluster label
        for rid in duplicate_group:
            # Use composite key
            composite_id = f"{repo_name}:{rid}"
            id_to_cluster[composite_id] = cluster_label
    
    # Get list of actual image files (must match what structure_feature.py will iterate)
    image_files = sorted([f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    # Create text content map
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
    
    # Create reverse mapping for sequential to real ID
    seq_to_id = {v: k for k, v in id_to_seq.items()}
    
    # Create one row per image file with SEQUENTIAL index matching filename
    rows = []
    for img_file in image_files:
        # Extract sequential index from filename (format: report_img_SEQ.png)
        base_name = img_file.rsplit('.', 1)[0]
        if '_' in base_name:
            # Format: report_img_SEQ -> take last part
            parts = base_name.split('_')
            seq_idx = int(parts[-1])
        else:
            seq_idx = int(base_name)
        
        # Get real report ID from sequence (now a composite string)
        composite_id = seq_to_id.get(seq_idx, str(seq_idx))
        
        cluster = id_to_cluster.get(composite_id, -1) # Default to -1 if not in GT
        description = content_map.get(composite_id, '').replace('\n', ' ').replace(',', ';')[:500]
        
        rows.append({
            'index': seq_idx,  # Use sequential index for SemCluster compatibility
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
    """Run SemCluster's feature extraction - OPTIMIZED to only compute query-corpus pairs"""
    print("\n" + "="*70)
    print("SEMCLUSTER FEATURE EXTRACTION (OPTIMIZED)")
    print("="*70)
    
    # Ensure directories end with / as expected by SemCluster code
    if not img_dir.endswith('/'):
        img_dir = img_dir + '/'
    if not xml_dir.endswith('/'):
        xml_dir = xml_dir + '/'
    
    n_with_images = len(reports_with_images)
    n_without_images = len(seq_to_id) - len(reports_with_images)
    
    # Image features (only for reports with images)
    print("\n[1/2] Extracting image features...")
    print(f"  Reports with images: {n_with_images}")
    print(f"  Reports without images: {n_without_images}")
    
    st_pairs = {}
    ct_pairs = {}

    if n_with_images > 0:
        # Configure numpy to raise exceptions on warnings
        import numpy as np
        np.seterr(all='raise')
        
        try:
            # Extract features only for reports with images AND only for query-corpus pairs
            st_list, ct_list, st_pairs, ct_pairs = image_main.image_main(
                img_dir,
                eval_csv_path,
                xml_dir,
                sample_df,
                query_to_valid_corpus,
                seq_to_id,
                parquet_df
            )
            print(f"  ✓ Structure features extracted: {len(st_list)-1} reports")
            print(f"  ✓ Content features extracted: {len(ct_list)-1} reports")
        except Exception as e:
            import traceback
            print(f"  ⚠ Image feature extraction failed: {type(e).__name__}: {e}")
            print(f"  → Traceback:")
            traceback.print_exc()
            print(f"  → Creating zero image features for reports with images")
            # Create zero features for reports that should have images
            header = list(range(1, 513))
            header.insert(0, 'index')
            st_list = [header]
            ct_list = [header]
            st_pairs = {}
            ct_pairs = {}
            # Only create features for reports that have images
            for seq_idx in sorted(seq_to_id.keys()):
                report_id = seq_to_id[seq_idx]
                if report_id in reports_with_images:
                    zero_row = [seq_idx] + [0.0] * 512
                    st_list.append(zero_row)
                    ct_list.append(zero_row)
        
        # Add zero features for reports WITHOUT images
        if n_without_images > 0:
            print(f"  → Adding zero image features for {n_without_images} reports without images")
            header = st_list[0]
            n_features = len(header) - 1
            
            # Build set of sequential IDs that already have features
            seq_ids_with_features = set(row[0] for row in st_list[1:])
            
            # Add zero rows for reports without images
            for seq_idx in sorted(seq_to_id.keys()):
                if seq_idx not in seq_ids_with_features:
                    report_id = seq_to_id[seq_idx]
                    if report_id not in reports_with_images:
                        zero_row = [seq_idx] + [0.0] * n_features
                        st_list.append(zero_row)
                        ct_list.append(zero_row)
            
            # Re-sort by sequential ID
            st_list = [st_list[0]] + sorted(st_list[1:], key=lambda x: x[0])
            ct_list = [ct_list[0]] + sorted(ct_list[1:], key=lambda x: x[0])
            print(f"  ✓ Total image features: {len(st_list)-1} reports ({n_with_images} real, {n_without_images} zeros)")
    else:
        # No images at all - create zero features for all reports
        print(f"  → No images available, creating zero image features for all {len(seq_to_id)} reports")
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
    
    # Text features - extract for ALL reports (with and without images)
    print("\n[2/2] Extracting text features...")
    print(f"  Processing text for all {len(seq_to_id)} reports...")
    
    # Create a CSV with ALL reports (not just those with images) for text extraction
    all_reports_csv = eval_csv_path.replace('evaluation.csv', 'evaluation_all_reports.csv')
    
    # Build text content for all reports
    rows = []
    for seq_idx in sorted(seq_to_id.keys()):
        composite_id = seq_to_id[seq_idx]
        # Parse composite ID "Repo:ID"
        if ':' in str(composite_id):
            repo_name, report_id_str = str(composite_id).split(':', 1)
            report_id = int(report_id_str)
            # Find report in parquet matching BOTH repo and ID
            report_row = parquet_df[(parquet_df['id'] == report_id) & (parquet_df['repo_name'] == repo_name)]
        else:
            # Fallback for legacy IDs
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
    print(f"  ✓ Created text CSV with {len(rows)} reports")
    
    # TextCNN feature extraction (GPU-accelerated) - OPTIMIZED to only compute query-corpus pairs
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
        print(f"  → TextCNN is REQUIRED for GPU-based evaluation")
        print(f"  → Check: vocab file exists, model checkpoint valid")
        raise RuntimeError(f"TextCNN extraction failed - cannot proceed without text features: {e}")
    
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
    """Merge sparse per-pair distances across modalities, respecting image availability.
    Applies 'Can Link' / 'Cannot Link' constraints from SemCluster logic.
    """
    st_pairs = feature_data.get('st_pairs', {}) or {}
    ct_pairs = feature_data.get('ct_pairs', {}) or {}
    p_pairs = feature_data.get('p_pairs', {}) or {}
    r_pairs = feature_data.get('r_pairs', {}) or {}
    
    # Note: We do NOT apply dynamic Min-Max normalization here because the feature extractors
    # already normalize to [0,1] (or close to it). Dynamic normalization distorts the 
    # absolute thresholds used in the constraints (e.g. 0.1 becomes 0.5 if range is small).
    
    combined_pairs = {}
    all_keys = set()
    for pair_dict in (st_pairs, ct_pairs, p_pairs, r_pairs):
        all_keys.update(pair_dict.keys())
        
    for key in all_keys:
        repo_name, query_id, corpus_id = key
        
        # Get raw normalized values (or None if missing)
        st_val = st_pairs.get(key)
        ct_val = ct_pairs.get(key)
        p_val = p_pairs.get(key)
        r_val = r_pairs.get(key)
        
        # --- APPLY SEMCLUSTER CONSTRAINTS ---
        # Must Link (Duplicate): ST < 0.1 AND P < 0.3
        if st_val is not None and p_val is not None:
            if st_val < 0.1 and p_val < 0.3:
                combined_pairs[key] = 0.0  # Force minimum distance (max similarity)
                continue

        # Cannot Link (Non-Duplicate): CT > 0.7 AND R > 0.8
        if ct_val is not None and r_val is not None:
            if ct_val > 0.7 and r_val > 0.8:
                combined_pairs[key] = 1.0  # Force maximum distance (min similarity)
                continue
        # ------------------------------------

        contributions = []
        
        # Image features (only if query has images)
        if query_id in reports_with_images:
            if st_val is not None:
                contributions.append(st_val)
            if ct_val is not None:
                contributions.append(ct_val)
        
        # Text features (always available)
        if p_val is not None:
            contributions.append(p_val)
        if r_val is not None:
            contributions.append(r_val)
            
        if contributions:
            combined_pairs[key] = round(sum(contributions) / len(contributions), 4)
            
    return combined_pairs


def avg_data_adaptive(feature_data, combined_pairs, seq_to_id, parquet_df):
    """Average sparse distances into a dense matrix indexed by sequential IDs."""
    st_list = feature_data['st_list']
    header = list(st_list[0])
    seq_headers = [int(col) for col in header[1:]]
    
    # Build repo lookup from parquet (fallback to unknown)
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
        
        # Parse composite ID to get repo and real ID
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
            
            # Parse corpus composite ID
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
                # Try both composite and integer keys
                # The feature extractors might use (Repo, IntID, IntID)
                key_int = (q_repo, q_real_id, c_real_id)
                value = combined_pairs.get(key_int)
                
                if value is None:
                    # Try with composite IDs just in case
                    key_comp = (q_repo, query_composite_id, corpus_composite_id)
                    value = combined_pairs.get(key_comp, np.nan)
            
            row.append(value)
        avg_rows.append(row)
    return avg_rows


def convert_to_similarity_matrix(feature_data, metric_name, seq_to_id, reports_with_images, parquet_df):
    """Convert sparse per-pair distances to a dense similarity DataFrame."""
    combined_pairs = _combine_sparse_pair_distances(feature_data, reports_with_images)
    print(f"  → Averaging sparse distance dictionaries ({len(combined_pairs)} computed pairs)...")
    avg_list = avg_data_adaptive(feature_data, combined_pairs, seq_to_id, parquet_df)
    
    # Convert to DataFrame with sequential indices first
    similarity_df = pd.DataFrame(data=avg_list[1:], columns=avg_list[0])
    
    # Map column names from sequential to real IDs (except 'index')
    new_columns = ['index']  # Keep 'index' as first column name
    for col in similarity_df.columns[1:]:  # Skip 'index' column
        seq_idx = int(col)
        # real_id is now a composite string "repo:id"
        real_id = seq_to_id.get(seq_idx, str(seq_idx))
        new_columns.append(real_id)
    
    similarity_df.columns = new_columns
    
    # Also map the first column values (report IDs) from sequential to real and keep as strings
    similarity_df['index'] = similarity_df['index'].apply(lambda x: seq_to_id.get(int(x), str(x))).astype(str)
    renamed_columns = ['index'] + [str(col) for col in similarity_df.columns[1:]]
    similarity_df.columns = renamed_columns
    
    print(f"\n{metric_name} similarity matrix: {len(similarity_df)} x {len(similarity_df.columns)-1}")
    return similarity_df, combined_pairs


def create_ground_truth_dataframe(sample_df):
    """Create ground truth DataFrame for evaluation"""
    rows = []
    for idx, query_row in sample_df.iterrows():
        query_id = query_row['query']
        repo_name = query_row['Repository_Name']
        # Use ground_truth_issues_with_images for FILTERED, ground_truth for FULL
        if 'ground_truth_issues_with_images' in query_row and pd.notna(query_row.get('ground_truth_issues_with_images')):
            gt_ids = parse_id_list(query_row['ground_truth_issues_with_images'])
        else:
            gt_ids = parse_id_list(query_row['ground_truth'])
        duplicate_group = [query_id] + gt_ids
        cluster_label = min(duplicate_group)
        
        for rid in duplicate_group:
            # Use composite ID to match similarity matrix
            composite_id = f"{repo_name}:{rid}"
            rows.append({'id': composite_id, 'group': cluster_label})
    
    gt_df = pd.DataFrame(rows)
    print(f"\nGround truth: {len(gt_df)} reports in {gt_df['group'].nunique()} duplicate groups")
    return gt_df


def generate_similarity_csv(similarity_df, ground_truth_df, parquet_df, output_path, query_to_repo, query_to_valid_corpus, top_k=None, combined_pairs=None):
    """
    Generate similarity matrix CSV in the required format:
    Project | query | corpus | score | rank | c_is_gt
    
    Args:
        query_to_repo: Dict mapping query_id -> repo_name
        query_to_valid_corpus: Dict mapping (repo, query_id) -> set of (repo, corpus_id) tuples
        top_k: Number of top similar results to keep per query (None = all results)
    """
    # Create ground truth mapping: composite query_id -> set of composite ground truth corpus IDs
    query_to_gt = {}
    for idx, query_row in ground_truth_df.iterrows():
        # Get all IDs in the same group
        group_id = query_row['group']
        group_members = ground_truth_df[ground_truth_df['group'] == group_id]['id'].tolist()
        # For each member, store the others as ground truth (all composite IDs)
        for qid in group_members:
            if qid not in query_to_gt:
                query_to_gt[qid] = set()
            query_to_gt[qid].update([m for m in group_members if m != qid])
    
    rows = []
    combined_pairs = combined_pairs or {}
    missing_pairs = []
    similarity_df = similarity_df.copy()
    similarity_df['index'] = similarity_df['index'].astype(str)
    similarity_df.columns = ['index'] + [str(col) for col in similarity_df.columns[1:]]
    similarity_lookup = similarity_df.set_index('index') if not similarity_df.empty else similarity_df
    
    # For each ACTUAL query report (not ground truth IDs)
    for query_id in sorted(query_to_repo.keys()):
        # Get project/repo name for this query
        query_project = query_to_repo.get(query_id, 'Unknown')

        # Construct composite key for lookup
        query_index_key = f"{query_project}:{query_id}"

        if similarity_df.empty or query_index_key not in similarity_lookup.index:
            # Try legacy lookup (just ID) if composite fails
            if str(query_id) in similarity_lookup.index:
                query_index_key = str(query_id)
            else:
                print(f"  ⚠ Skipping query {query_id} ({query_index_key}): no similarity row present")
                continue

        # Get ground truth set for this query (composite IDs)
        gt_set = query_to_gt.get(query_index_key, set())

        # Get valid corpus for this query using (repo, query_id) key
        repo_query_key = (query_project, query_id)
        valid_corpus_tuples = query_to_valid_corpus.get(repo_query_key, set())

        # Get similarity row for this query (fallback only)
        query_row = None
        if not similarity_df.empty:
            try:
                query_row = similarity_lookup.loc[query_index_key]
                if isinstance(query_row, pd.DataFrame):
                    query_row = query_row.iloc[0]
            except KeyError:
                print(f"  ⚠ Similarity lookup missing for query {query_id} (repo {query_project}); skipping")
                continue

        # Collect all corpus similarities (excluding self, only from valid corpus)
        similarities = []
        for col in similarity_df.columns:
            if col == 'index':
                continue

            # col is now "repo:id"
            corpus_composite_id = str(col)
            try:
                if ':' in corpus_composite_id:
                    col_repo, col_id_str = corpus_composite_id.split(':', 1)
                    corpus_id = int(col_id_str)
                else:
                    col_repo = query_project # Assumption? Or skip?
                    corpus_id = int(col)
            except ValueError:
                continue

            if corpus_id == query_id and col_repo == query_project:
                continue

            # FILTER: Only include corpus IDs that are in the valid corpus for this query
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
                missing_pairs.append(key)
                continue
            similarities.append((corpus_composite_id, float(distance)))

        # Sort by distance (ascending - lower distance = more similar)
        similarities.sort(key=lambda x: x[1], reverse=False)

        # Keep only top K results (or all if top_k is None)
        if top_k is not None:
            top_similarities = similarities[:top_k]
        else:
            top_similarities = similarities

        # Add ranked results with ground truth indicator
        for rank, (corpus_composite_id, distance) in enumerate(top_similarities, start=1):
            c_is_gt = 1 if corpus_composite_id in gt_set else 0
            if rank == 1 or c_is_gt:
                print(f"DEBUG: Query {query_index_key}, Corpus {corpus_composite_id}, Distance {distance}, c_is_gt {c_is_gt}, GT set: {list(gt_set)}")
            rows.append({
                'Project': query_project,
                'query': query_id,
                'corpus': corpus_composite_id,
                'score': round(distance, 6),
                'rank': rank,
                'c_is_gt': c_is_gt
            })
    
    if missing_pairs:
        missing_sample = missing_pairs[:5]
        print(f"  ⚠ Missing distances for {len(missing_pairs)} query-corpus pairs (skipped). Example: {missing_sample}")
    
    # Create DataFrame and save
    # Ensure downstream code always sees the expected columns even if rows is empty
    result_columns = ['Project', 'query', 'corpus', 'score', 'rank', 'c_is_gt']
    result_df = pd.DataFrame(rows, columns=result_columns)
    result_df.to_csv(output_path, index=False)
    
    print(f"\n✓ Similarity matrix CSV saved to: {output_path}")
    total_pairs = len(result_df)
    unique_queries = result_df['query'].nunique() if total_pairs else 0
    print(f"  Total query-corpus pairs: {total_pairs}")
    print(f"  Unique queries: {unique_queries}")
    print(f"  Top-K per query: {'ALL' if top_k is None else top_k}")
    if total_pairs > 0:
        print(f"  Score range: {result_df['score'].min():.4f} - {result_df['score'].max():.4f}")
    else:
        print("  ⚠ No valid query-corpus rows were produced. Check filtering criteria or similarity inputs.")
    
    return result_df


def run_clustering_and_evaluate(similarity_df, ground_truth_df, eval_csv_path):
    """Run SemCluster clustering and evaluate results"""
    print("\n" + "="*70)
    print("CLUSTERING & EVALUATION")
    print("="*70)
    
    # Convert to component DataFrames (for cluster.semi)
    st_data = pd.DataFrame(data=[[0]], columns=['index'])  # Dummy - not used in clustering
    ct_data = pd.DataFrame(data=[[0]], columns=['index'])
    p_data = pd.DataFrame(data=[[0]], columns=['index'])
    r_data = pd.DataFrame(data=[[0]], columns=['index'])
    
    # Run SemCluster's semi-supervised clustering
    print("\nRunning SemCluster clustering (K=2, max_iter=50)...")
    try:
        cluster_result = cluster.semi(
            eval_csv_path, 
            2,  # K clusters
            50,  # max iterations
            similarity_df,  # Combined similarity matrix
            st_data, ct_data, p_data, r_data  # Component matrices (not used)
        )
        print(f"  ✓ Clustering complete: {cluster_result}")
    except Exception as e:
        print(f"  ✗ Clustering failed: {e}")
        cluster_result = None
    
    # Evaluate retrieval performance
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
    
    # Debug output
    print("\n" + "-"*70)
    print("SAMPLE RANKINGS (Top-5 per query)")
    print("-"*70)
    debug_retrieval(similarity_df, ground_truth_df, top_n=5)
    
    return cluster_result, mrr, map_score, hits_dict


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='SemCluster Evaluation')
    parser.add_argument('--dataset', type=str, default='FILTERED', 
                       choices=['FILTERED', 'FULL'],
                       help='Dataset to evaluate: FILTERED (images only) or FULL (all queries)')
    parser.add_argument('--n-queries', type=int, default=-1,
                       help='Number of queries to process (-1 for all available)')
    parser.add_argument('--query-id', type=int, default=None,
                       help='Specific query ID to process (overrides n-queries)')
    args = parser.parse_args()
    
    print("="*70)
    print(f"SEMCLUSTER PARQUET EVALUATION - {args.dataset} DATASET")
    print("="*70)
    
    # Configuration based on dataset type
    if args.dataset == 'FILTERED':
        ground_truth_csv = 'Overall - FILTERED_trimmed_year_1_corpus_with_gt.csv'
        output_suffix = 'FILTERED'
        require_images = True
        print("  Mode: FILTERED - Queries with images only")
    else:  # FULL
        ground_truth_csv = 'Overall - FULL_trimmed_year_1_corpus_with_gt.csv'
        output_suffix = 'FULL'
        require_images = False
        print("  Mode: FULL - All queries (image + text-only)")
    
    parquet_file = '/home/mhasan02/SemCluster-v2/bug_reports_with_images.parquet'
    n_queries = args.n_queries if args.n_queries is not None else -1
    
    # Output directories
    img_dir = f'file/pic_file_parquet_{args.dataset.lower()}'
    xml_dir = f'file/xml_file_parquet_{args.dataset.lower()}'
    eval_csv = f'file/label_file_parquet/evaluation_{args.dataset.lower()}.csv'
    similarity_csv_path = f'semcluster_similarity_matrix_{output_suffix}.csv'
    
    # Clean directories to avoid stale data
    if os.path.exists(img_dir):
        import shutil
        shutil.rmtree(img_dir)
    if os.path.exists(xml_dir):
        import shutil
        shutil.rmtree(xml_dir)

    os.makedirs(os.path.dirname(eval_csv), exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(xml_dir, exist_ok=True)
    
    # Step 1: Load sample queries
    print(f"\n[STEP 1] Loading queries from {ground_truth_csv}...")
    sample_df = load_sample_queries(ground_truth_csv, n_queries=n_queries, min_duplicates=1, require_images=require_images)
    print(f"  Selected {len(sample_df)} queries")
    if len(sample_df) <= 50:
        print(f"  Query IDs: {sorted(sample_df['query'].tolist())}")
    
    # Filter by specific query ID if requested
    if args.query_id is not None:
        print(f"  Filtering for specific query ID: {args.query_id}")
        sample_df = sample_df[sample_df['query'] == args.query_id]
        if len(sample_df) == 0:
            print(f"  ✗ Query {args.query_id} not found in dataset!")
            return
    
    # Collect all needed report IDs grouped by repository
    # Also collect valid corpus IDs for each query (using repo+id tuples to avoid collisions)
    report_ids_by_repo = {}  # repo_name -> set of IDs
    query_to_valid_corpus = {}  # (repo, query_id) -> set of (repo, corpus_id) tuples

    for idx, row in sample_df.iterrows():
        query_id = row['query']
        repo_name = row['Repository_Name']

        # Add query ID to needed reports
        if repo_name not in report_ids_by_repo:
            report_ids_by_repo[repo_name] = set()
        report_ids_by_repo[repo_name].add(query_id)

        # Add ground truth IDs
        if require_images:
            # FILTERED: Column L
            gt_ids = parse_id_list(row.get('ground_truth_issues_with_images', '[]'))
        else:
            # FULL: Column E
            gt_ids = parse_id_list(row.get('ground_truth', '[]'))

        report_ids_by_repo[repo_name].update(gt_ids)

        # Add corpus IDs
        if require_images:
            # FILTERED: Column M
            corpus_ids = parse_id_list(row.get('corpus_issues_with_images', '[]'))
        else:
            # FULL: Column D
            corpus_ids = parse_id_list(row.get('corpus', '[]'))

        report_ids_by_repo[repo_name].update(corpus_ids)

        # Store valid corpus for this query as (repo, id) tuples
        query_key = (repo_name, query_id)
        query_to_valid_corpus[query_key] = set((repo_name, cid) for cid in corpus_ids)
    

    
    total_ids = sum(len(ids) for ids in report_ids_by_repo.values())
    print(f"  Total unique reports needed: {total_ids}")
    print(f"  Repositories: {len(report_ids_by_repo)}")
    for repo, ids in sorted(report_ids_by_repo.items()):
        print(f"    {repo}: {len(ids)} reports")
    
    # DEBUG: Check query_to_valid_corpus
    print(f"  → DEBUG: query_to_valid_corpus has {len(query_to_valid_corpus)} entries")
    if len(query_to_valid_corpus) > 0:
        sample_key = list(query_to_valid_corpus.keys())[0]
        sample_corpus = query_to_valid_corpus[sample_key]
        print(f"  → DEBUG: Sample query {sample_key} has {len(sample_corpus)} corpus reports")
    
    # Step 2: Load data from parquet
    print(f"\n[STEP 2] Loading bug reports from {parquet_file}...")
    parquet_df = load_parquet_data(parquet_file, report_ids_by_repo)
    
    # Step 3: Extract images
    print(f"\n[STEP 3] Extracting images to {img_dir}...")
    img_count, id_to_seq, seq_to_id, reports_with_images = extract_images_from_parquet(parquet_df, img_dir)
    
    if img_count == 0:
        print("\n✗ No images extracted! Cannot proceed.")
        return
    
    # Step 4: Prepare evaluation CSV
    print(f"\n[STEP 4] Creating evaluation CSV at {eval_csv}...")
    eval_df, id_to_cluster = prepare_evaluation_csv(sample_df, parquet_df, img_dir, eval_csv, id_to_seq)
    
    # Step 5: Run SemCluster pipeline
    print(f"\n[STEP 5] Running SemCluster feature extraction...")
    feature_data = run_semcluster_pipeline(
        eval_csv,
        img_dir,
        xml_dir,
        reports_with_images,
        seq_to_id,
        parquet_df,
        sample_df,
        query_to_valid_corpus
    )
    
    required_keys = ('st_list', 'ct_list', 'p_list', 'r_list')
    if feature_data is None or any(feature_data.get(key) is None for key in required_keys):
        print("\n✗ Feature extraction failed! Cannot proceed to evaluation.")
        return
    
    # Step 6: Create similarity matrix
    print(f"\n[STEP 6] Creating similarity matrix...")
    similarity_df, combined_pairs = convert_to_similarity_matrix(
        feature_data,
        "Combined",
        seq_to_id,
        reports_with_images,
        parquet_df
    )
    
    # Step 7: Create ground truth
    print(f"\n[STEP 7] Preparing ground truth...")
    ground_truth_df = create_ground_truth_dataframe(sample_df)
    
    # Step 8: Generate similarity matrix CSV (main output)
    print(f"\n[STEP 8] Generating similarity matrix CSV...")
    # Extract actual query IDs from sample_df
    query_to_repo = dict(zip(sample_df['query'], sample_df['Repository_Name']))
    
    similarity_result_df = generate_similarity_csv(
        similarity_df, 
        ground_truth_df, 
        parquet_df,
        similarity_csv_path,
        query_to_repo,  # Pass mapping instead of set
        query_to_valid_corpus,  # Pass valid corpus mapping
        top_k=None,  # Keep ALL similarities for full matrix (needed for clustering)
        combined_pairs=combined_pairs
    )
    
    # Step 9: Calculate retrieval metrics
    print(f"\n[STEP 9] Calculating retrieval metrics...")
    cluster_result, mrr, map_score, hits_dict = run_clustering_and_evaluate(
        similarity_df, 
        ground_truth_df, 
        eval_csv
    )
    
    # Final summary
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)
    print(f"Queries evaluated: {len(sample_df)}")
    print(f"Total unique reports: {len(seq_to_id)}")
    print(f"Images extracted: {img_count}")
    print(f"\nOutput file: {similarity_csv_path}")
    print(f"  Format: Project | query | corpus | score | rank")
    print(f"  Total rows: {len(similarity_result_df)}")
    print(f"\nRetrieval Performance:")
    print(f"  MRR: {mrr:.4f}")
    print(f"  MAP: {map_score:.4f}")
    print(f"  HITS@1:  {hits_dict.get(1, 0):.4f}")
    print(f"  HITS@5:  {hits_dict.get(5, 0):.4f}")
    print(f"  HITS@10: {hits_dict.get(10, 0):.4f}")
    print("="*70)


if __name__ == '__main__':
    main()
