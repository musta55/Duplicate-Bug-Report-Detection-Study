import image.structure_feature
import image.content_feature
import os
import pandas as pd
import tensorflow as tf
import gc


def get_gpu_memory_info():
    """Get GPU memory usage information"""
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            # Get memory info from nvidia-smi
            import subprocess
            result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', 
                                   '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                total_used = sum(int(line.split(',')[0]) for line in lines)
                total_mem = sum(int(line.split(',')[1]) for line in lines)
                return total_used, total_mem
    except:
        pass
    return None, None


def image_main(pic_dir, label_csv, xml_dir, sample_df, query_to_valid_corpus, seq_to_id, parquet_df):
    """
    Extract image features - OPTIMIZED to only compute query-corpus pairs.
    
    Args:
        sample_df: DataFrame with query information
        query_to_valid_corpus: Dict mapping (repo, query_id) -> set of (repo, corpus_id) tuples
        seq_to_id: Mapping from sequential index to real report ID
        parquet_df: DataFrame with repo_name for ID lookup
    """
    import sys
    print("  → [DEBUG] Entered image_main function")
    sys.stdout.flush()
    
    print("  → Starting OPTIMIZED structure feature extraction...")
    print("  → Computing only query-corpus pairs (not full N×N matrix)")
    sys.stdout.flush()
    
    # Check GPU memory before starting
    mem_used, mem_total = get_gpu_memory_info()
    if mem_used and mem_total and mem_total > 0:
        print(f"  → GPU memory: {mem_used}MB / {mem_total}MB ({mem_used/mem_total*100:.1f}% used)")
        sys.stdout.flush()
    
    # Check image count
    pic_list = sorted([f for f in os.listdir(pic_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    num_images = len(pic_list)
    print(f"  → Processing {num_images} images")
    sys.stdout.flush()
    
    # Force garbage collection before starting
    gc.collect()
    if tf.config.list_physical_devices('GPU'):
        tf.keras.backend.clear_session()
    
    print("  → [DEBUG] About to call getSTdis...")
    sys.stdout.flush()
    st_list, st_pairs = image.structure_feature.getSTdis(pic_dir, label_csv, xml_dir, sample_df, query_to_valid_corpus, seq_to_id, parquet_df)
    print(f"  → [DEBUG] getSTdis returned {len(st_list)} rows")
    print("  → Structure features done, starting content features...")
    sys.stdout.flush()
    
    # Clear GPU memory between structure and content
    gc.collect()
    if tf.config.list_physical_devices('GPU'):
        tf.keras.backend.clear_session()
    
    print("  → [DEBUG] About to call getCTdis...")
    sys.stdout.flush()
    ct_list, ct_pairs = image.content_feature.getCTdis(xml_dir, pic_dir, label_csv, sample_df, query_to_valid_corpus, seq_to_id, parquet_df)
    print(f"  → [DEBUG] getCTdis returned {len(ct_list)} rows")
    print("  → Content features done!")
    sys.stdout.flush()
    
    # Final cleanup
    gc.collect()
    
    return st_list, ct_list, st_pairs, ct_pairs
