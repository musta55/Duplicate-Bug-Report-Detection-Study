import sys
import numpy as np

import text.text_feature as rfe


def del_invalid(p_list, r_list):
    for i in range(1, len(p_list)):
        del p_list[i][1]
    for i in range(1, len(r_list)):
        del r_list[i][1]
    del (p_list[1])
    del (r_list[1])
    return p_list, r_list


def text_main(label_csv, sample_df=None, query_to_valid_corpus=None, seq_to_id=None, parquet_df=None):
    """
    Extract text features - OPTIMIZED to only compute query-corpus pairs.
    
    Args:
        label_csv: Path to evaluation CSV
        sample_df: DataFrame with query information
        query_to_valid_corpus: Dict mapping (repo, query_id) -> set of (repo, corpus_id) tuples
        seq_to_id: Mapping from sequential index to real report ID
        parquet_df: DataFrame with repo_name for ID lookup
    """
    np.set_printoptions(threshold=sys.maxsize)

    # report feature extraction & dis matrix calculation
    text_feature_list = rfe.extract_report_feature(label_csv)
    p_list, r_list, p_pairs, r_pairs = rfe.cal_sim_matrix(
        label_csv,
        text_feature_list,
        sample_df,
        query_to_valid_corpus,
        seq_to_id,
        parquet_df
    )
    p_list, r_list = del_invalid(p_list, r_list)

    return p_list, r_list, p_pairs, r_pairs
