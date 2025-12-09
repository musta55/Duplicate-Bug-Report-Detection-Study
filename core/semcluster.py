import pandas as pd
import numpy as np
from . import cluster
import image.image_main as img
import text.text_main as txt


def avg_data(ST, CT, P, R):
    row, col = len(ST), len(ST[0])
    AVG = [ST[0]]
    for i in range(1, row):
        a_list = [ST[i][0]]
        for j in range(1, col):
            data1 = float(ST[i][j])
            data2 = float(CT[i][j])
            data3 = float(P[i][j])
            data4 = float(R[i][j])
            output_data = float(format((data1 + data2 + data3 + data4) / 4, '.4f'))
            a_list.append(output_data)
        AVG.append(a_list)
    return AVG


def calculate_retrieval_metrics(similarity_df, ground_truth_df, k_values=range(1, 11)):
    """
    Calculates MRR, MAP, and HITS@k for the given similarity matrix.
    By default, computes HITS@k for k=1, 2, 3, ..., 10.
    """
    report_ids = ground_truth_df['id'].unique()
    ground_truth = {group: list(ids) for group, ids in ground_truth_df.groupby('group')['id']}

    reciprocal_ranks = []
    average_precisions = []
    hits_at_k = {k: 0 for k in k_values}

    for query_id in report_ids:
        # Get true duplicates for the current query
        true_duplicates = []
        for group_ids in ground_truth.values():
            if query_id in group_ids:
                true_duplicates = [pid for pid in group_ids if pid != query_id]
                break

        if not true_duplicates:
            continue

        # Get ranked list of candidates from similarity matrix
        # Ensure the query_id is a string for matching the 'index' column
        ranked_list_series = similarity_df[similarity_df['index'] == str(query_id)]
        if ranked_list_series.empty:
            continue

        ranked_list = ranked_list_series.iloc[0, 1:].sort_values(ascending=True)
        ranked_ids = [col for col in ranked_list.index if col != str(query_id)]

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

    mrr = np.mean(reciprocal_ranks) if reciprocal_ranks else 0
    map_score = np.mean(average_precisions) if average_precisions else 0

    # Filter for queries that had duplicates to get a fair HITS score
    total_queries_with_duplicates = sum(
        1 for query_id in report_ids if any(query_id in g and len(g) > 1 for g in ground_truth.values()))
    hits_scores = {k: (v / total_queries_with_duplicates) if total_queries_with_duplicates > 0 else 0 for k, v in
                   hits_at_k.items()}

    return mrr, map_score, hits_scores


def debug_retrieval(similarity_df, ground_truth_df, top_n=5):
    """Print per-query top-N ranked items with distances and mark true duplicates.
    Useful to sanity-check perfect metrics on tiny datasets.
    """
    report_ids = ground_truth_df['id'].unique()
    # Map group -> ids
    gt_map = {group: list(ids) for group, ids in ground_truth_df.groupby('group')['id']}

    # Ensure index is string in similarity df
    if similarity_df['index'].dtype != object:
        similarity_df['index'] = similarity_df['index'].astype(str)

    print("\nTop-{} neighbors per query (lower = more similar):".format(top_n))
    for qid in report_ids:
        # List of true duplicates for this qid
        true_dups = []
        for g_ids in gt_map.values():
            if qid in g_ids:
                true_dups = [pid for pid in g_ids if pid != qid]
                break
        row = similarity_df[similarity_df['index'] == str(qid)]
        if row.empty:
            print(f"- Query {qid}: no row in similarity matrix")
            continue
        ranked = row.iloc[0, 1:].sort_values(ascending=True)
        # Build top list
        items = []
        for col, val in ranked.items():
            # Handle composite IDs (strings)
            pid = str(col)
            if pid == str(qid):
                continue
            items.append((pid, float(val), (pid in true_dups)))
        print(f"- Query {qid}: true_dups={true_dups}")
        for pid, val, is_true in items[:top_n]:
            mark = "<-- true" if is_true else ""
            print(f"    {pid:>6}  dist={val:.4f} {mark}")


if __name__ == '__main__':
    print("STARTING...")
    # Define file paths
    pic_directory = "file/pic_file/"
    label_file_demo = "file/label_file/demo.csv"
    xml_directory = "file/xml_file/"
    label = "file/label_file/evaluate.csv"  # This is the correct variable name

    # Get data from image and text processing modules
    st_list, ct_list = img.image_main(pic_directory, label_file_demo, xml_directory)
    p_list, r_list = txt.text_main(label_file_demo)

    # Create DataFrames directly, using the first row as column headers.
    p_data = pd.DataFrame(data=p_list[1:], columns=p_list[0])
    r_data = pd.DataFrame(data=r_list[1:], columns=r_list[0])
    st_data = pd.DataFrame(data=st_list[1:], columns=st_list[0])
    ct_data = pd.DataFrame(data=ct_list[1:], columns=ct_list[0])

    # Calculate the averaged data and create its DataFrame
    all_list = avg_data(st_list, ct_list, p_list, r_list)
    all_data = pd.DataFrame(data=all_list[1:], columns=all_list[0])

    print("FINISHING CLUSTERING...")
    type_dict = cluster.semi(label, 2, 50, all_data, st_data, ct_data, p_data, r_data)
    print("result:", type_dict)

    print("\nCALCULATING RETRIEVAL METRICS...")

    # FIX: Read the full CSV and select only the first and last columns for ground truth
    full_ground_truth_df = pd.read_csv(label, header=None)
    ground_truth_df = pd.DataFrame({
        'id': full_ground_truth_df.iloc[:, 0],  # First column is the ID
        'group': full_ground_truth_df.iloc[:, -1]  # Last column is the group
    })

    # Ensure 'index' column is string type for proper lookup
    all_data['index'] = all_data['index'].astype(str)

    mrr, map_score, hits_scores = calculate_retrieval_metrics(all_data, ground_truth_df)

    print("\n--- Retrieval Metrics ---")
    print(f"MRR: {mrr:.4f}")
    print(f"MAP: {map_score:.4f}")
    print("HITS@k:")
    for k in sorted(hits_scores.keys()):
        print(f"  K={k:2d}: {hits_scores[k]:.4f}")

    # Optional: print per-query rankings for sanity check
    debug_retrieval(all_data, ground_truth_df, top_n=5)