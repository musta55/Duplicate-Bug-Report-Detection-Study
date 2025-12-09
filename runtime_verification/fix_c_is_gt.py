import pandas as pd
import ast


def parse_ground_truth(gt_str):
    gt_str = gt_str.strip('[]')
    return set(int(x.strip()) for x in gt_str.split('|') if x.strip().isdigit())

gt_file = 'Dataset/Overall - FILTERED_trimmed_year_1_corpus_with_gt.csv'
sim_file = 'output/semcluster_similarity_matrix_FILTERED.csv'
out_file = 'output/semcluster_similarity_matrix_FILTERED_with_gt.csv'

gt_df = pd.read_csv(gt_file)
# Map: (query as int) -> set of ground truth corpus IDs (from ground_truth column only)
query_gt_map = {}
for idx, row in gt_df.iterrows():
    try:
        query_id = int(str(row['query']).strip())
        gt_str = str(row['ground_truth_issues_with_images'])
        gt_set = parse_ground_truth(gt_str)
        query_gt_map[query_id] = gt_set
    except Exception:
        continue

# Read similarity matrix
sim_df = pd.read_csv(sim_file)


# Add c_is_gt column: match query and corpus to ground truth

def is_gt(row):
    try:
        query_id = int(str(row['query']).strip())
        corpus_id = int(str(row['corpus']).strip())
    except ValueError:
        return 0
    gt_set = query_gt_map.get(query_id, set())
    return 1 if corpus_id in gt_set else 0

sim_df['c_is_gt'] = sim_df.apply(is_gt, axis=1)

# Save to new file
sim_df.to_csv(out_file, index=False)
print(f'Output written to {out_file}')
