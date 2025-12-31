import pandas as pd
import argparse
from core.metrics import Metrics

def calculate_projectwise_metrics(sim_matrix_csv, gt_csv, output_csv):
    sim_df = pd.read_csv(sim_matrix_csv)
    gt_df = pd.read_csv(gt_csv)
    metrics = Metrics()
    projects = sim_df['Project'].unique()
    results = []
    summary = []

    for project in projects:
        project_df = sim_df[sim_df['Project'] == project].copy()
        gt_proj_df = gt_df[gt_df['Repository_Name'] == project]
        queries = project_df['query'].unique()

        # Identify queries with images using ground truth
        queries_with_image = set(gt_proj_df[gt_proj_df['query_has_image'] == True]['query'])
        # Text Only: queries without images
        text_only_df = project_df[~project_df['query'].isin(queries_with_image)].copy()
        # Text+Image: queries with images
        text_img_df = project_df[project_df['query'].isin(queries_with_image)].copy()

        # Prepare for metrics
        text_only_df = text_only_df.rename(columns={'query': 'q_id'})
        text_img_df = text_img_df.rename(columns={'query': 'q_id'})

        # Calculate metrics for both modes
        mrr_text = metrics.computeMRR(text_only_df, 'rank') if not text_only_df.empty else 0.0
        mrr_img = metrics.computeMRR(text_img_df, 'rank') if not text_img_df.empty else 0.0
        recall1_text = metrics.computeRecall_K(text_only_df, 1, 'rank') if not text_only_df.empty else 0.0
        recall1_img = metrics.computeRecall_K(text_img_df, 1, 'rank') if not text_img_df.empty else 0.0
        recall5_text = metrics.computeRecall_K(text_only_df, 5, 'rank') if not text_only_df.empty else 0.0
        recall5_img = metrics.computeRecall_K(text_img_df, 5, 'rank') if not text_img_df.empty else 0.0
        recall10_text = metrics.computeRecall_K(text_only_df, 10, 'rank') if not text_only_df.empty else 0.0
        recall10_img = metrics.computeRecall_K(text_img_df, 10, 'rank') if not text_img_df.empty else 0.0

        # Calculate differences and RI
        mrr_diff = mrr_img - mrr_text
        mrr_ri = ((mrr_img - mrr_text) / mrr_text * 100) if mrr_text else 0.0
        recall1_diff = recall1_img - recall1_text
        recall5_diff = recall5_img - recall5_text
        recall10_diff = recall10_img - recall10_text

        results.append({
            'Repository': project,
            '#queries': len(queries),
            '#queries with image': len(queries_with_image),
            'MRR_Text Only': f"{mrr_text:.4f}",
            'MRR_Text + Image': f"{mrr_img:.4f}",
            'MRR_Diff Img-Text': f"{mrr_diff:.4f}",
            'MRR_RI': f"{mrr_ri:.2f}%",
            'Recall@1_Text Only': f"{recall1_text:.4f}",
            'Recall@1_Text + Image': f"{recall1_img:.4f}",
            'Recall@1_Diff Img-Text': f"{recall1_diff:.4f}",
            'Recall@5_Text Only': f"{recall5_text:.4f}",
            'Recall@5_Text + Image': f"{recall5_img:.4f}",
            'Recall@5_Diff Img-Text': f"{recall5_diff:.4f}",
            'Recall@10_Text Only': f"{recall10_text:.4f}",
            'Recall@10_Text + Image': f"{recall10_img:.4f}",
            'Recall@10_Diff Img-Text': f"{recall10_diff:.4f}",
        })
        summary.append({'Repository': project, '#queries': len(queries)})

    # Print summary for verification
    print("\nProject count:", len(results))
    print("Repository\t#queries")
    for row in summary:
        print(f"{row['Repository']}\t{row['#queries']}")

    # Custom column order and header
    columns = [
        'Repository', '#queries', '#queries with image',
        'MRR_Text Only', 'MRR_Text + Image', 'MRR_Diff Img-Text', 'MRR_RI',
        'Recall@1_Text Only', 'Recall@1_Text + Image', 'Recall@1_Diff Img-Text',
        'Recall@5_Text Only', 'Recall@5_Text + Image', 'Recall@5_Diff Img-Text',
        'Recall@10_Text Only', 'Recall@10_Text + Image', 'Recall@10_Diff Img-Text',
    ]
    df_out = pd.DataFrame(results)[columns]

    # Compute overall (weighted by #queries) and average (unweighted mean) rows
    metric_cols = [
        'MRR_Text Only', 'MRR_Text + Image', 'MRR_Diff Img-Text', 'MRR_RI',
        'Recall@1_Text Only', 'Recall@1_Text + Image', 'Recall@1_Diff Img-Text',
        'Recall@5_Text Only', 'Recall@5_Text + Image', 'Recall@5_Diff Img-Text',
        'Recall@10_Text Only', 'Recall@10_Text + Image', 'Recall@10_Diff Img-Text',
    ]
    # Convert metric columns to float (except RI, which is % string)
    df_metrics = df_out.copy()
    for col in metric_cols:
        if 'RI' not in col:
            df_metrics[col] = df_metrics[col].astype(float)
    # Weighted overall (by #queries)
    total_queries = df_out['#queries'].astype(int).sum()
    overall = {'Repository': 'Overall', '#queries': total_queries, '#queries with image': '',}
    for col in metric_cols:
        if 'RI' in col:
            # Compute RI for overall
            text_col = col.replace('_RI', '_Text Only')
            img_col = col.replace('_RI', '_Text + Image')
            text_sum = (df_metrics[text_col] * df_out['#queries'].astype(int)).sum()
            img_sum = (df_metrics[img_col] * df_out['#queries'].astype(int)).sum()
            text_avg = text_sum / total_queries if total_queries else 0.0
            img_avg = img_sum / total_queries if total_queries else 0.0
            ri = ((img_avg - text_avg) / text_avg * 100) if text_avg else 0.0
            overall[col] = f"{ri:.2f}%"
        else:
            val = (df_metrics[col] * df_out['#queries'].astype(int)).sum() / total_queries if total_queries else 0.0
            overall[col] = f"{val:.4f}"
    # Unweighted average
    avg = {'Repository': 'Average', '#queries': total_queries, '#queries with image': '',}
    for col in metric_cols:
        if 'RI' in col:
            text_col = col.replace('_RI', '_Text Only')
            img_col = col.replace('_RI', '_Text + Image')
            text_mean = df_metrics[text_col].mean()
            img_mean = df_metrics[img_col].mean()
            ri = ((img_mean - text_mean) / text_mean * 100) if text_mean else 0.0
            avg[col] = f"{ri:.2f}%"
        else:
            val = df_metrics[col].mean()
            avg[col] = f"{val:.4f}"

    # Append rows
    df_out = pd.concat([df_out, pd.DataFrame([overall, avg])], ignore_index=True)
    df_out.to_csv(output_csv, index=False)
    print(f"Projectwise metrics matrix saved to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate projectwise metrics matrix for FULL or FILTERED dataset using ground truth.")
    parser.add_argument('--sim_matrix', type=str, required=True, help='Input similarity matrix CSV')
    parser.add_argument('--gt', type=str, required=True, help='Ground truth CSV')
    parser.add_argument('--output', type=str, required=True, help='Output CSV file path')
    args = parser.parse_args()
    calculate_projectwise_metrics(args.sim_matrix, args.gt, args.output)
