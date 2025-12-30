
import pandas as pd
import numpy as np
import argparse
from core.metrics import Metrics

def calculate_projectwise_metrics(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    metrics = Metrics()
    projects = df['Project'].unique()
    results = []

    for project in projects:
        project_df = df[df['Project'] == project].copy()
        queries = project_df['query'].unique()


        # Text Only: SF and CF are missing, empty, or 0
        text_only_df = project_df[
            (project_df['SF'].isnull() | (project_df['SF'] == '') | (project_df['SF'] == 0) | (project_df['SF'] == 0.0)) &
            (project_df['CF'].isnull() | (project_df['CF'] == '') | (project_df['CF'] == 0) | (project_df['CF'] == 0.0))
        ].copy()

        # Text+Image: all BB, RS, SF, CF are present (not null/empty), regardless of value
        text_img_df = project_df[
            project_df[['BB', 'RS', 'SF', 'CF']].notnull().all(axis=1) &
            (project_df[['BB', 'RS', 'SF', 'CF']] != '').all(axis=1)
        ].copy()

        # For queries with image, count unique queries in text_img_df
        queries_with_image = text_img_df['query'].nunique()

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

        results.append({
            'Repository': project,
            '#queries': len(queries),
            '#queries with image': queries_with_image,
            'MRR_Text': mrr_text,
            'MRR_Img': mrr_img,
            'MRR_Diff': mrr_img - mrr_text,
            'MRR_RI': (mrr_img - mrr_text) / mrr_text if mrr_text else 0,
            'Recall@1_Text': recall1_text,
            'Recall@1_Img': recall1_img,
            'Recall@1_Diff': recall1_img - recall1_text,
            'Recall@5_Text': recall5_text,
            'Recall@5_Img': recall5_img,
            'Recall@5_Diff': recall5_img - recall5_text,
            'Recall@10_Text': recall10_text,
            'Recall@10_Img': recall10_img,
            'Recall@10_Diff': recall10_img - recall10_text,
        })

    df_out = pd.DataFrame(results)
    # Format float columns to 4 decimal points, except percentage column
    float_cols = [col for col in df_out.columns if df_out[col].dtype == float and col != 'MRR_RI']
    df_out[float_cols] = df_out[float_cols].applymap(lambda x: f"{x:.4f}")
    # Format percentage columns
    df_out['MRR_RI'] = df_out['MRR_RI'].apply(lambda x: f"{float(x)*100:.2f}%" if isinstance(x, (float, int)) else x)
    df_out.to_csv(output_csv, index=False)
    print(f"Project-wise metrics matrix saved to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate projectwise metrics matrix.")
    parser.add_argument('--input', type=str, required=True, help='Input CSV file path')
    parser.add_argument('--output', type=str, required=True, help='Output CSV file path')
    args = parser.parse_args()
    calculate_projectwise_metrics(args.input, args.output)
