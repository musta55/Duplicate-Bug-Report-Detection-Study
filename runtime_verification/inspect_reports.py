import pandas as pd
import pyarrow.parquet as pq

parquet_file = 'Dataset/bug_reports_with_images.parquet'
table = pq.read_table(parquet_file)
df = table.to_pandas()

ids_to_check = [450, 418, 339, 423]
subset = df[df['id'].isin(ids_to_check) & (df['repo_name'] == 'Aegis')]

for _, row in subset.iterrows():
    print(f"\n--- ID: {row['id']} ---")
    print(f"Title: {row['title']}")
    print(f"Description: {row['description'][:200]}...")
