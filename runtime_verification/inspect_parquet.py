import pyarrow.parquet as pq
import pandas as pd

parquet_file = 'bug_reports_with_images.parquet'
table = pq.read_table(parquet_file)
df = table.to_pandas()

print(df[['repo_name', 'id']].head(10))
print(f"Unique repos: {df['repo_name'].unique()}")
