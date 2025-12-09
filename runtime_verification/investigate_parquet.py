#!/usr/bin/env python3

import pyarrow.parquet as pq
import pandas as pd

# Load parquet
print("Loading parquet file...")
table = pq.read_table('Dataset/bug_reports_with_images.parquet')
df = table.to_pandas()

print(f"\n{'='*70}")
print("PARQUET FILE STRUCTURE")
print(f"{'='*70}")
print(f"Total rows: {len(df)}")
print(f"Columns: {list(df.columns)}")
print(f"\nColumn data types:")
for col in df.columns:
    print(f"  {col}: {df[col].dtype}")

print(f"\n{'='*70}")
print("UNIQUE REPOSITORIES")
print(f"{'='*70}")
repo_counts = df['repo_name'].value_counts()
print(f"Total unique repositories: {len(repo_counts)}")
print(f"\nTop 10 repositories by bug report count:")
for repo, count in repo_counts.head(10).items():
    print(f"  {repo}: {count} reports")

print(f"\n{'='*70}")
print("AEGIS PROJECT ANALYSIS")
print(f"{'='*70}")
aegis_df = df[df['repo_name'].str.contains('aegis', case=False, na=False)]
print(f"Total Aegis reports: {len(aegis_df)}")

if len(aegis_df) > 0:
    print(f"\nAegis repository names:")
    for repo in aegis_df['repo_name'].unique():
        count = len(aegis_df[aegis_df['repo_name'] == repo])
        print(f"  {repo}: {count} reports")
    
    print(f"\nAegis ID range:")
    print(f"  Min ID: {aegis_df['id'].min()}")
    print(f"  Max ID: {aegis_df['id'].max()}")
    
    print(f"\nAegis reports with valid images:")
    valid_images = aegis_df[aegis_df['valid_image'] == True]
    print(f"  {len(valid_images)} out of {len(aegis_df)} ({len(valid_images)/len(aegis_df)*100:.1f}%)")
    
    print(f"\nSample Aegis IDs (first 30):")
    sample_ids = sorted(aegis_df['id'].unique())[:30]
    print(f"  {sample_ids}")
    
    print(f"\nLooking for specific queries [450, 772, 1085]:")
    for query_id in [450, 772, 1085]:
        matches = aegis_df[aegis_df['id'] == query_id]
        if len(matches) > 0:
            print(f"  Query {query_id}: FOUND - valid_image={matches.iloc[0]['valid_image']}")
        else:
            print(f"  Query {query_id}: NOT FOUND")
    
    # Check all Aegis IDs that have images
    print(f"\nAll Aegis IDs with valid_image=True:")
    aegis_with_images = aegis_df[aegis_df['valid_image'] == True]['id'].unique()
    print(f"  Total: {len(aegis_with_images)}")
    print(f"  IDs: {sorted(aegis_with_images)}")
else:
    print("No Aegis reports found in parquet file!")

print(f"\n{'='*70}")
print("IMAGE AVAILABILITY STATISTICS")
print(f"{'='*70}")
print(f"Reports with valid_image=True: {len(df[df['valid_image'] == True])}")
print(f"Reports with valid_image=False: {len(df[df['valid_image'] == False])}")
print(f"Percentage with valid images: {len(df[df['valid_image'] == True])/len(df)*100:.2f}%")

print(f"\n{'='*70}")
print("ID RANGE ANALYSIS")
print(f"{'='*70}")
print(f"Overall ID range: {df['id'].min()} to {df['id'].max()}")
print(f"\nID distribution by ranges:")
ranges = [(0, 500), (500, 1000), (1000, 1500), (1500, 2000), (2000, 5000), (5000, 10000), (10000, 15000)]
for start, end in ranges:
    count = len(df[(df['id'] >= start) & (df['id'] < end)])
    print(f"  {start:5d} - {end:5d}: {count:6d} reports")

# Check for duplicate IDs across different repos
print(f"\n{'='*70}")
print("DUPLICATE ID ANALYSIS")
print(f"{'='*70}")
id_counts = df.groupby('id').size()
duplicates = id_counts[id_counts > 1]
if len(duplicates) > 0:
    print(f"Found {len(duplicates)} IDs that appear in multiple repositories:")
    print(f"Sample duplicate IDs (first 10):")
    for id_val in duplicates.head(10).index:
        repos = df[df['id'] == id_val]['repo_name'].unique()
        print(f"  ID {id_val}: appears in {list(repos)}")
else:
    print("No duplicate IDs found across different repositories")
