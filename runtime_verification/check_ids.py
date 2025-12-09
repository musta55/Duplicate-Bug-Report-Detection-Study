import os
import pandas as pd
import pyarrow.parquet as pq

PARQUET_FILE = 'Dataset/bug_reports_with_images.parquet'
XML_DIR = 'file/xml_file_parquet_full/'

print("Reading Parquet...")
table = pq.read_table(PARQUET_FILE)
df = table.to_pandas()
parquet_ids = set(df['id'].unique())
print(f"Parquet IDs: {len(parquet_ids)}")

print("Reading XML directory...")
xml_files = os.listdir(XML_DIR)
xml_ids = set()
for f in xml_files:
    if f.endswith('.xml') and 'layout' in f:
        try:
            fid = int(f.replace('layout', '').replace('.xml', ''))
            xml_ids.add(fid)
        except:
            pass
print(f"XML IDs: {len(xml_ids)}")

intersection = parquet_ids.intersection(xml_ids)
print(f"Intersection: {len(intersection)}")
