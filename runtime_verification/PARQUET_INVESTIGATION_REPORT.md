# Parquet File Investigation Report

## Executive Summary

**ALL 3 AEGIS QUERIES (450, 772, 1085) ARE PRESENT IN THE PARQUET FILE WITH VALID IMAGES**

The issue with missing queries in your evaluation results is **NOT** due to missing data in the parquet file, but rather a problem in the evaluation script logic.

## Key Findings

### 1. Parquet File Overview
- **Total records**: 65,291 bug reports
- **Unique repositories**: 29
- **Records with valid images**: 9,547 (14.62%)
- **ID range**: 1 to 15,342

### 2. Aegis Project Data
- **Total Aegis reports**: 874
- **Aegis reports with valid images**: 90 (10.3%)
- **ID range**: 1 to 1,269

### 3. Critical Discovery: ALL Required Queries Exist
```
Query 450:  ✓ FOUND - valid_image=True
Query 772:  ✓ FOUND - valid_image=True  
Query 1085: ✓ FOUND - valid_image=True
```

All 90 Aegis IDs with valid images:
```
[17, 34, 35, 46, 50, 63, 91, 92, 96, 104, 123, 155, 175, 223, 231, 282, 287, 297, 
339, 343, 353, 354, 383, 397, 411, 413, 418, 422, 423, 436, 450, 453, 471, 475, 
479, 492, 495, 531, 560, 590, 635, 637, 656, 682, 713, 751, 758, 759, 765, 767, 
772, 783, 790, 805, 827, 842, 847, 848, 852, 862, 879, 880, 905, 906, 918, 982, 
985, 996, 1012, 1025, 1026, 1043, 1046, 1049, 1050, 1051, 1081, 1085, 1115, 1118, 
1141, 1149, 1176, 1189, 1209, 1213, 1219, 1221, 1257, 1268]
```

### 4. Major Issue: ID Collisions Across Repositories

**CRITICAL**: The parquet file has **8,124 duplicate IDs** across different repositories!

Examples:
- ID 1 appears in **19 different repositories**
- ID 450 appears in **Aegis** but may also appear in other repos
- ID 772 appears in **Aegis** but may also appear in other repos
- ID 1085 appears in **Aegis** but may also appear in other repos

This creates ambiguity when loading data by ID alone!

## Root Cause Analysis

The evaluation script in `run_parquet_evaluation.py` has these problematic sections:

1. **Load data by ID only** (line ~86):
```python
df_filtered = df[df['id'].isin(report_ids)].copy()
```

2. **Deduplication strategy** (line ~92):
```python
df_valid = df_filtered[df_filtered['valid_image'] == True].drop_duplicates(subset=['id'], keep='first')
```

**The Problem**: When the script loads IDs like `[450, 772, 1085]`, it might get:
- ID 450 from **Anki-Android** (first occurrence) instead of **Aegis**
- ID 772 from **AntennaPod** (first occurrence) instead of **Aegis**  
- ID 1085 from **Anki-Android** (first occurrence) instead of **Aegis**

The script uses `keep='first'`, so it takes whichever repository appears first in the parquet file for duplicate IDs. This explains why you're seeing queries from the wrong project!

## Expected vs Actual Behavior

### Expected (from GT file):
- Aegis project
- Queries: 450, 772, 1085 (all from Aegis)

### Actual (from similarity CSV):
- Query 1085 is processed
- But queries 450 and 772 are missing
- Corpus IDs like 719, 905, 1025, 1043 appear (these might be from other repos with duplicate IDs)

## Repository Statistics

Top 10 repositories by bug report count:
1. cgeo: 8,508 reports
2. Anki-Android: 7,283 reports
3. thunderbird-android: 4,259 reports
4. AntennaPod: 3,877 reports
5. StreetComplete: 3,775 reports
6. focus-android: 3,768 reports
7. apps-android-commons: 2,746 reports
8. Slide: 2,696 reports
9. collect: 2,661 reports
10. openfoodfacts-androidapp: 2,409 reports

(Aegis has only 874 reports, much smaller than these projects)

## Recommended Solutions

### Solution 1: Filter by Repository Name (RECOMMENDED)
Modify the data loading to filter by BOTH repository name AND ID:

```python
def load_parquet_data(parquet_path, report_ids, repo_name=None):
    table = pq.read_table(parquet_path)
    df = table.to_pandas()
    
    # Filter by both repo and ID to avoid collisions
    if repo_name:
        df = df[df['repo_name'] == repo_name]
    
    df_filtered = df[df['id'].isin(report_ids)].copy()
    # ... rest of function
```

### Solution 2: Update Ground Truth File Format
Add repository name to the GT CSV:
```csv
Repository_Name,query,query_has_image,ground_truth,...
Aegis,450,TRUE,[418| 121| ...],...
```

Then ensure the script uses this to filter correctly.

### Solution 3: Use Composite Keys
Create composite keys (repo_name, id) throughout the evaluation pipeline to ensure uniqueness.

## Impact Assessment

This bug affects:
- ✗ FILTERED evaluation (currently producing wrong results)
- ✗ FULL evaluation (likely also affected)
- ✗ Any project with low bug report counts (likely to have IDs stolen by larger projects)
- ✓ Similarity calculations (once correct data is loaded, these should work)
- ✓ Feature extraction (works correctly on whatever data it receives)

## Next Steps

1. **Immediate**: Fix `load_parquet_data()` to filter by repository name
2. **Verify**: Re-run FILTERED evaluation and confirm all 3 Aegis queries appear
3. **Validate**: Check other small projects (andOTP, KISS, etc.) for similar issues
4. **Long-term**: Consider restructuring parquet to use globally unique IDs or composite keys
