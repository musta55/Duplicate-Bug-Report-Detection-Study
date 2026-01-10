# SemCluster Evaluation Scripts Guide

## Overview

There are three evaluation scripts in this project. Use them based on your needs:

## Scripts

### 1. `run_parquet_evaluation.py` ⭐ **PRIMARY/RECOMMENDED**

**Use this for**: Complete evaluation from scratch using parquet data

**Features:**
- ✅ Full pipeline: parquet → images → features → embeddings → similarities → metrics
- ✅ Most reliable and tested
- ✅ Generates all intermediate files (images, XMLs, CSVs)
- ✅ Direct sparse pair computation (efficient)
- ✅ Applies SemCluster constraints (Can Link / Cannot Link)

**Usage:**
```bash
# FILTERED dataset (queries with images only)
python run_parquet_evaluation.py --dataset FILTERED

# FULL dataset (all queries, adaptive fusion)
python run_parquet_evaluation.py --dataset FULL

# Process specific number of queries
python run_parquet_evaluation.py --dataset FILTERED --n-queries 10

# Process specific query ID
python run_parquet_evaluation.py --dataset FILTERED --query-id 450
```

**Output:**
- `output/semcluster_similarity_matrix_FILTERED.csv` (or FULL)
- `file/pic_file_parquet_filtered/` (extracted images)
- `file/xml_file_parquet_filtered/` (UI layout XMLs)
- `file/label_file_parquet/evaluation_filtered.csv`

---

### 2. `run_evaluation_from_embeddings.py` ⚡ **FAST PATH**

**Use this for**: Quick evaluation when you already have embeddings

**Features:**
- ✅ Loads pre-computed embeddings from pickle files
- ✅ Much faster (skips image/text extraction)
- ✅ Requires pickle files: `embeddings/text_embeddings.pkl`, `structure_embeddings.pkl`, `content_embeddings.pkl`
- ✅ Multiprocessing for parallel computation

**Usage:**
```bash
# Requires existing pickle files in embeddings/
python run_evaluation_from_embeddings.py --dataset FILTERED
python run_evaluation_from_embeddings.py --dataset FULL
```

**Prerequisites:**
You must have pre-generated embeddings:
- `embeddings/text_embeddings.pkl`
- `embeddings/structure_embeddings.pkl`
- `embeddings/content_embeddings.pkl`

**Note:** If you don't have embeddings, use `run_parquet_evaluation.py` instead.

---

### 3. `run_evaluation.py` ⚠️ **EXPERIMENTAL/MERGED**

**Status**: Experimental - attempts to auto-detect and switch between full/fast modes

**Features:**
- Checks if embeddings exist
- Auto-switches between full pipeline and fast path
- `--force-rebuild` flag to regenerate embeddings

**Issues:**
- May have pickle format compatibility issues
- Similarity scores might be incorrect
- **Not recommended for production use**

**If you want to use it:**
```bash
python run_evaluation.py --dataset FILTERED
python run_evaluation.py --dataset FILTERED --force-rebuild
```

---

## Recommended Workflow

### For Regular Evaluation (Most Common)
```bash
# Use the primary script - always works
python run_parquet_evaluation.py --dataset FILTERED
```

### For Fast Re-evaluation (If you have embeddings)
```bash
# Use fast path with pre-computed embeddings
python run_evaluation_from_embeddings.py --dataset FILTERED
```

### To Generate Embeddings for Future Fast Runs
Currently, embeddings are generated during the full pipeline but not saved as pickle files by `run_parquet_evaluation.py`. If you need pickle files for the fast path, you would need to:

1. Run full pipeline: `python run_parquet_evaluation.py --dataset FILTERED`
2. Manually convert outputs to pickle format (future feature)

---

## Output Format

All scripts generate the same output format:

**`output/semcluster_similarity_matrix_FILTERED.csv`:**
```csv
Project,query,corpus,score,rank,c_is_gt,BB,RS,SF,CF
Aegis,450,Aegis:339,0.083735,1,0,0.031293,0.219914,0.0,
Aegis,450,Aegis:423,0.090616,2,0,0.036529,0.235319,0.0,
...
```

**Columns:**
- `Project`: Repository name
- `query`: Query bug report ID
- `corpus`: Candidate bug report ID (composite format: "Repo:ID")
- `score`: Combined similarity score (lower = more similar)
- `rank`: Rank of this candidate for the query
- `c_is_gt`: 1 if corpus is ground truth duplicate, 0 otherwise
- `BB`: Bag of Words (problem description) distance
- `RS`: Recurrence Sequence (procedure steps) distance
- `SF`: Structure Features (UI layout) distance
- `CF`: Content Features (visual appearance) distance

---

## Troubleshooting

### All scores are 1.0
- **Cause**: Embeddings not loaded correctly or normalization stats failed
- **Solution**: Use `run_parquet_evaluation.py` (primary script)

### Component scores (BB/RS/SF/CF) are empty
- **Cause**: Feature extraction failed or embeddings missing
- **Solution**: Use `run_parquet_evaluation.py` with `--dataset FILTERED`

### "CRITICAL: TextCNN feature extraction failed"
- **Check**: TextCNN model files exist in `text/text_feature_extraction/runs/TextCNN_model/`
- **Check**: Vocabulary file exists
- **Check**: CUDA/GPU available for TensorFlow

### Missing images (all SF/CF = 0.0)
- **Expected for FULL dataset**: Only ~10-12% of reports have images
- **For FILTERED dataset**: All queries should have images
- **Check**: Parquet file contains image data

---

## Performance Comparison

| Script | Mode | Time (FILTERED, 125 queries) | Requires |
|--------|------|------------------------------|----------|
| `run_parquet_evaluation.py` | Full | ~15-30 minutes | Parquet data |
| `run_evaluation_from_embeddings.py` | Fast | ~2-5 minutes | Pickle files |
| `run_evaluation.py` | Auto | Varies | Either |

---

## Recommendation

**Always use `run_parquet_evaluation.py` unless you have a specific reason to use the fast path.**

The full pipeline is:
- ✅ More reliable
- ✅ Better tested
- ✅ More transparent (see all intermediate outputs)
- ✅ Applies proper SemCluster constraints
- ✅ Correct sparse pair computation

The fast path is experimental and may have issues with embedding format compatibility.
