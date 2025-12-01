# SemCluster Evaluation Status

## Current Run Information

**Start Time:** Saturday, November 29, 2025 at 11:38 PM EST

### FILTERED Dataset Evaluation
- **Tmux Session:** `semcluster_filtered`
- **Process ID:** 3844203
- **CPU Usage:** ~109% (actively processing)
- **Memory:** 10.3 GB
- **Expected Duration:** 15-30 minutes (with caching)
- **Output File:** `semcluster_similarity_matrix_FILTERED.csv`
- **Log File:** `logs/evaluation_filtered_20251129_233850.log`

### FULL Dataset Evaluation
- **Tmux Session:** `semcluster_full`
- **Process ID:** 3844579
- **CPU Usage:** ~70% (actively processing)
- **Memory:** 10.4 GB
- **Expected Duration:** 2-3 days (48-72 hours with caching)
- **Output File:** `semcluster_similarity_matrix_FULL.csv`
- **Log File:** `logs/evaluation_full_20251129_233856.log`

## Monitoring Commands

### Check Progress
```bash
# Quick status check
./check_progress.sh

# Watch real-time updates every 30 seconds
watch -n 30 ./check_progress.sh
```

### Attach to Sessions
```bash
# FILTERED evaluation
tmux attach -t semcluster_filtered

# FULL evaluation
tmux attach -t semcluster_full

# Detach from session: Press Ctrl+B, then D
```

### Check Process Status
```bash
# See running processes
ps aux | grep run_parquet_evaluation.py | grep -v grep

# Check output file sizes
ls -lh semcluster_similarity_matrix_*.csv

# Count rows in output
wc -l semcluster_similarity_matrix_*.csv
```

### View Logs
```bash
# FILTERED log (live)
tail -f logs/evaluation_filtered_20251129_233850.log

# FULL log (live)
tail -f logs/evaluation_full_20251129_233856.log
```

## Expected Results

### FILTERED Dataset (92 queries, 100% with images)
- **Total rows expected:** ~8,372 (92 queries × ~91 corpus items)
- **Current incomplete file:** 198 rows (only 1 query)
- **Metrics (from README):**
  - MRR: 0.1725
  - HITS@1: 9.8%
  - HITS@10: 29.3%

### FULL Dataset (1,034 queries, 10% with images)
- **Total rows expected:** ~528,897 (already achieved in previous run)
- **Current file:** 528,897 rows ✓
- **Metrics (from README):**
  - MRR: 0.0918
  - HITS@1: 7.0%
  - HITS@10: 10.7%

## Must-Link / Cannot-Link Analysis

**After evaluations complete**, you can analyze thresholds with:

```python
import pandas as pd
import numpy as np

# Load results
filtered_df = pd.read_csv('semcluster_similarity_matrix_FILTERED.csv')

# Analyze ground truth score distribution
gt_scores = filtered_df[filtered_df['c_is_gt'] == 1]['score']
non_gt_scores = filtered_df[filtered_df['c_is_gt'] == 0]['score']

# Propose thresholds
must_link_threshold = gt_scores.quantile(0.25)  # 25th percentile of GT scores
cannot_link_threshold = non_gt_scores.quantile(0.95)  # 95th percentile of non-GT

print(f"Proposed Must-Link threshold: > {must_link_threshold:.4f}")
print(f"Proposed Cannot-Link threshold: < {cannot_link_threshold:.4f}")
```

## Troubleshooting

### If Evaluation Stops or Crashes
```bash
# Check if process is still running
ps aux | grep run_parquet_evaluation.py

# View recent errors in log
tail -100 logs/evaluation_filtered_20251129_233850.log

# Restart if needed
tmux kill-session -t semcluster_filtered
bash run_filtered_eval.sh
```

### Clean Restart (if needed)
```bash
# Kill both sessions
tmux kill-session -t semcluster_filtered
tmux kill-session -t semcluster_full

# Remove incomplete output (optional)
rm semcluster_similarity_matrix_FILTERED.csv
rm semcluster_similarity_matrix_FULL.csv

# Restart
bash run_filtered_eval.sh
bash run_full_eval.sh
```

## Caching Notes

The evaluation script includes caching for:
- **VGG16 features** (image content)
- **Structure features** (XML layouts)
- **Text embeddings** (Word2Vec + TextCNN)

Cached features are stored in the parquet file, significantly reducing runtime for subsequent evaluations.

## Next Steps

1. **Monitor progress** using `./check_progress.sh`
2. **Wait for FILTERED** to complete (~15-30 minutes)
3. **Wait for FULL** to complete (~2-3 days)
4. **Verify output files** have complete data
5. **Run metrics analysis** to calculate MRR, HITS@k
6. **Determine must-link/cannot-link thresholds** based on score distributions

---

**Last Updated:** November 29, 2025 at 11:42 PM EST
