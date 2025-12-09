#!/bin/bash
# Resume FULL dataset evaluation using resume_parquet_evaluation.py

echo "========================================================================"
echo "SEMCLUSTER FULL EVALUATION (RESUME MODE)"
echo "========================================================================"
echo "Dataset: Dataset/Overall - FULL_trimmed_year_1_corpus_with_gt.csv"
echo "Script: resume_parquet_evaluation.py"
echo "========================================================================"

# Ensure logs directory exists
mkdir -p logs

# Run evaluation
CUDA_VISIBLE_DEVICES=0 python3 resume_parquet_evaluation.py --dataset FULL 2>&1 | tee logs/evaluation_resume_full_$(date +%Y%m%d_%H%M%S).log
