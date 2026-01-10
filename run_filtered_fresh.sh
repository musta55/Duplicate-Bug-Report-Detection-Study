#!/bin/bash
# Run FILTERED dataset evaluation from scratch (no pickle files)
# This script will be executed in tmux

set -e  # Exit on error

echo "=========================================="
echo "FILTERED Dataset Evaluation - Fresh Run"
echo "=========================================="
echo "Started at: $(date)"
echo ""

# Use semcluster_gpu Python and add tesseract to PATH
export PATH="/home/mhasan02/.conda/envs/tesseract_env/bin:$PATH"
PYTHON_BIN="/home/mhasan02/.conda/envs/semcluster_gpu/bin/python"

# Go to project directory
cd /home/mhasan02/SemCluster-v2-clean

# Remove any existing pickle files to force fresh generation
echo "[1/5] Removing old pickle files..."
rm -f embeddings/*_filtered.pkl
rm -f embeddings/text_embeddings.pkl embeddings/structure_embeddings.pkl embeddings/content_embeddings.pkl
echo "  ✓ Old pickle files removed"
echo ""

# Clean output directories
echo "[2/5] Cleaning old output directories..."
rm -rf file/pic_file_parquet_filtered
rm -rf file/xml_file_parquet_filtered
rm -f file/label_file_parquet/evaluation_filtered.csv
echo "  ✓ Old outputs cleaned"
echo ""

# Clean previous results
echo "[3/5] Backing up previous results..."
if [ -f output/semcluster_similarity_matrix_FILTERED.csv ]; then
    mv output/semcluster_similarity_matrix_FILTERED.csv output/semcluster_similarity_matrix_FILTERED_backup_$(date +%Y%m%d_%H%M%S).csv
    echo "  ✓ Previous similarity matrix backed up"
fi
if [ -f output/Projectwise_Retrieval_Filtered.csv ]; then
    mv output/Projectwise_Retrieval_Filtered.csv output/Projectwise_Retrieval_Filtered_backup_$(date +%Y%m%d_%H%M%S).csv
    echo "  ✓ Previous projectwise metrics backed up"
fi
echo ""

# Run the evaluation from scratch
echo "[4/5] Running evaluation from scratch..."
echo "  This will take 15-30 minutes..."
echo "  Processing all queries with full feature extraction"
echo ""
$PYTHON_BIN run_parquet_evaluation.py --dataset FILTERED 2>&1 | tee filtered_evaluation_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "[5/5] Evaluation complete!"
echo "Finished at: $(date)"
echo ""
echo "Output files:"
echo "  - output/semcluster_similarity_matrix_FILTERED.csv"
echo "  - output/Projectwise_Retrieval_Filtered.csv"
echo ""
echo "Log saved to: filtered_evaluation_*.log"
echo "=========================================="
