#!/bin/bash
#
# SemCluster End-to-End Test Pipeline
#
# This script runs the complete testing pipeline:
# 1. Generate embeddings and similarity matrices (test_scenarios.py)
# 2. Calculate component scores and metrics (post_process_results.py)
#
# Outputs: test_output/
#   - semcluster_similarity_matrix_FILTERED.csv (60 pairs, 3 projects)
#   - semcluster_similarity_matrix_FULL.csv (375 pairs, 5 projects)
#   - Projectwise_Retrieval_FILTERED.csv (metrics for 3 projects)
#   - Projectwise_Retrieval_FULL.csv (metrics for 5 projects)
#   - embeddings/*.pkl (BB, RS, SF, CF features)
#
# Usage:
#   ./test_pipeline_quick.sh
#   # Or with cleanup:
#   rm -rf test_output/ && ./test_pipeline_quick.sh
#

echo "=========================================="
echo "SemCluster Pipeline Quick Test"
echo "=========================================="
echo ""
echo "Testing:"
echo "  1. FILTERED dataset (100% text + 100% images)"
echo "  2. FULL dataset simulation (100% text + 10-12% images)"
echo "  3. Pickle file creation and loading"
echo "  4. Similarity matrix computation with adaptive fusion"
echo ""

cd "$(dirname "$0")/.."
python testing/test_scenarios.py

echo ""
echo "Running post-processing to add component scores and metrics..."
python testing/post_process_results.py

echo ""
echo "Done!"
