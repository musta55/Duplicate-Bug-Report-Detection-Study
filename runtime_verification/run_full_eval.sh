#!/bin/bash
# Run FULL dataset evaluation in tmux
# This will process ALL queries with ALL bug reports

SESSION_NAME="semcluster_full"

# Check if session already exists
tmux has-session -t $SESSION_NAME 2>/dev/null

if [ $? == 0 ]; then
    echo "❌ Tmux session '$SESSION_NAME' already exists!"
    echo "   Attach with: tmux attach -t $SESSION_NAME"
    echo "   Or kill with: tmux kill-session -t $SESSION_NAME"
    exit 1
fi

echo "========================================================================"
echo "SEMCLUSTER FULL EVALUATION (TMUX)"
echo "========================================================================"
echo "Session name: $SESSION_NAME"
echo "Dataset: Dataset/Overall - FULL_trimmed_year_1_corpus_with_gt.csv"
echo "Queries: ALL (~2323 queries, including text-only)"
echo "Expected time: 2-3 days (48-72 hours)"
echo "========================================================================"
echo ""

# Create new tmux session
tmux new-session -d -s $SESSION_NAME

# Send commands to the session
tmux send-keys -t $SESSION_NAME "cd /home/mhasan02/SemCluster-v2-clean" C-m
tmux send-keys -t $SESSION_NAME "conda activate semcluster_gpu" C-m
tmux send-keys -t $SESSION_NAME "echo '========================================'" C-m
tmux send-keys -t $SESSION_NAME "echo 'FULL EVALUATION STARTED'" C-m
tmux send-keys -t $SESSION_NAME "echo 'Start time: \$(date)'" C-m
tmux send-keys -t $SESSION_NAME "echo '========================================'" C-m
tmux send-keys -t $SESSION_NAME "" C-m

# Run evaluation with FULL dataset flag and single GPU
tmux send-keys -t $SESSION_NAME "CUDA_VISIBLE_DEVICES=0 python3 run_parquet_evaluation.py --dataset FULL 2>&1 | tee logs/evaluation_full_\$(date +%Y%m%d_%H%M%S).log" C-m

# Add completion message
tmux send-keys -t $SESSION_NAME "echo ''" C-m
tmux send-keys -t $SESSION_NAME "echo '========================================'" C-m
tmux send-keys -t $SESSION_NAME "echo 'FULL EVALUATION COMPLETE'" C-m
tmux send-keys -t $SESSION_NAME "echo 'End time: \$(date)'" C-m
tmux send-keys -t $SESSION_NAME "echo '========================================'" C-m
tmux send-keys -t $SESSION_NAME "echo 'Output: output/semcluster_similarity_matrix_FULL.csv'" C-m
tmux send-keys -t $SESSION_NAME "echo ''" C-m
tmux send-keys -t $SESSION_NAME "echo 'Session will remain open. To close: tmux kill-session -t $SESSION_NAME'" C-m

echo "✓ Tmux session created and evaluation started!"
echo ""
echo "To monitor progress:"
echo "  tmux attach -t $SESSION_NAME"
echo ""
echo "To detach from session (while inside):"
echo "  Press: Ctrl+B, then D"
echo ""
echo "To check if still running:"
echo "  tmux list-sessions"
echo ""
echo "To kill session:"
echo "  tmux kill-session -t $SESSION_NAME"
echo ""
