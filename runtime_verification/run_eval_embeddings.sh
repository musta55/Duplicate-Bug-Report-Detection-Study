#!/bin/bash
SESSION_NAME="semcluster_eval_emb"

# Kill if exists
tmux kill-session -t $SESSION_NAME 2>/dev/null

tmux new-session -d -s $SESSION_NAME
tmux send-keys -t $SESSION_NAME "cd /home/mhasan02/SemCluster-v2-clean" C-m
tmux send-keys -t $SESSION_NAME "conda activate semcluster_gpu" C-m
tmux send-keys -t $SESSION_NAME "echo 'Starting evaluation from embeddings...'" C-m
tmux send-keys -t $SESSION_NAME "python3 -u run_evaluation_from_embeddings.py --dataset FULL 2>&1 | tee evaluation_from_embeddings.log" C-m
tmux send-keys -t $SESSION_NAME "echo 'Done.'" C-m

echo "Started tmux session: $SESSION_NAME"
echo "Attach with: tmux attach -t $SESSION_NAME"
