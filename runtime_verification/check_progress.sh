#!/bin/bash
# Quick progress check for both evaluations

echo "========================================================================"
echo "SEMCLUSTER EVALUATION PROGRESS CHECK"
echo "========================================================================"
echo "Time: $(date)"
echo ""

echo "--- TMUX SESSIONS ---"
tmux list-sessions 2>/dev/null || echo "No tmux sessions running"
echo ""

echo "--- OUTPUT FILE SIZES ---"
ls -lh output/semcluster_similarity_matrix_*.csv 2>/dev/null || echo "No output files yet"
echo ""

echo "--- ROW COUNTS ---"
if [ -f output/semcluster_similarity_matrix_FILTERED.csv ]; then
    FILTERED_ROWS=$(wc -l < output/semcluster_similarity_matrix_FILTERED.csv)
    echo "FILTERED: $FILTERED_ROWS rows"
else
    echo "FILTERED: No file yet"
fi

if [ -f output/semcluster_similarity_matrix_FULL.csv ]; then
    FULL_ROWS=$(wc -l < output/semcluster_similarity_matrix_FULL.csv)
    echo "FULL: $FULL_ROWS rows"
else
    echo "FULL: No file yet"
fi
echo ""

echo "--- RECENT FILTERED OUTPUT (last 15 lines) ---"
tmux capture-pane -t semcluster_filtered -p 2>/dev/null | tail -15 || echo "Session not found"
echo ""

echo "--- RECENT FULL OUTPUT (last 15 lines) ---"
tmux capture-pane -t semcluster_full -p 2>/dev/null | tail -15 || echo "Session not found"
echo ""

echo "========================================================================"
echo "To attach to sessions:"
echo "  tmux attach -t semcluster_filtered"
echo "  tmux attach -t semcluster_full"
echo ""
echo "To detach: Press Ctrl+B, then D"
echo "========================================================================"
