#!/bin/bash
# Monitor FULL evaluation progress

echo "=== FULL Evaluation Monitor ==="
echo ""

# Check if process is running
PID=$(ps aux | grep "[p]ython run_parquet_evaluation.py --dataset FULL" | awk '{print $2}' | head -1)

if [ -z "$PID" ]; then
    echo "❌ No FULL evaluation process running"
    echo ""
    echo "Checking if output file was generated..."
    if [ -f "output/semcluster_similarity_matrix_FULL.csv" ]; then
        echo "✅ Output file exists:"
        ls -lh output/semcluster_similarity_matrix_FULL.csv
        echo ""
        echo "Unique queries in output:"
        python3 -c "import pandas as pd; df = pd.read_csv('output/semcluster_similarity_matrix_FULL.csv'); print(f'  {df[\"query\"].nunique():,} queries')"
    fi
    exit 0
fi

echo "✅ Process running (PID: $PID)"
echo ""

# Process info
ps -p $PID -o pid,ppid,%cpu,%mem,etime,cmd | tail -1
echo ""

# GPU memory
echo "GPU Memory:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader | head -4
echo ""

# Check image extraction progress
if [ -d "file/pic_file_parquet_full" ]; then
    IMG_COUNT=$(ls file/pic_file_parquet_full/*.png 2>/dev/null | wc -l)
    echo "Images extracted: $IMG_COUNT / 4021"
fi

# Check for checkpoint files
if [ -f "content_features_checkpoint.pkl" ]; then
    echo "Content feature checkpoint found (processing in progress)"
fi

echo ""
echo "Run this script again to check progress"
echo "Or use: watch -n 30 ./monitor_eval.sh"
