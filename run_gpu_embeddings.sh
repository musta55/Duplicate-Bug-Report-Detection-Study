#!/bin/bash
# Activate conda environment
source /opt/anaconda/etc/profile.d/conda.sh
conda activate semcluster_gpu

# Set LD_LIBRARY_PATH to include conda lib and cudnn lib
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/python3.9/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH

# Run the script
python3 generate_embeddings.py
