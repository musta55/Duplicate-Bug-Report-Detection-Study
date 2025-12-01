import os
import re
import numpy as np
import jieba.posseg
import json
import subprocess
import sys

from text.text_feature_extraction.word_segment import word_segment2token


def text_feature_extraction(samples):
    """
    Extract text features using TextCNN model.
    Calls TF 1.15 subprocess to load vocab and run inference.
    """
    curpath = os.path.dirname(os.path.realpath(__file__))
    checkpoint_dir = os.path.join(curpath, 'runs', 'TextCNN_model', 'checkpoints')
    vocab_path = os.path.join(checkpoint_dir, "..", "vocab")
    
    # Call TF 1.15 subprocess
    tf115_script = os.path.join(curpath, 'run_textcnn_tf115.py')
    tf115_python = os.path.expanduser("~/.conda/envs/tf115_gpu/bin/python")
    
    # Prepare samples as JSON
    samples_json = json.dumps(samples, ensure_ascii=False)
    
    print(f"\n[TextCNN] Calling TF 1.15 subprocess for vocab loading...")
    print(f"[TextCNN] Script: {tf115_script}")
    print(f"[TextCNN] Python: {tf115_python}")
    
    # Run subprocess with TF 1.15 Python directly
    cmd = [
        tf115_python,
        tf115_script,
        samples_json,
        checkpoint_dir,
        vocab_path
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
            check=True
        )
        
        # Print stderr (progress messages)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        
        # Parse stdout (JSON result)
        res = json.loads(result.stdout)
        print(f"[TextCNN] Successfully processed {len(res)} samples via TF 1.15")
        return res
        
    except subprocess.CalledProcessError as e:
        print(f"[TextCNN] ERROR: TF 1.15 subprocess failed with code {e.returncode}")
        print(f"[TextCNN] STDOUT: {e.stdout}")
        print(f"[TextCNN] STDERR: {e.stderr}")
        raise RuntimeError(f"TextCNN inference failed: {e.stderr}")
    except subprocess.TimeoutExpired:
        raise RuntimeError("TextCNN inference timed out after 10 minutes")
    except json.JSONDecodeError as e:
        print(f"[TextCNN] ERROR: Failed to parse JSON result")
        print(f"[TextCNN] STDOUT: {result.stdout}")
        raise RuntimeError(f"TextCNN returned invalid JSON: {e}")
