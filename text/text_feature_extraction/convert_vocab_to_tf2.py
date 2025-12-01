#!/usr/bin/env python
"""
Convert TensorFlow 1.15 VocabularyProcessor pickle to TF 2.x compatible JSON format.
Run this script in the tf115_gpu environment to extract the vocabulary mapping.
"""
import os
import sys
import json
import pickle

# Must run in tf115_gpu environment
import tensorflow as tf
from tensorflow.contrib import learn

def convert_vocab(vocab_path, output_json_path):
    """
    Load TF 1.15 vocab pickle and save as JSON
    
    Args:
        vocab_path: Path to vocab pickle file
        output_json_path: Path to save JSON vocab file
    """
    print(f"[Converter] Loading TF 1.15 vocabulary from: {vocab_path}")
    
    # Load using TF 1.15 contrib.learn
    vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
    
    # Extract vocabulary mapping
    vocab_dict = {}
    if hasattr(vocab_processor, 'vocabulary_'):
        # Get the actual word -> index mapping
        reverse_vocab = vocab_processor.vocabulary_._reverse_mapping
        
        # Create word -> index mapping
        for idx, word in enumerate(reverse_vocab):
            vocab_dict[word] = idx
    
    print(f"[Converter] Extracted {len(vocab_dict)} words")
    print(f"[Converter] Max document length: {vocab_processor.max_document_length}")
    
    # Create output structure
    vocab_data = {
        'max_document_length': vocab_processor.max_document_length,
        'vocabulary': vocab_dict,
        'vocab_size': len(vocab_dict)
    }
    
    # Save as JSON
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(vocab_data, f, ensure_ascii=False, indent=2)
    
    print(f"[Converter] Saved vocabulary to: {output_json_path}")
    print(f"[Converter] Sample words: {list(vocab_dict.keys())[:10]}")
    
    return vocab_data


if __name__ == '__main__':
    # Default paths
    script_dir = os.path.dirname(os.path.realpath(__file__))
    vocab_path = os.path.join(script_dir, 'runs', 'TextCNN_model', 'vocab')
    output_json = os.path.join(script_dir, 'runs', 'TextCNN_model', 'vocab.json')
    
    if len(sys.argv) > 1:
        vocab_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_json = sys.argv[2]
    
    print("=" * 70)
    print("TensorFlow 1.15 Vocabulary Converter")
    print("=" * 70)
    
    vocab_data = convert_vocab(vocab_path, output_json)
    
    print("\n" + "=" * 70)
    print(f"✓ Conversion complete!")
    print(f"  Vocabulary size: {vocab_data['vocab_size']}")
    print(f"  Max length: {vocab_data['max_document_length']}")
    print(f"  Output: {output_json}")
    print("=" * 70)
