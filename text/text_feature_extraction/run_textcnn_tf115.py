#!/usr/bin/env python
"""
Standalone TextCNN inference script using TensorFlow 1.15.
This script is called by text_feature_extraction.py via subprocess.
It loads the vocab using TF 1.15's contrib.learn and runs inference.
"""
import os
import sys
import json
import pickle
import numpy as np

# Must run in tf115_gpu environment
import tensorflow as tf
from tensorflow.contrib import learn

def text_feature_extraction_tf115(samples_json, checkpoint_dir, vocab_path):
    """
    Extract text features using TextCNN with TF 1.15
    
    Args:
        samples_json: JSON string containing list of samples
        checkpoint_dir: Path to checkpoint directory
        vocab_path: Path to vocab file
        
    Returns:
        JSON string containing results
    """
    samples = json.loads(samples_json)
    
    # Preprocess samples
    x_raw_list = []
    sentences_list = []
    for sample in samples:
        import re
        sentences = re.split(' |。', sample)
        sentences = [item for item in filter(lambda x: x != '', sentences)]
        sentences_list.append(sentences)
        x_raw = []
        for sentence in sentences:
            # Simple tokenization - word_segment2token not available here
            from jieba import posseg
            tmp = ' '.join([word for word, flag in posseg.cut(sentence)])
            x_raw.append(tmp.strip())
        x_raw_list.append(x_raw)

    # Load vocabulary using TF 1.15 native VocabularyProcessor
    print(f"[TF1.15] Loading vocabulary from {vocab_path}", file=sys.stderr)
    vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
    print(f"[TF1.15] Vocabulary loaded: {len(vocab_processor.vocabulary_)} words, "
          f"max_len={vocab_processor.max_document_length}", file=sys.stderr)
    
    # Transform text to word IDs
    x_test_list = []
    for x_raw in x_raw_list:
        x_test = np.array(list(vocab_processor.transform(x_raw)))
        x_test_list.append(x_test)

    print(f"[TF1.15] Loading model checkpoint from {checkpoint_dir}", file=sys.stderr)
    
    # Load checkpoint and run inference
    checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
    print(f"[TF1.15] Checkpoint: {checkpoint_file}", file=sys.stderr)

    graph = tf.Graph()
    with graph.as_default():
        # CPU-only session (no GPU libraries for TF 1.15 with CUDA 12.5)
        session_conf = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False,
            device_count={'GPU': 0})  # Force CPU
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # Load meta graph
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)
            
            # Get tensors
            input_x = graph.get_operation_by_name("input_x").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
            predictions = graph.get_operation_by_name("output/predictions").outputs[0]
            
            res = []
            
            for i, x_test in enumerate(x_test_list):
                print(f"[TF1.15] Processing sample {i+1}/{len(x_test_list)}", file=sys.stderr)
                
                # Run prediction
                all_predictions = sess.run(predictions, {input_x: x_test, dropout_keep_prob: 1.0})
                
                # Extract procedures and problems
                procedures = []
                problem_widget = -1
                problems = []
                
                for j, pred in enumerate(all_predictions):
                    if pred == 1:  # procedures
                        procedures.append(sentences_list[i][j])
                    else:  # problem
                        if problem_widget == -1:
                            problem_widget = j
                        problems.append(sentences_list[i][j])
                
                dict_res = {
                    'procedures_list': procedures,
                    'problem_widget': problem_widget,
                    'problems_list': problems
                }
                res.append(dict_res)
    
    print(f"[TF1.15] Inference complete", file=sys.stderr)
    return json.dumps(res, ensure_ascii=False)


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: run_textcnn_tf115.py <samples_json> <checkpoint_dir> <vocab_path>", 
              file=sys.stderr)
        sys.exit(1)
    
    samples_json = sys.argv[1]
    checkpoint_dir = sys.argv[2]
    vocab_path = sys.argv[3]
    
    result = text_feature_extraction_tf115(samples_json, checkpoint_dir, vocab_path)
    print(result)  # stdout only contains JSON result
