#!/usr/bin/env python
"""
TextCNN inference using TensorFlow 2.x with JSON vocabulary.
This replaces the TF 1.15 subprocess approach.
"""
import os
import re
import json
import numpy as np
import tensorflow as tf
import jieba.posseg

from text.text_feature_extraction.word_segment import word_segment2token


class VocabularyProcessor:
    """TF 2.x compatible vocabulary processor using JSON vocab"""
    
    def __init__(self, max_document_length, vocabulary):
        self.max_document_length = max_document_length
        self.vocabulary_ = vocabulary
        self._word_to_id = {word: idx for word, idx in vocabulary.items()}
        
    def transform(self, raw_documents):
        """Transform documents to word id sequences"""
        for doc in raw_documents:
            word_ids = []
            words = doc.split()
            for word in words[:self.max_document_length]:
                # Use vocabulary mapping, default to 0 (UNK) for unknown words
                word_ids.append(self._word_to_id.get(word, 0))
            # Pad to max_document_length
            while len(word_ids) < self.max_document_length:
                word_ids.append(0)
            yield np.array(word_ids[:self.max_document_length], dtype=np.int32)
    
    @classmethod
    def restore(cls, vocab_json_path):
        """Restore vocabulary from JSON file"""
        print(f"[VocabProcessor] Loading vocabulary from {vocab_json_path}")
        with open(vocab_json_path, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        max_len = vocab_data['max_document_length']
        vocabulary = vocab_data['vocabulary']
        
        print(f"[VocabProcessor] Loaded {len(vocabulary)} words, max_len={max_len}")
        return cls(max_document_length=max_len, vocabulary=vocabulary)


def text_feature_extraction(samples):
    """
    Extract text features using TextCNN model with TF 2.x.
    GPU-accelerated inference.
    """
    x_raw_list = []
    sentences_list = []
    
    total_samples = len(samples)
    print(f"[TextCNN] Processing {total_samples} samples...")
    
    for i, sample in enumerate(samples):
        if i % 1000 == 0:
            print(f"[TextCNN] Tokenizing sample {i}/{total_samples}...")
            
        # Coerce non-string samples (NaN, None) to empty string to avoid TypeError in re.split
        if sample is None:
            sample = ''
        try:
            # If it's a float NaN, convert to empty string
            if isinstance(sample, float) and np.isnan(sample):
                sample = ''
        except Exception:
            pass
        sample = str(sample)
        sentences = re.split(' |。', sample)
        sentences = [item for item in filter(lambda x: x != '', sentences)]
        sentences_list.append(sentences)
        x_raw = []
        for sentence in sentences:
            tmp = word_segment2token(sentence)
            x_raw.append(tmp.strip())
        x_raw_list.append(x_raw)

    res = []
    curpath = os.path.dirname(os.path.realpath(__file__))
    
    # Configuration
    checkpoint_dir = os.path.join(curpath, 'runs', 'TextCNN_model', 'checkpoints')
    vocab_json_path = os.path.join(checkpoint_dir, "..", "vocab.json")
    
    # Check if JSON vocab exists, fallback to TF 1.15 subprocess if not
    if not os.path.exists(vocab_json_path):
        print(f"[TextCNN] WARNING: {vocab_json_path} not found!")
        print(f"[TextCNN] Falling back to TF 1.15 subprocess...")
        return text_feature_extraction_tf115(samples)
    
    # Load vocabulary from JSON
    vocab_processor = VocabularyProcessor.restore(vocab_json_path)
    
    # Transform text to word IDs
    x_test_list = []
    for x_raw in x_raw_list:
        x_test = np.array(list(vocab_processor.transform(x_raw)))
        x_test_list.append(x_test)

    print(f"\n[TextCNN] Loading model checkpoint from {checkpoint_dir}")
    print(f"[TextCNN] Running inference on GPU with TensorFlow {tf.__version__}")
    
    # Load checkpoint and run inference with TF 2.x
    checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
    print(f"[TextCNN] Checkpoint: {checkpoint_file}")

    # TF 2.x: Use tf.compat.v1 for checkpoint loading
    tf.compat.v1.disable_eager_execution()
    
    graph = tf.Graph()
    with graph.as_default():
        # GPU configuration with memory growth
        session_conf = tf.compat.v1.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False,
            gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
        sess = tf.compat.v1.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.compat.v1.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)
            
            print("[TextCNN] ✓ Model restored successfully, running on GPU")

            # Get the placeholders from the graph by name
            input_x = graph.get_operation_by_name("input_x").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

            # Tensors we want to evaluate
            predictions = graph.get_operation_by_name("output/predictions").outputs[0]

            for j in range(len(x_test_list)):
                # Run prediction in batches to avoid OOM
                x_test = x_test_list[j]
                num_sentences = len(x_test)
                batch_size = 128  # Conservative batch size for stability
                all_predictions = []
                
                try:
                    for k in range(0, num_sentences, batch_size):
                        batch_x = x_test[k:k+batch_size]
                        batch_predictions = sess.run(
                            predictions, 
                            {input_x: batch_x, dropout_keep_prob: 1.0}
                        )
                        all_predictions.extend(batch_predictions)
                except Exception as e:
                    print(f"[TextCNN] ERROR processing report {j}: {e}")
                    # Fallback: treat all as problems (0.0) or skip
                    print(f"[TextCNN] Fallback: treating remaining sentences as problems")
                    remaining = num_sentences - len(all_predictions)
                    all_predictions.extend([0.0] * remaining)

                problems = []
                procedures = []
                for i in range(len(all_predictions)):
                    short_sentences = sentences_list[j]
                    if all_predictions[i] == 0.0:
                        # label = 0 represent bug descriptions
                        problems.append(short_sentences[i])
                    else:
                        # label = 1 represent reproduction steps
                        procedures.append(short_sentences[i])

                # lexical analysis to extract problem widget
                problem_widget = ''
                if len(procedures) >= 1:
                    last_procedure = procedures[-1]
                    last_procedure_seged = jieba.posseg.cut(last_procedure.strip())
                    first_v = False
                    for x in last_procedure_seged:
                        if first_v:
                            if x.flag != 'x' and x.flag != 'm':
                                problem_widget += x.word
                        else:
                            if x.flag == 'v':
                                first_v = True
                
                dict_res = {
                    'procedures_list': procedures, 
                    'problem_widget': problem_widget, 
                    'problems_list': problems
                }
                res.append(dict_res)
    
    print(f"[TextCNN] ✓ Processed {len(res)} samples on GPU")
    return res


def text_feature_extraction_tf115(samples):
    """
    Fallback: Extract text features using TF 1.15 subprocess.
    Used only if vocab.json doesn't exist.
    """
    import subprocess
    import sys
    
    curpath = os.path.dirname(os.path.realpath(__file__))
    checkpoint_dir = os.path.join(curpath, 'runs', 'TextCNN_model', 'checkpoints')
    vocab_path = os.path.join(checkpoint_dir, "..", "vocab")
    
    tf115_script = os.path.join(curpath, 'run_textcnn_tf115.py')
    tf115_python = os.path.expanduser("~/.conda/envs/tf115_gpu/bin/python")
    
    samples_json = json.dumps(samples, ensure_ascii=False)
    
    print(f"\n[TextCNN] Calling TF 1.15 subprocess (CPU fallback)...")
    
    cmd = [tf115_python, tf115_script, samples_json, checkpoint_dir, vocab_path]
    
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600, check=True
        )
        
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        
        res = json.loads(result.stdout)
        print(f"[TextCNN] ✓ Processed {len(res)} samples via TF 1.15 (CPU)")
        return res
        
    except subprocess.CalledProcessError as e:
        print(f"[TextCNN] ERROR: TF 1.15 subprocess failed with code {e.returncode}")
        print(f"[TextCNN] STDERR: {e.stderr}")
        raise RuntimeError(f"TextCNN inference failed: {e.stderr}")
    except subprocess.TimeoutExpired:
        raise RuntimeError("TextCNN inference timed out after 10 minutes")
    except json.JSONDecodeError as e:
        print(f"[TextCNN] ERROR: Failed to parse JSON result")
        raise RuntimeError(f"TextCNN returned invalid JSON: {e}")
