
import time
import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

from text.text_feature_extraction.word_segment import word_segment2token

def test_segmentation():
    print("Testing word_segment2token performance...")
    
    sample_text = "This is a test sentence to check if jieba is working correctly and efficiently."
    
    start_time = time.time()
    # First call (should initialize jieba and load stopwords)
    res = word_segment2token(sample_text)
    first_call_time = time.time() - start_time
    print(f"First call time: {first_call_time:.4f} seconds")
    print(f"Result: {res}")
    
    # Second call (should be fast)
    start_time = time.time()
    res = word_segment2token(sample_text)
    second_call_time = time.time() - start_time
    print(f"Second call time: {second_call_time:.4f} seconds")
    
    # Loop
    start_time = time.time()
    for i in range(1000):
        word_segment2token(sample_text)
    loop_time = time.time() - start_time
    print(f"1000 calls time: {loop_time:.4f} seconds")
    print(f"Average per call: {loop_time/1000:.6f} seconds")

if __name__ == "__main__":
    test_segmentation()
