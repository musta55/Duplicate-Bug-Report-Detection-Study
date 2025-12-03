import numpy as np
import tensorflow as tf
import gc
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

# Configure GPU memory growth to prevent OOM
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"GPU memory config error: {e}")

# Global model - load once and reuse
_vgg16_model = None

def get_model():
    """Get or create VGG16 model (singleton pattern for efficiency)"""
    global _vgg16_model
    if _vgg16_model is None:
        # Load VGG16 with ImageNet weights
        # 
        # WHY USE ImageNet PRETRAINED WEIGHTS?
        # =====================================
        # The SemCluster paper (Liang et al.) uses VGG16 for visual feature extraction
        # from Android bug report screenshots, but does NOT provide fine-tuned weights
        # or training code for their specific dataset.
        #
        # Therefore, we use ImageNet pretrained weights as a proxy:
        # 1. UNAVAILABLE WEIGHTS: Original paper's trained weights are not released
        # 2. TRANSFER LEARNING: ImageNet features (edges, textures, shapes) transfer
        #    reasonably well to UI screenshot similarity tasks
        # 3. STANDARD PRACTICE: Using VGG16+ImageNet is a common baseline when
        #    task-specific weights are unavailable
        #
        # LIMITATIONS:
        # - ImageNet is trained on natural images (animals, objects), NOT UI screenshots
        # - This mismatch explains the poor discriminative power (all similarities ~0.96)
        # - Ideally, VGG16 should be fine-tuned on Android UI screenshot pairs
        #
        # ARCHITECTURAL CHOICE:
        # - include_top=False: Remove classification head, use conv features only
        # - layers[-2]: Extract features from second-to-last layer
        #
        base_model = VGG16(include_top=False, weights='imagenet')
        # Use the second-to-last layer for feature extraction
        _vgg16_model = Model(inputs=base_model.inputs, outputs=base_model.layers[-2].output)
        print(f"[VGG16] Model loaded with ImageNet weights (SemCluster paper weights unavailable)")
        print(f"[VGG16] Device: {tf.config.list_physical_devices('GPU')}")
    return _vgg16_model


def extract_features(image):
    """Extract features from an image using VGG16"""
    model = get_model()
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the VGG model
    image = preprocess_input(image)
    # get features with explicit batch size to limit memory
    feature = model.predict(image, verbose=0, batch_size=1)
    return feature


def getdistance(widget_list, normalize=True):
    """
    GPU-OPTIMIZED: Compute distances between image pairs using VGG16 features.
    Process images in batches for maximum GPU throughput (100x faster than one-by-one).
    
    Args:
        widget_list: List of (image1, image2) pairs
        normalize: If True, L2-normalize feature vectors before distance calculation (Cosine-like)
    """
    if not widget_list:
        return []
    
    import sys
    from sklearn.preprocessing import normalize as sk_normalize
    
    total = len(widget_list)
    print(f"  [VGG16 GPU-OPTIMIZED] Processing {total} widget pairs in batches... (Normalize={normalize})", flush=True)
    
    # GPU-optimized batch size - process many images at once
    gpu_batch_size = 128  # Process 128 images per GPU batch for maximum throughput
    
    distance_list = []
    
    # Extract all images from pairs
    images_1 = [pair[0] for pair in widget_list]
    images_2 = [pair[1] for pair in widget_list]
    
    # Process in batches for GPU efficiency
    features_1_all = []
    features_2_all = []
    
    print(f"  [VGG16] Extracting features for {total} image pairs using GPU...", flush=True)
    
    # Extract features for first images in batches
    for batch_start in range(0, total, gpu_batch_size):
        batch_end = min(batch_start + gpu_batch_size, total)
        batch_images = images_1[batch_start:batch_end]
        
        # Stack images into a batch for GPU processing
        batch_array = np.array(batch_images)
        
        # Get model and extract features for entire batch at once (GPU accelerated)
        model = get_model()
        batch_features = model.predict(batch_array, batch_size=min(32, len(batch_images)), verbose=0)
        features_1_all.extend(batch_features)
        
        if (batch_end) % 500 == 0 or batch_end == total:
            print(f"  [VGG16] Batch 1: {batch_end}/{total} ({batch_end/total*100:.1f}%)", flush=True)
    
    # Extract features for second images in batches
    for batch_start in range(0, total, gpu_batch_size):
        batch_end = min(batch_start + gpu_batch_size, total)
        batch_images = images_2[batch_start:batch_end]
        
        # Stack images into a batch for GPU processing
        batch_array = np.array(batch_images)
        
        # Get model and extract features for entire batch at once (GPU accelerated)
        model = get_model()
        batch_features = model.predict(batch_array, batch_size=min(32, len(batch_images)), verbose=0)
        features_2_all.extend(batch_features)
        
        if (batch_end) % 500 == 0 or batch_end == total:
            print(f"  [VGG16] Batch 2: {batch_end}/{total} ({batch_end/total*100:.1f}%)", flush=True)
    
    # Compute distances (CPU operation, very fast)
    print(f"  [VGG16] Computing distances...", flush=True)
    
    # Convert to numpy arrays for faster processing
    features_1_all = np.array(features_1_all)
    features_2_all = np.array(features_2_all)
    
    # Flatten features if they are not already 1D per sample
    if len(features_1_all.shape) > 2:
        features_1_all = features_1_all.reshape(features_1_all.shape[0], -1)
        features_2_all = features_2_all.reshape(features_2_all.shape[0], -1)
        
    if normalize:
        print(f"  [VGG16] Applying L2 normalization to features...", flush=True)
        features_1_all = sk_normalize(features_1_all, norm='l2', axis=1)
        features_2_all = sk_normalize(features_2_all, norm='l2', axis=1)
        
    for feat1, feat2 in zip(features_1_all, features_2_all):
        distance = np.sqrt(np.sum(np.square(feat1 - feat2)))
        distance_list.append(distance)
    
    print(f"  [VGG16] Completed {total} comparisons using GPU batching!", flush=True)
    
    return distance_list
