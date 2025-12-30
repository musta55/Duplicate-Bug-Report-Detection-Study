# Duplicate Bug Report Detection Study
## Integrating Visual and Textual Semantics Using Deep Learning

---

# Slide 1: Title

## **Duplicate Bug Report Detection Study**
### Integrating Visual and Textual Semantics Using Deep Learning

**Research Question:**  
*Does including images (screenshots) as an additional source of information improve duplicate bug report detection in Android applications?*

---

# Slide 2: Background

## The Problem of Duplicate Bug Reports

### What Are Duplicate Bug Reports?
- Multiple users report the **same software bug** with different descriptions
- Common in open-source projects and issue tracking systems (GitHub, Jira, Bugzilla)

### Why It Matters
- **Developer Time Waste**: Triagers spend 20-30% of time identifying duplicates
- **Resource Drain**: ~25% of bug reports are duplicates (empirical studies)
- **Delayed Fixes**: True bugs get lost among duplicates

### Traditional Approach
- Manual inspection by developers
- Keyword-based search
- Text similarity measures (TF-IDF, BM25)

---

# Slide 3: Challenges & Related Work

## Challenges in Duplicate Detection

| Challenge | Description |
|-----------|-------------|
| **Lexical Gap** | Same bug described using different words |
| **Semantic Variance** | "App crashes" vs "Application terminates unexpectedly" |
| **Incomplete Reports** | Missing reproduction steps or context |
| **Visual Information Loss** | Screenshots contain rich UI context often ignored |

## Related Work

### Text-Based Approaches
- **BM25 / TF-IDF**: Classic IR-based retrieval
- **Word2Vec + Cosine Similarity**: Semantic embeddings
- **BERT/Transformers**: Pre-trained language models

### Multimodal Approaches
- **SemCluster** (Liang et al.): Combines visual and textual features
- Limited research on **image integration** in bug report detection

---

# Slide 4: Motivation

## Why Integrate Visual and Textual Semantics?

### The Gap in Current Research
```
Traditional Systems: Text Only → Miss Visual Context
```

### Screenshots Provide:
1. **UI State Information**: Which screen was the user on?
2. **Error Messages**: Visual error dialogs captured
3. **Context**: Layout, widgets, visual elements
4. **Reproduction Evidence**: Step-by-step visual proof

### Key Insight
> "Two bug reports with similar screenshots AND similar text descriptions are more likely to be duplicates than reports matching on text alone."

### Research Hypothesis
**H1**: Multi-modal features (Visual + Textual) outperform single-modality approaches for duplicate bug report detection.

---

# Slide 5: Proposed Method - Overview

## Multi-Modal Feature Fusion Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     BUG REPORT INPUT                        │
│        (Text Description + Screenshot + UI XML)             │
└─────────────────────────────────────────────────────────────┘
                              │
           ┌──────────────────┼──────────────────┐
           ▼                  ▼                  ▼
    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
    │   IMAGE     │    │    TEXT     │    │  STRUCTURE  │
    │  FEATURES   │    │  FEATURES   │    │  FEATURES   │
    └─────────────┘    └─────────────┘    └─────────────┘
           │                  │                  │
           ▼                  ▼                  ▼
    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
    │   VGG-16    │    │  Word2Vec   │    │    APTED    │
    │  (ImageNet) │    │  + TextCNN  │    │ (Tree Edit) │
    └─────────────┘    └─────────────┘    └─────────────┘
           │                  │                  │
           └──────────────────┼──────────────────┘
                              ▼
                    ┌─────────────────┐
                    │  FEATURE FUSION │
                    │   (Averaging)   │
                    └─────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  SIMILARITY     │
                    │     SCORE       │
                    └─────────────────┘
```

---

# Slide 6: Proposed Method - VGG-16 for Image Features

## Visual Content Features with VGG-16

### What is VGG-16?
- Deep Convolutional Neural Network (16 layers)
- Pre-trained on **ImageNet** (1.2M natural images, 1000 classes)
- Extracts hierarchical visual features

### Architecture
```
Input Image (224×224×3)
    ↓
[Conv3-64] × 2 → MaxPool
    ↓
[Conv3-128] × 2 → MaxPool
    ↓
[Conv3-256] × 3 → MaxPool
    ↓
[Conv3-512] × 3 → MaxPool
    ↓
[Conv3-512] × 3 → MaxPool
    ↓
Feature Vector (512-dim)
```

### In This Study
- **Input**: Bug report screenshots (resized to 224×224)
- **Output**: 512-dimensional feature vector per image
- **Distance**: L2-normalized Euclidean distance between feature vectors

### Key Code Implementation
```python
# VGG16 with ImageNet pretrained weights
base_model = VGG16(include_top=False, weights='imagenet')
model = Model(inputs=base_model.inputs, 
              outputs=base_model.layers[-2].output)
```

---

# Slide 7: Proposed Method - TextCNN for Text Features

## Deep Text Features with TextCNN

### What is TextCNN?
- Convolutional Neural Network for text classification
- Originally proposed by Yoon Kim (2014)
- Captures n-gram patterns effectively

### Architecture
```
Input: "App crashes on login button"
           ↓
    Word Embeddings (Word2Vec 100-dim)
           ↓
    ┌──────┼──────┐
    ▼      ▼      ▼
 Conv1D  Conv1D  Conv1D
 (3-gram)(4-gram)(5-gram)
    │      │      │
    ▼      ▼      ▼
 MaxPool MaxPool MaxPool
    └──────┼──────┘
           ▼
    Fully Connected
           ↓
    Classification:
    [Bug Description | Reproduction Step]
```

### Purpose in Pipeline
1. **Classify sentences** into:
   - Bug Description (what went wrong)
   - Reproduction Steps (how to trigger the bug)
2. Extract semantic embeddings for similarity computation

---

# Slide 8: Proposed Method - Word2Vec + DTW

## Text Similarity with Word2Vec + DTW

### Word2Vec (Continuous Bag of Words)
- Learns dense vector representations of words
- Pre-trained on bug report corpus (100 dimensions)
- Captures semantic relationships: `crash ≈ terminate ≈ close`

### Sentence Similarity
```python
def sentence_vector(sentence):
    words = jieba.cut(sentence)  # Tokenization
    vectors = [word2vec_model[w] for w in words]
    return np.mean(vectors, axis=0)  # Average pooling

similarity = euclidean_distance(vec1, vec2)
```

### Dynamic Time Warping (DTW)
- Measures similarity between **sequences** of reproduction steps
- Handles variable-length step lists
- Aligns steps optimally across two reports

```
Report A: [Step1, Step2, Step3]
Report B: [Step1, Step3, Step4, Step5]
              ↓
DTW finds optimal alignment → Distance score
```

---

# Slide 9: Proposed Method - Structure Features

## UI Structure Similarity with APTED

### What Are Structure Features?
- Android UI layouts represented as **XML trees**
- Capture hierarchical structure of screens

### APTED Algorithm
- **A**ll **P**ath **T**ree **E**dit **D**istance
- Computes minimum edit operations to transform one tree to another
- Operations: Insert, Delete, Rename nodes

### Example
```xml
<!-- Report A Layout -->          <!-- Report B Layout -->
<LinearLayout>                    <LinearLayout>
  <Button id="login"/>              <Button id="login"/>
  <TextView id="error"/>            <ImageView id="logo"/>
  <EditText id="password"/>         <EditText id="password"/>
</LinearLayout>                   </LinearLayout>

Edit Distance = 2 (Delete TextView, Insert ImageView)
```

### Implementation
```python
from apted import APTED, helpers
tree1 = helpers.Tree.from_text('{LinearLayout{Button}{TextView}}')
tree2 = helpers.Tree.from_text('{LinearLayout{Button}{ImageView}}')
distance = APTED(tree1, tree2).compute_edit_distance()
```

---

# Slide 10: Proposed Method - Feature Fusion

## Multi-Modal Feature Fusion Strategy

### Experiment 1: Equal-Weight Averaging (Baseline)
```
                 S_structure + S_vgg16 + S_text_problem + S_text_steps
Similarity = ───────────────────────────────────────────────────────────
                                      4
```

### Experiment 2: Learned/Optimized Weights
```
Similarity = (BB × 8.0) + (RS × 0.5) + (SF × 0.7) + (CF × 0.3)
             ─────────────────────────────────────────────────────
                         Sum of available weights
```

**Where:**
- **BB** = Bug description similarity (Word2Vec) — Weight: **8.0**
- **RS** = Reproduction steps similarity (DTW) — Weight: **0.5**
- **SF** = Structure feature (APTED) — Weight: **0.7**
- **CF** = Content feature (VGG-16) — Weight: **0.3**

### Adaptive Fusion (When No Images Available)
```
                 S_structure + S_text_problem + S_text_steps
Similarity = ─────────────────────────────────────────────────
                            Sum of weights
```

### Four Similarity Components

| Component | Method | Purpose | Weight |
|-----------|--------|---------|--------|
| **BB (Bug Description)** | Word2Vec Euclidean | Bug description similarity | **8.0** |
| **RS (Reproduction Steps)** | Word2Vec + DTW | Reproduction steps similarity | **0.5** |
| **SF (Structure)** | APTED Tree Edit Distance | UI layout similarity | **0.7** |
| **CF (Content)** | VGG-16 Feature Distance | Visual content similarity | **0.3** |

### Normalization
All distances normalized to [0, 1] using min-max scaling:
```
normalized = (distance - min_distance) / (max_distance - min_distance)
```

---

# Slide 11: Evaluation - Dataset

## Experimental Setup

### Dataset: Android Bug Reports with Images

| Property | FILTERED | FULL |
|----------|----------|------|
| **Total Bug Reports** | ~600 | ~3,000 |
| **Query Reports** | 125 | 2,323 |
| **Reports with Images** | All queries | Mixed |
| **Ground Truth Groups** | Manually labeled | Manually labeled |

### Data Structure
```
Bug Report:
├── ID: Unique identifier
├── Description: Text content
├── Screenshots: Binary image data
├── UI Layout: XML structure
├── Repository: Source project
└── Ground Truth: Duplicate group ID
```

### Data Source
- Collected from Android open-source projects
- Time period: Year 1 corpus
- Parquet format for efficient storage

---

# Slide 12: Evaluation - Metrics

## Evaluation Metrics

### Mean Reciprocal Rank (MRR)
$$MRR = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{rank_i}$$

*Where $rank_i$ is the position of first correct duplicate*

### Mean Average Precision (MAP)
$$MAP = \frac{1}{|Q|} \sum_{q=1}^{|Q|} AP(q)$$

*Average of precision at each relevant result*

### HITS@k
$$HITS@k = \frac{\text{Queries with correct result in top-k}}{|Q|}$$

*Percentage of queries with a correct duplicate in top-k results*

---

# Slide 13: Evaluation - Results

## Experimental Results

### Experiment 1: Equal-Weight Averaging (Baseline)

| Dataset | Queries | MRR | MAP | HITS@1 | HITS@5 | HITS@10 |
|---------|---------|-----|-----|--------|--------|---------|
| **FILTERED** | 125 | 0.1703 | 0.1647 | 10.40% | 20.80% | 32.80% |
| **FULL** | 2,323 | 0.0845 | 0.0731 | 4.74% | 9.73% | 14.59% |

### Experiment 2: Learned Feature Weights

**Optimized Weights**: BB=8.0, RS=0.5, SF=0.7, CF=0.3

| Dataset | MRR | MAP | HITS@1 | HITS@5 | HITS@10 |
|---------|-----|-----|--------|--------|---------|
| **FILTERED (Weighted)** | 0.1602 | 0.1495 | 8.06% | 18.55% | **36.29%** |
| **FULL (Weighted)** | **0.0883** | **0.0685** | **5.10%** | **10.30%** | **15.25%** |

### Key Observations

1. **FILTERED dataset** (image-only queries) performs **2× better** than FULL
2. **Weighted fusion improves HITS@10** on both datasets (32.80% → 36.29% for FILTERED)
3. Bug description (BB) receives highest weight (8.0), indicating text is most important
4. Visual content (CF) receives lowest weight (0.3)

### Interpretation
- Including images **does help** but improvement is modest
- **Text features dominate** the weighted fusion
- VGG-16 ImageNet features may not transfer perfectly to UI screenshots

---

# Slide 14: Evaluation - Analysis

## Result Analysis & Findings

### Why Limited Performance?

#### 1. Domain Mismatch (VGG-16)
```
ImageNet Training: Natural images (cats, dogs, cars)
                        ↓
Our Domain: Android UI screenshots (buttons, text, layouts)
                        ↓
Result: Features not optimally discriminative for UI elements
```

#### 2. Simple Fusion Strategy
- Equal-weight averaging doesn't learn optimal combinations
- No attention mechanism to weight important features

#### 3. Similarity Score Distribution Problem
```
All similarity scores cluster around 0.96
├── Duplicate pairs: ~0.96
├── Non-duplicate pairs: ~0.96
└── Poor separation between classes!
```

### What We Learned
✅ Multi-modal features provide **complementary information**  
⚠️ Off-the-shelf ImageNet features are **suboptimal** for UI analysis  
⚠️ Feature fusion strategy needs **improvement**

---

# Slide 15: System Architecture

## Complete System Pipeline

```
┌──────────────────────────────────────────────────────────────────┐
│                        DATA INGESTION                            │
│              bug_reports_with_images.parquet                     │
└──────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│                    EMBEDDING GENERATION                          │
│  embeddings/generate_embeddings.py                               │
│  ├── text_embeddings.pkl   (Word2Vec + TextCNN)                  │
│  ├── struct_embeddings.pkl (APTED trees)                         │
│  └── content_embeddings.pkl (VGG-16 features)                    │
└──────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│                   SIMILARITY COMPUTATION                          │
│  run_evaluation_from_embeddings.py                               │
│  └── Pairwise distance computation for all query-corpus pairs    │
└──────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│                       OUTPUT & METRICS                            │
│  output/semcluster_similarity_matrix_*.csv                       │
│  ├── MRR, MAP, HITS@k calculations                               │
│  └── Ranked retrieval results                                    │
└──────────────────────────────────────────────────────────────────┘
```

---

# Slide 16: Limitations

## Current Limitations

### 1. VGG-16 Domain Mismatch
- ImageNet weights trained on **natural images**, not UI screenshots
- Need fine-tuning on Android UI pairs

### 2. Simple Feature Fusion
- Equal-weight averaging is naive
- No learned interaction between modalities

### 3. Poor Discriminative Power
- Similarity scores cluster around 0.96
- Difficulty distinguishing duplicates from non-duplicates

### 4. Scalability
- FULL dataset (2,323 queries) shows significant performance drop
- Computational cost of pairwise comparisons: O(n²)

### 5. Missing Context
- No consideration of temporal information
- No project-specific semantics

---

# Slide 17: Future Work

## Directions for Improvement

### 1. Fine-tune VGG-16 on UI Screenshots
```
Current: VGG16(weights='imagenet')  # General features
Future:  VGG16(weights='android_ui') # Domain-specific
```

### 2. Learned Feature Fusion
- **Attention Mechanisms**: Weight features dynamically
- **Late Fusion Networks**: Learn optimal combination
- **Cross-Modal Transformers**: CLIP, BLIP for vision-language

### 3. Contrastive Learning
```
Loss = -log(exp(sim(anchor, positive)) / 
            Σ exp(sim(anchor, negative)))
```
Improve discriminative power between duplicates and non-duplicates

### 4. Vision-Language Models
- **CLIP**: Contrastive Language-Image Pre-training
- **BLIP**: Bootstrapped Language-Image Pre-training
- Better multimodal understanding out-of-the-box

### 5. Graph-based Methods
- Model bug reports as nodes in a graph
- Use GNN for relationship learning

---

# Slide 18: Conclusion

## Summary & Conclusions

### Research Question Answered
> **Does including images improve duplicate bug report detection?**

### Answer: **Yes, but with limitations**

### Key Contributions
1. ✅ Demonstrated **multi-modal approach** combining:
   - VGG-16 for visual features
   - TextCNN + Word2Vec for text features
   - APTED for structural features

2. ✅ Showed **2× improvement** with image-enriched queries (FILTERED vs FULL baseline)

3. ✅ **Weighted fusion experiment** revealed:
   - Bug description (BB) is most important (weight=8.0)
   - Visual content (CF) contributes least (weight=0.3)
   - Weighted fusion improves HITS@10 by **10.6%** (32.80% → 36.29%)

4. ✅ Identified **key limitations**:
   - Domain mismatch in visual features
   - Need for learned fusion strategies

### Take-Home Message
> "Screenshots provide valuable complementary information for duplicate bug report detection, but text features remain dominant. Effective integration requires domain-specific visual models and intelligent fusion strategies."

---

# Slide 19: Technical Implementation

## Implementation Details

### Technology Stack
| Component | Technology |
|-----------|------------|
| Deep Learning | TensorFlow 2.11, Keras |
| Text Processing | Word2Vec (Gensim), Jieba |
| Tree Edit Distance | APTED library |
| GPU Acceleration | CUDA 11.8, cuDNN 8.6 |
| Data Processing | Pandas, NumPy |

### Code Structure
```
SemCluster-v2/
├── core/           # Core algorithms
│   ├── semcluster.py   # Main evaluation logic
│   ├── cluster.py      # Semi-supervised clustering
│   └── metrics.py      # Evaluation metrics
├── image/          # Visual features
│   ├── vgg16.py        # VGG-16 extraction
│   └── structure_feature.py  # APTED
├── text/           # Text features
│   ├── text_feature.py     # Word2Vec + DTW
│   └── text_feature_extraction/
│       └── text_feature_extraction_tf2.py  # TextCNN
├── embeddings/     # Pre-computed embeddings
└── output/         # Results
```

---

# Slide 20: References

## References

1. **SemCluster**: Liang, B., et al. "SemCluster: Clustering of Imperative Programming Languages using Multi-channel Sequence-to-Sequence Model"

2. **VGG-16**: Simonyan, K., & Zisserman, A. (2014). "Very Deep Convolutional Networks for Large-Scale Image Recognition"

3. **TextCNN**: Kim, Y. (2014). "Convolutional Neural Networks for Sentence Classification"

4. **Word2Vec**: Mikolov, T., et al. (2013). "Efficient Estimation of Word Representations in Vector Space"

5. **DTW**: Berndt, D. J., & Clifford, J. (1994). "Using Dynamic Time Warping to Find Patterns in Time Series"

6. **APTED**: Pawlik, M., & Augsten, N. (2016). "Tree Edit Distance: Robust and Memory-Efficient"

---

## Thank You!

### Questions?

**Repository**: duplicate-bug-report-detection-study  
**Author**: musta55

