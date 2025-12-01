# Duplicate Bug Report Detection Study

Empirical study examining whether including images (screenshots) enhances duplicate bug report detection in Android applications.

## Research Question

**Does including images as an additional source of information improve duplicate bug detection?**

## Key Findings

### Impact of Images on Retrieval Performance

Comparing queries **with images** vs **without images** in the FULL dataset:

| Metric | With Images (103 queries) | Without Images (618 queries) | Improvement |
|--------|---------------------------|------------------------------|-------------|
| MRR | 0.1892 | 0.0769 | **+146%** |
| HITS@1 | 15.5% | 5.7% | **+171%** |
| HITS@5 | 22.3% | 7.3% | **+205%** |
| HITS@10 | 22.3% | 8.6% | **+159%** |

**Conclusion:** Images provide approximately **2.5× improvement** in duplicate detection performance.

### Overall Performance

| Dataset | Queries | MRR | HITS@1 | HITS@10 |
|---------|---------|-----|--------|---------|
| FILTERED | 92 (100% with images) | 0.1725 | 9.8% | 29.3% |
| FULL | 1034 (10% with images) | 0.0918 | 7.0% | 10.7% |

## System Architecture

### Multi-Modal Features

SemCluster combines four types of features:

1. **Structure Features**: UI layout similarity from XML layouts
2. **VGG16 (Image Content)**: Deep visual features from screenshots
   - Uses ImageNet pretrained weights (original paper weights unavailable)
3. **Word2Vec + DTW**: Text similarity for bug descriptions and reproduction steps
4. **TextCNN**: Deep text embeddings (added in this study)

### Feature Fusion

Simple averaging with adaptive weights:
- Reports with images: (Structure + VGG16 + Text1 + Text2) / 4
- Text-only reports: (Structure + Text1 + Text2) / 3

## Project Structure

```
.
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── environment.yml              # Conda environment specification
│
├── cluster.py                   # Clustering algorithms
├── configure.py                 # Configuration utilities
├── main.py                      # Original evaluation script
├── metrics.py                   # Evaluation metrics (MRR, HITS@k)
│
├── run_parquet_evaluation.py    # Main evaluation script (supports FILTERED/FULL)
├── run_filtered_eval.sh         # Run FILTERED dataset evaluation
├── run_full_eval.sh             # Run FULL dataset evaluation
│
├── image/                       # Image feature extraction
│   ├── vgg16.py                # VGG16 visual features (GPU-accelerated)
│   ├── structure_feature.py    # UI layout features
│   ├── content_feature.py      # Image content analysis
│   └── ...
│
├── text/                        # Text feature extraction
│   ├── text_main.py            # Text processing entry point
│   ├── text_feature.py         # Word2Vec + DTW similarity
│   └── text_feature_extraction/
│       ├── text_feature_extraction_tf2.py  # TextCNN (TF 2.11, GPU)
│       ├── dtw.py              # Dynamic Time Warping
│       ├── bugdata_format_model_100  # Pre-trained Word2Vec model
│       └── runs/TextCNN_model/ # Pre-trained TextCNN checkpoints
│
├── Overall - FILTERED_trimmed_year_1_corpus_with_gt.csv  # FILTERED ground truth
├── Overall - FULL_trimmed_year_1_corpus_with_gt.csv      # FULL ground truth
│
├── semcluster_similarity_matrix_FILTERED.csv  # FILTERED results (8,372 rows)
└── semcluster_similarity_matrix_FULL.csv      # FULL results (528,896 rows)
```

## Setup

### Prerequisites

- Python 3.9
- CUDA 11.8 (for GPU acceleration)
- cuDNN 8.6

### Installation

**Option 1: Using Conda (Recommended)**

```bash
conda env create -f environment.yml
conda activate semcluster_gpu
```

**Option 2: Using pip**

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Data Requirements

Place your bug report dataset in parquet format at:
- `file/label_file/evaluate.parquet`

The parquet file should contain columns:
- `id`: Bug report ID
- `description`: Text description
- `imgs`: Screenshots (binary data)
- `valid_image`: Boolean flag
- `repo`: Repository name

## Usage

### Run FILTERED Evaluation (92 reports, all with images)

```bash
bash run_filtered_eval.sh
```

Output: `semcluster_similarity_matrix_FILTERED.csv`

### Run FULL Evaluation (1034 queries, mixed)

```bash
bash run_full_eval.sh
```

Output: `semcluster_similarity_matrix_FULL.csv`

Expected runtime: ~25 hours on NVIDIA A40 GPU

### Custom Evaluation

```bash
python run_parquet_evaluation.py --dataset {FILTERED|FULL}
```

## Results Format

Output CSV contains pairwise similarity scores:

| Column | Description |
|--------|-------------|
| Project | Repository name |
| query | Query bug report ID |
| corpus | Candidate bug report ID |
| score | Similarity score (0-1, higher = more similar) |
| rank | Rank of this candidate for the query |
| c_is_gt | 1 if candidate is ground truth duplicate, 0 otherwise |

## Limitations

1. **VGG16 Domain Mismatch**: ImageNet weights trained on natural images, not UI screenshots
2. **Simple Fusion**: Equal-weight averaging doesn't learn optimal feature combination
3. **Poor Discrimination**: All similarity scores cluster around 0.96 (poor separation between duplicates and non-duplicates)
4. **Absolute Performance**: Even with images, HITS@10 is only 22% (78% failure rate)

## Future Work

1. Fine-tune VGG16 on Android UI screenshot pairs
2. Use learned fusion (e.g., attention mechanisms, learned weights)
3. Try vision-language models (CLIP, BLIP) for better multimodal understanding
4. Contrastive learning to improve discriminative power

## Citation

Original SemCluster paper:
```
Liang, B., et al. (Year). "SemCluster: Clustering of Imperative Programming Language 
Fragments using Multi-channel Sequence-to-Sequence Model"
```

This study extends SemCluster by:
- Migrating to TensorFlow 2.11 with GPU support
- Adding TextCNN for improved text embeddings
- Empirically evaluating the impact of images on duplicate detection

## License

[Add your license here]

## Contact

[Add contact information]
