# Duplicate Bug Report Detection Study

Empirical study examining whether including images (screenshots) enhances duplicate bug report detection in Android applications.

## Research Question

**Does including images as an additional source of information improve duplicate bug detection?**

## Key Findings

### Overall Performance

| Dataset | Queries | MRR | MAP | HITS@1 | HITS@5 | HITS@10 |
|---------|---------|-----|-----|--------|--------|---------|
| FILTERED | 125 | 0.1703 | 0.1647 | 10.40% | 20.80% | 32.80% |
| FULL | 2323 | 0.0845 | 0.0731 | 4.74% | 9.73% | 14.59% |

## System Architecture

### Multi-Modal Features

SemCluster combines four types of features:

1. **Structure Features**: UI layout similarity from XML layouts
2. **VGG16 (Image Content)**: Deep visual features from screenshots
   - Uses ImageNet pretrained weights
3. **Word2Vec + DTW**: Text similarity for bug descriptions and reproduction steps
4. **TextCNN**: Deep text embeddings

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
├── core/                        # Core logic and utilities
│   ├── cluster.py               # Clustering algorithms
│   ├── configure.py             # Configuration utilities
│   ├── semcluster.py            # Core SemCluster logic
│   └── metrics.py               # Evaluation metrics (MRR, MAP, HITS@k)
│
├── embeddings/                  # Pre-computed embeddings and generation script
│   ├── generate_embeddings.py  # Script to generate embeddings
│   ├── text_embeddings.pkl     # BB + RS text features
│   ├── structure_embeddings.pkl # SF structure features
│   └── content_embeddings.pkl  # CF visual features
│
├── run_evaluation_from_embeddings.py # Main evaluation script
│
├── image/                       # Image feature extraction
│   ├── vgg16.py                # VGG16 visual features (GPU-accelerated)
│   ├── structure_feature.py    # UI layout features (Tree Edit Distance)
│   ├── content_feature.py      # Image content analysis
│   ├── image_main.py           # Image processing entry point
│   ├── vgg16_stub.py           # VGG16 model stub
│   └── widget.py               # Widget extraction utilities
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
├── Dataset/
│   ├── Overall - FILTERED_trimmed_year_1_corpus_with_gt.csv  # 125 queries (100% text + 100% images)
│   ├── Overall - FULL_trimmed_year_1_corpus_with_gt.csv      # 2,323 queries (100% text + 10-12% images)
│   └── bug_reports_with_images.parquet
│
├── output/                      # Evaluation results
│   ├── semcluster_similarity_matrix_FILTERED.csv
│   ├── semcluster_similarity_matrix_FULL.csv
│   ├── Projectwise_Retrieval_Filtered.csv
│   └── Projectwise_Retrieval_Full.csv
│
└── testing/                     # Testing infrastructure
    ├── test_pipeline_quick.sh  # Quick pipeline test script
    ├── test_scenarios.py       # Test scenario generator
    ├── post_process_results.py # Results post-processing
    └── README.md               # Testing documentation
```

See [DATASET_INFO.md](DATASET_INFO.md) for detailed dataset coverage information.

## Testing

### Pipeline Testing

Validate the core pipeline: embedding generation → similarity computation → metrics calculation

```bash
# Run complete pipeline test (FILTERED + FULL scenarios)
cd testing
./test_pipeline_quick.sh

# View results
cat ../test_output/pipeline_test_report.json
cat ../test_output/Projectwise_Retrieval_FILTERED.csv
cat ../test_output/Projectwise_Retrieval_FULL.csv
```

**Test Coverage:**
- ✅ Embedding generation: text (BB+RS), structure (SF), content (CF)
- ✅ Pickle file creation and loading
- ✅ FILTERED scenario: 100% text + 100% images (4-way fusion)
- ✅ FULL scenario: 100% text + 10-12% images (adaptive 2/3/4-way fusion)
- ✅ Distance metrics: Euclidean (text/content) + Tree Edit Distance (structure)
- ✅ Metrics calculation: MRR, MAP, Recall@1/5/10 per project
- ✅ Component scores: BB, RS, SF, CF extraction

See [testing/README.md](testing/README.md) for complete testing documentation.

---

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

### Quick Start

**For complete evaluation from scratch (recommended):**

```bash
# FILTERED dataset (queries with images only) - 125 queries
python run_parquet_evaluation.py --dataset FILTERED

# FULL dataset (all queries, adaptive fusion) - 2,323 queries  
python run_parquet_evaluation.py --dataset FULL
```

This will:
1. Load bug reports from parquet
2. Extract images and UI layouts
3. Run feature extraction (text + image)
4. Compute similarities
5. Generate evaluation results in `output/`

**See [README_EVALUATION.md](README_EVALUATION.md) for detailed guide on all evaluation scripts.**

### Advanced Options

```bash
# Process specific number of queries
python run_parquet_evaluation.py --dataset FILTERED --n-queries 10

# Process specific query ID
python run_parquet_evaluation.py --dataset FILTERED --query-id 450
```

### Output Files

The evaluation generates:

- **Similarity Matrix**: `output/semcluster_similarity_matrix_{FILTERED|FULL}.csv`
  - Format: `Project,query,corpus,score,rank,c_is_gt,BB,RS,SF,CF`
  - Contains ranked candidates for each query with component scores
  
- **Projectwise Metrics**: `output/Projectwise_Retrieval_{FILTERED|FULL}.csv`
  - Per-project retrieval metrics (MRR, MAP, Recall@K)

- **Intermediate Files**:
  - `file/pic_file_parquet_{filtered|full}/` - Extracted images
  - `file/xml_file_parquet_{filtered|full}/` - UI layout XMLs
  - `file/label_file_parquet/evaluation_{filtered|full}.csv` - Evaluation manifest

### Results

The console output will show:
```
EVALUATION COMPLETE
======================================================================
Queries evaluated: 125
Total unique reports: 1,234
Images extracted: 450
Output file: output/semcluster_similarity_matrix_FILTERED.csv
Retrieval Performance:
  MRR: 0.1703
  MAP: 0.1647
  HITS@1:  10.40%
  HITS@5:  20.80%
  HITS@10: 32.80%
======================================================================
```

Output: `output/semcluster_similarity_matrix_FULL.csv`

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
4. **Absolute Performance**: HITS@10 remains low (25.38% for FILTERED, 14.99% for FULL)

## Future Work

1. Fine-tune VGG16 on Android UI screenshot pairs
2. Use learned fusion (e.g., attention mechanisms, learned weights)
3. Try vision-language models (CLIP, BLIP) for better multimodal understanding
4. Contrastive learning to improve discriminative power

## Citation

SemCluster paper:
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
