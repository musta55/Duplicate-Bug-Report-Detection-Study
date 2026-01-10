# SemCluster Testing Framework

End-to-end testing infrastructure for the SemCluster duplicate bug report detection system.

## Overview

This testing framework validates the complete SemCluster pipeline from embedding generation to similarity computation and metrics calculation across multiple projects.

## Quick Start

```bash
# Run the complete test pipeline
./test_pipeline_quick.sh

# Results will be generated in test_output/:
# - semcluster_similarity_matrix_FILTERED.csv
# - semcluster_similarity_matrix_FULL.csv
# - Projectwise_Retrieval_FILTERED.csv
# - Projectwise_Retrieval_FULL.csv
# - embeddings/ (3 pickle files per dataset: text, structure, content)
```

## Test Scenarios

### 1. FILTERED Dataset (100% Coverage)
- **Purpose**: Test with complete feature availability
- **Coverage**: 100% text + 100% images
- **Projects**: 3 projects (AndroidApp, WebUI, DesktopClient)
- **Reports**: 18 reports total (8+6+4)
- **Queries**: 12 queries (4 per project)
- **Pairs**: 60 query-corpus pairs
- **Ground Truth**: 24 relevant pairs

### 2. FULL Dataset (Sparse Images)
- **Purpose**: Test with realistic sparse image availability
- **Coverage**: 100% text + 10% images
- **Projects**: 5 projects (MobileApp, WebPortal, BackendAPI, Dashboard, AdminTool)
- **Reports**: 80 reports total (25+20+15+12+8)
- **Queries**: 25 queries (5 per project)
- **Pairs**: 375 query-corpus pairs
- **Ground Truth**: 50 relevant pairs
- **Fusion**: Adaptive (4-way for images, 2-way for text-only)

## Component Features

### Embeddings Format (Main Project Compatible)
The test generates pickle files in the same format as the main project:

**Text Embeddings** (`text_embeddings_*.pkl`):
- **BB (Bag of Words)**: Problem description (100-dim Word2Vec)
- **RS (Recurrence Sequence)**: Procedure steps (100-dim Word2Vec)
- Format: `{key: {'problem_vector': np.array, 'procedure_vectors': [np.array, ...]}}`
- Distance: Euclidean (BB), DTW (RS)

**Structure Embeddings** (`structure_embeddings_*.pkl`):
- **SF (Structure Feature)**: UI tree structure (APTED)
- Format: `{key: Tree}`
- Trees: 5 varied templates (complex, medium, simple, list, nested)
- Distance: Tree Edit Distance

**Content Embeddings** (`content_embeddings_*.pkl`):
- **CF (Content Feature)**: Visual features (VGG16, 512-dim)
- Format: `{key: [{'bbox': [x, y, w, h], 'vector': np.array}, ...]}`
- Distance: Euclidean

## Output Files

### Similarity Matrix CSV
Columns: `Project`, `query`, `corpus`, `score`, `rank`, `c_is_gt`, `BB`, `RS`, `SF`, `CF`

- **score**: Combined similarity score (average of available features)
- **rank**: Ranking within each query
- **c_is_gt**: Ground truth label (1=relevant, 0=not relevant)
- **BB/RS/SF/CF**: Individual component scores

### Projectwise Retrieval CSV
Columns: `Repository`, `#queries`, `#queries with image`, `MRR_Text Only`, `MRR_Text+Image`, `MRR_Diff`, `MRR_RI`, `MAP_Text Only`, `MAP_Text+Image`, `MAP_Diff`, `MAP_RI`, `Recall@1/5/10` (Text Only & Text+Image)

Metrics:
- **MRR**: Mean Reciprocal Rank
- **MAP**: Mean Average Precision
- **Recall@K**: Recall at K (K=1, 5, 10)
- **Diff**: Difference between Text+Image and Text-Only
- **RI**: Relative Improvement percentage

## File Structure

```
testing/
├── README.md                    # This file
├── test_scenarios.py            # Main test implementation (587 lines)
├── post_process_results.py      # Post-processing for metrics (290 lines)
├── test_pipeline_quick.sh       # Test runner script
└── test_output/                 # Generated results
    ├── semcluster_similarity_matrix_FILTERED.csv
    ├── semcluster_similarity_matrix_FULL.csv
    ├── Projectwise_Retrieval_FILTERED.csv
    ├── Projectwise_Retrieval_FULL.csv
    ├── embeddings/
    │   ├── text_embeddings_filtered.pkl       # BB + RS combined
    │   ├── structure_embeddings_filtered.pkl   # SF (trees)
    │   ├── content_embeddings_filtered.pkl     # CF (VGG16)
    │   ├── text_embeddings_full.pkl
    │   ├── structure_embeddings_full.pkl
    │   └── content_embeddings_full.pkl
    └── pipeline_test_report.json
```

## Implementation Details

### Test Pipeline (`test_scenarios.py`)

**Key Classes:**
- `SemClusterPipelineTester`: Main test orchestrator

**Key Methods:**
- `test_filtered_scenario()`: Test 100% coverage scenario
- `test_full_scenario()`: Test sparse image scenario
- `_compute_similarity_matrix()`: Compute all feature distances
- `_compute_similarity_matrix_adaptive()`: Adaptive fusion based on availability
- `_save_similarity_matrix_csv()`: Generate CSV with ground truth

**Ground Truth Simulation:**
- For each query, marks positions [1] and [2] as relevant (c_is_gt=1)
- Ensures multiple queries have relevant items at different ranks
- Enables proper MRR/MAP/Recall calculation

### Post-Processing (`post_process_results.py`)

**Key Functions:**
- `add_component_scores()`: Calculate BB, RS, SF, CF distances from embeddings
- `generate_projectwise_metrics()`: Calculate MRR, MAP, Recall@K per project
- Uses `core.metrics.Metrics` class (matching main project implementation)

**Ranking Strategies:**
- Text-Only: Average of BB + RS
- Text+Image: Average of BB + RS + SF + CF (when available)

## Example Results

### FILTERED Dataset (3 Projects)
```
Repository      Queries  MRR   MAP   Recall@5
AndroidApp      4        0.46  0.44  0.75
DesktopClient   4        0.63  0.65  1.00
WebUI           4        0.42  0.45  1.00
```

### FULL Dataset (5 Projects)
```
Repository      Queries  MRR   MAP   Recall@5
AdminTool       5        0.77  0.61  0.90
BackendAPI      5        0.42  0.34  0.40
Dashboard       5        0.52  0.48  1.00
MobileApp       5        0.13  0.13  0.10
WebPortal       5        0.62  0.40  0.60
```

## Dependencies

```python
# Core
numpy
pandas
scipy

# Text
gensim (Word2Vec)

# Images
tensorflow (VGG16)
opencv-python

# Structure
apted (Tree Edit Distance)

# Project modules
core.metrics (Metrics class)
embeddings.generate_embeddings
```

## Usage in Development

### Run Full Test
```bash
cd /path/to/SemCluster-v2-clean
rm -rf test_output/
./testing/test_pipeline_quick.sh
```

### Check Results
```bash
# View similarity matrices
head test_output/semcluster_similarity_matrix_FILTERED.csv
head test_output/semcluster_similarity_matrix_FULL.csv

# View projectwise metrics
cat test_output/Projectwise_Retrieval_FILTERED.csv
cat test_output/Projectwise_Retrieval_FULL.csv

# Verify embeddings
ls -lh test_output/embeddings/
```

### Customize Tests

Edit `test_scenarios.py` to modify:
- Number of projects (lines 89-93 for FILTERED, 218-224 for FULL)
- Number of reports per project
- Image coverage percentage
- Tree structure templates (lines 107-111)

## Validation

The test framework validates:
1. ✅ Embedding generation in main project format (text, structure, content)
2. ✅ Pickle file creation (3 files per dataset)
3. ✅ Similarity matrix computation with adaptive fusion
4. ✅ Component score calculation (BB, RS, SF, CF extracted from embeddings)
5. ✅ Ground truth simulation for realistic evaluation
6. ✅ Projectwise metrics calculation using core.metrics.Metrics
7. ✅ Multi-project evaluation with varying performance

## Notes

- **Embedding Format**: Matches main project (text/structure/content) - NO separate BB/RS/SF/CF files
- **Ground Truth**: Simulated pattern (positions [1] and [2]) - replace with real data for production
- **Tree Structures**: 5 templates provide variation - extend for more diversity
- **Fusion Strategy**: Adaptive based on feature availability (4-way, 3-way, or 2-way)

## Troubleshooting

**No metrics calculated (all 0.0)**
- Ensure ground truth pairs exist (c_is_gt=1)
- Check CSV has multiple queries

**SF showing 0**
- Verify tree structures are varied (not all identical)
- Check APTED library is installed

**Missing component scores**
- Run post_process_results.py after test_scenarios.py
- Use test_pipeline_quick.sh to run both automatically

## Future Enhancements

1. Add real ground truth data from labeled datasets
2. Extend to more UI tree structure templates
3. Add cross-project evaluation metrics
4. Include statistical significance tests
5. Add visualization of results (charts, heatmaps)
