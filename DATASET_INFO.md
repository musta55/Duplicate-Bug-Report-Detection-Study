# SemCluster Dataset Information

## Dataset Variants

### FILTERED Dataset
**File:** `Dataset/Overall - FILTERED_trimmed_year_1_corpus_with_gt.csv`

- **Size:** 125 queries
- **Text Coverage:** 100% (all reports have descriptions)
- **Image Coverage:** 100% (all reports have screenshots)
- **Use Case:** Training, testing, and evaluation where visual features are required

### FULL Dataset
**File:** `Dataset/Overall - FULL_trimmed_year_1_corpus_with_gt.csv`

- **Size:** 2,323 queries
- **Text Coverage:** 100% (all reports have descriptions)
- **Image Coverage:** 10-12% (only subset of reports have screenshots)
- **Use Case:** Large-scale evaluation, text-heavy analysis

## Feature Coverage by Dataset

| Feature Type | FILTERED | FULL | Model/Method |
|--------------|----------|------|--------------|
| **Text (Bug Description)** | 100% | 100% | TextCNN + Word2Vec (100-dim) |
| **Text (Reproduction Steps)** | 100% | 100% | TextCNN + Word2Vec (100-dim) + DTW |
| **Structure (UI Layout)** | 100% | 10-12% | XML → APTED Tree Edit Distance |
| **Content (Visual Features)** | 100% | 10-12% | VGG16 (512-dim widget vectors) |

## Multi-Modal Fusion Strategy

SemCluster uses **adaptive feature fusion** based on available modalities:

```python
# 4-way fusion (when image available)
similarity = (text_desc + text_steps + structure + content) / 4

# 3-way fusion (when no content features)
similarity = (text_desc + text_steps + structure) / 3

# 2-way fusion (when no image at all)
similarity = (text_desc + text_steps) / 2
```

### FILTERED Dataset
- Most comparisons use **4-way fusion** (all features available)
- Expected to leverage full multi-modal capabilities

### FULL Dataset
- ~10-12% comparisons use **4-way fusion** (both reports have images)
- ~20-24% comparisons use **3-way fusion** (one report has image)
- ~65-80% comparisons use **2-way fusion** (neither has image)
- Text features dominate, visual features provide additional signal when available

## Embedding File Sizes

### FILTERED Dataset Embeddings
```
text_embeddings.pkl       ~500KB   (125 reports × 100-dim vectors)
structure_embeddings.pkl  ~300KB   (125 reports × tree structures)
content_embeddings.pkl    ~5-10MB  (125 reports × widget features)
```

### FULL Dataset Embeddings
```
text_embeddings.pkl       ~10-15MB  (2,323 reports × 100-dim vectors)
structure_embeddings.pkl  ~1-2MB    (230-280 reports × tree structures)
content_embeddings.pkl    ~2-5MB    (230-280 reports × widget features)
```

## Dataset Selection Guidelines

### Use FILTERED When:
- ✅ Developing/testing visual feature extraction
- ✅ Evaluating multi-modal fusion effectiveness
- ✅ Comparing all 4 feature types
- ✅ Quick iteration during development
- ✅ Training models that require visual data

### Use FULL When:
- ✅ Large-scale evaluation with realistic data distribution
- ✅ Testing robustness to missing visual features
- ✅ Evaluating text-based methods
- ✅ Analyzing performance across different data availability scenarios
- ✅ Production-like evaluation (real-world data is sparse)

## Data Sources

### Parquet File
**File:** `Dataset/bug_reports_with_images.parquet`
- Contains binary image data embedded in the parquet format
- Includes metadata: `id`, `repo_name`, `description`, `title`, `image` (binary)

### XML Layout Files
**Directories:**
- `file/xml_file_parquet_filtered/` - Layout files for FILTERED dataset
- `file/xml_file_parquet_full/` - Layout files for FULL dataset
- Format: `layout{id}.xml` with widget bounding boxes

### Generated Images (Optional)
**Directories:**
- `file/pic_file_parquet_filtered/` - Extracted images (FILTERED)
- `file/pic_file_parquet_full/` - Extracted images (FULL)
- Format: `report_img_{id}.png`

## Performance Expectations

### FILTERED Dataset
- **MRR:** ~0.65-0.70 (baseline)
- **MAP:** ~0.60-0.65
- **HITS@10:** ~0.85-0.90
- All 4 features contribute to similarity

### FULL Dataset
- **MRR:** ~0.45-0.55 (more challenging)
- **MAP:** ~0.40-0.50
- **HITS@10:** ~0.70-0.80
- Text features dominate, visual features provide boost for 10-12% of pairs

## Notes

1. **Image Coverage in FULL:** The 10-12% image coverage in FULL dataset reflects real-world bug report data where screenshots are optional and not always provided.

2. **Adaptive Fusion:** The system automatically handles missing features by averaging only available similarities, ensuring graceful degradation.

3. **Evaluation Impact:** When evaluating on FULL dataset, most gains come from text features, but visual features can provide significant boost for the subset of image-containing reports.

4. **Storage Optimization:** The sparse image coverage in FULL dataset means structure and content embeddings consume less disk space relative to text embeddings.
