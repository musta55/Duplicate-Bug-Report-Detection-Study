# Retrieval Evaluation Report
## Filtered Dataset Evaluation

### 1. Ground Truth Data
The following table represents a sample of the ground truth data used for evaluation, extracted from the year 1 corpus.

| Project | Query ID | Ground Truth IDs | GT Size | Corpus Size | Has Image |
|---------|----------|-----------------|---------|-------------|-----------|
| Aegis | 450 | [418] | 1 | 192 | True |
| Aegis | 772 | [759] | 1 | 211 | True |
| Aegis | 1085 | [1031,  1043] | 2 | 161 | True |
| Anki-Android | 3821 | [3761] | 1 | 2808 | True |
| Anki-Android | 4558 | [4557] | 1 | 275 | True |
| Anki-Android | 7450 | [7369] | 1 | 765 | True |
| Anki-Android | 8010 | [5793] | 1 | 907 | True |
| Anki-Android | 8032 | [7369,  7450] | 2 | 915 | True |
| Anki-Android | 8139 | [7369,  7450,  8032] | 3 | 942 | True |
| Anki-Android | 8150 | [6614,  8119] | 2 | 945 | True |

#### Dataset Summary
- **Total Queries**: 124
- **Total Projects**: 23
- **Queries with Images**: 124 (100%)

### 2. Similarity Metrics
**Example**: Ranked retrieval results for query Aegis #1085

| Project | Query | Corpus ID | Score | Rank | Is Ground Truth? |
|---------|-------|-----------|-------|------|------------------|
| Aegis | 1085 | 772 | 0.4971 | 1 | No (0) |
| Aegis | 1085 | 1043 | 0.4944 | 2 | Yes (1) |
| Aegis | 1085 | 11568 | 0.4938 | 3 | No (0) |
| Aegis | 1085 | 11566 | 0.4936 | 4 | No (0) |
| Aegis | 1085 | 995 | 0.4936 | 5 | No (0) |

### 6.1. Inputs for Retrieval Evaluation

| Input | Source | Description |
|-------|--------|-------------|
| Similarity Matrix | SemCluster Pipeline | Matrix of similarity scores (0-1) between query and corpus reports. Higher scores indicate higher similarity. Generated from 4 features: structure, content, problem text, and reproduction text. |
| Ground Truth | Dataset/Overall - FILTERED_trimmed_year_1_corpus_with_gt.csv | Maps each query to its known duplicate reports. Extracted from the 'ground_truth' column in the FILTERED dataset. |

#### Dataset Statistics
- Total query reports: 125
- Reports with images: 125 (100%)
- Unique projects: 23
- Average corpus size: 549.9 candidates per query
- Average ground truth size: 1.34 duplicates per query

### 6.2. Process for Calculating Metrics

| Step | Action | Description |
|------|--------|-------------|
| 1 | Select a Query Report | Iterate through each of the 125 bug reports in the dataset. |
| 2 | Generate Ranked List | Use the Similarity Matrix to sort all corpus candidates from most similar (highest score) to least similar (lowest score). |
| 3 | Identify True Duplicates | Look up the query's 'ground_truth' column to find all known duplicates (1-2 per query on average). |
| 4 | Calculate Metrics | Compare the ranked list against true duplicates to calculate MRR, MAP, and HITS@k. |

#### Feature Extraction Pipeline
- **Text Features**: TextCNN embeddings for problem description and reproduction steps
- **Structure Features**: Tree Edit Distance on UI layout hierarchies extracted from screenshots
- **Content Features**: VGG16 deep features on widget-level image crops
- **Adaptive Averaging**: Combine all 4 features with equal weights for reports with images; use only text features for reports without images

### 6.3. Outputs of Retrieval Evaluation

| Metric | Description | Value |
|--------|-------------|-------|
| **MRR** | Measures how high the first correct duplicate is ranked, on average. Higher is better (max = 1.0). | **0.1714** |
| **MAP** | Rewards system for ranking all true duplicates highly. Accounts for multiple duplicates per query. | **0.1658** |
| **HITS@k** | Percentage of queries where a true duplicate is in the top k results. | See below |

#### HITS@k Breakdown
| k | HITS@k |
|---|--------|
| 1 | 10.48% |
| 2 | 12.10% |
| 3 | 15.32% |
| 4 | 16.94% |
| 5 | 20.97% |
| 6 | 23.39% |
| 7 | 26.61% |
| 8 | 29.84% |
| 9 | 31.45% |
| 10 | 33.06% |

### Summary

- **Evaluation completed for 124 queries across 23 projects**
- **MRR of 0.1714** indicates the first correct duplicate appears substantially higher on average compared to previous runs
- **MAP of 0.1658** reflects improved average precision over the evaluated queries
- **HITS@1 of 10.48%** shows 13/124 queries find a duplicate in the top-1 result
- **HITS@10 of 33.06%** shows 41/124 queries find a duplicate in the top-10 results

### Key Findings

- 1. ✅ **Data Completeness**: The FILTERED evaluation uses only queries with valid screenshots (100% have images in this subset)
- 2. ✅ **Fixed Issues**: Resolved ID collision and filtering inconsistencies prior to evaluation
- 3. ⚠️ **Performance**: Retrieval performance on the FILTERED set is moderate — image-rich queries yield better performance here compared to earlier reported numbers
- 4. 📊 **Multi-modal Features**: Combines image (structure + content) and text (problem + reproduction) features

---

## Impact of Images within the FULL Dataset (Unweighted)

To evaluate the impact of images under the unweighted (simple-average) fusion, we computed retrieval metrics on the FULL similarity outputs and partitioned queries into those that contain screenshots and those that do not (as indicated in the FULL ground-truth CSV).

| Subset | # Queries | MRR | MAP | HITS@10 |
|--------|-----------:|:---:|:---:|:-------:|
| All (FULL, unweighted) | 1961 | 0.0864 | 0.0664 | 15.09% |
| With Images | 297 | 0.0747 | 0.0526 | 12.46% |
| Without Images | 1664 | 0.0885 | 0.0688 | 15.56% |

Notes:
- The unweighted setting uses a simple average across available feature distances (structure, content, bug behavior, reproduction steps), and falls back to text-only averaging when images are absent.
- These results show that, for the FULL corpus under simple averaging, queries without screenshots slightly outperform those with screenshots on these retrieval metrics. This may reflect the sparsity and noise in the FULL image set (many placeholder/invalid images) and motivates using filtered image subsets or alternative weighting strategies when incorporating visual features.