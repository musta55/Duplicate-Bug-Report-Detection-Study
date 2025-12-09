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
- **Total Queries**: 125
- **Total Projects**: 23
- **Queries with Images**: 125 (100%)

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
| **MRR** | Measures how high the first correct duplicate is ranked, on average. Higher is better (max = 1.0). | **0.1320** |
| **MAP** | Rewards system for ranking all true duplicates highly. Accounts for multiple duplicates per query. | **0.1026** |
| **HITS@k** | Percentage of queries where a true duplicate is in the top k results. | See below |

#### HITS@k Breakdown
| k | HITS@k | Queries Found |
|---|--------|---------------|
| 1 | 0.0692 | 18/260 |
| 2 | 0.1077 | 28/260 |
| 3 | 0.1346 | 35/260 |
| 4 | 0.1462 | 38/260 |
| 5 | 0.1808 | 47/260 |
| 6 | 0.2000 | 52/260 |
| 7 | 0.2038 | 53/260 |
| 8 | 0.2192 | 57/260 |
| 9 | 0.2346 | 61/260 |
| 10 | 0.2538 | 66/260 |

### Summary

- **Evaluation completed for 260 queries across 23 projects**
- **MRR of 0.1320** means the first correct duplicate appears at rank ~7.6 on average
- **MAP of 0.1026** indicates overall precision across all relevant results is 10.3%
- **HITS@1 of 0.0692** shows 6.9% of queries find a duplicate in the top-1 result
- **HITS@10 of 0.2538** shows 25.4% of queries find a duplicate in the top-10 results

### Key Findings
1. ✅ **Data Completeness**: All 3 Aegis queries (450, 772, 1085) are correctly included in the evaluation
2. ✅ **Fixed Issues**: Resolved ID collision bug and min_duplicates filter that was excluding queries
3. ⚠️ **Performance**: Retrieval performance is moderate with 25.4% of queries finding duplicates in top-10
4. 📊 **Multi-modal Features**: Combines image (structure + content) and text (problem + reproduction) features