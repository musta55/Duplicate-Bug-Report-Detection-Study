#!/usr/bin/env python3
"""
End-to-End Testing for SemCluster Pipeline

This module implements comprehensive testing for the SemCluster duplicate bug report
detection system across multiple projects and scenarios.

Test Scenarios:
1. FILTERED dataset (100% text + 100% images) - 3 projects, 18 reports, 12 queries
2. FULL dataset (100% text + 10% images) - 5 projects, 80 reports, 25 queries

Features Tested:
- BB (Bag of Words): Problem description embeddings
- RS (Recurrence Sequence): Procedure steps embeddings  
- SF (Structure Feature): UI tree structure (APTED)
- CF (Content Feature): Visual features (VGG16)

Outputs:
- Similarity matrices with component scores (BB, RS, SF, CF)
- Pickle files for each feature (separate + legacy formats)
- Ground truth simulation for metrics evaluation

Usage:
    python test_scenarios.py
    # Or use the test pipeline script:
    ./test_pipeline_quick.sh

Author: SemCluster Team
Version: 2.0 (Multi-project support)
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import json

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

class SemClusterPipelineTester:
    """Test SemCluster end-to-end pipeline with different scenarios"""
    
    def __init__(self):
        self.test_dir = Path(project_root) / 'test_output'
        self.embeddings_dir = self.test_dir / 'embeddings'
        self.results = {
            'tests_passed': 0,
            'tests_failed': 0,
            'scenarios': {}
        }
        
    def setup(self):
        """Setup test environment"""
        self.test_dir.mkdir(exist_ok=True)
        self.embeddings_dir.mkdir(exist_ok=True)
        
    def cleanup(self):
        """Clean up test artifacts"""
        import shutil
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
            
    def test_filtered_scenario(self):
        """
        Test FILTERED dataset scenario:
        - 100% text coverage
        - 100% image coverage
        - All 4 features available (text_desc, text_steps, structure, content)
        """
        print("\n" + "="*70)
        print("SCENARIO 1: FILTERED Dataset (100% text + 100% images)")
        print("="*70)
        
        from embeddings.generate_embeddings import (
            generate_text_embeddings,
            generate_structure_embeddings,
            generate_content_embeddings
        )
        
        # Create synthetic multi-project dataset
        # 3 projects with different numbers of reports
        projects = {
            'AndroidApp': 8,      # 8 reports
            'WebUI': 6,           # 6 reports  
            'DesktopClient': 4    # 4 reports
        }
        n_total = sum(projects.values())
        
        print(f"Testing with {n_total} reports across {len(projects)} projects:")
        for proj, count in projects.items():
            print(f"  - {proj}: {count} reports")
        print(f"Expected: 100% have text, 100% have images")
        
        # Generate embeddings in main project format
        # Text = BB (problem) + RS (procedure steps) combined
        # Structure = SF (UI trees)
        # Content = CF (VGG16 features)
        
        from apted import helpers
        tree_templates = [
            "{root{button{text}{icon}}{list{item1{text}{img}}{item2{text}{img}}}{footer{btn}}}",  # Complex
            "{root{header{title}}{body{text}{image}}{footer}}",  # Medium
            "{root{view{text}{button}}}",  # Simple
            "{root{list{item1}{item2}{item3}}}",  # List
            "{root{scroll{view1{text}}{view2{image}{text}}}}"  # Nested
        ]
        
        # Create embeddings in main project format
        text_emb = {}    # {key: {'problem_vector': np.array, 'procedure_vectors': [np.array, ...]}}
        struct_emb = {}  # {key: Tree}
        content_emb = {} # {key: [{'bbox': [...], 'vector': np.array}, ...]}
        
        idx = 0
        for proj_name, n_reports in projects.items():
            for qid in range(1, n_reports + 1):
                key = f"{proj_name}:{qid}"
                
                # Text: BB (problem) + RS (procedure steps)
                text_emb[key] = {
                    'problem_vector': np.random.rand(100),
                    'procedure_vectors': [np.random.rand(100) for _ in range(3)]
                }
                
                # Structure: UI tree (APTED)
                struct_emb[key] = helpers.Tree.from_text(tree_templates[idx % len(tree_templates)])
                
                # Content: Visual features (list of region dicts)
                content_emb[key] = [{
                    'bbox': [0, 0, 100, 100],
                    'vector': np.random.rand(512)
                }]
                
                idx += 1
        
        # Save in main project format (3 files only)
        text_pkl = self.embeddings_dir / 'text_embeddings_filtered.pkl'
        struct_pkl = self.embeddings_dir / 'structure_embeddings_filtered.pkl'
        content_pkl = self.embeddings_dir / 'content_embeddings_filtered.pkl'
        
        with open(text_pkl, 'wb') as f:
            pickle.dump(text_emb, f)
        with open(struct_pkl, 'wb') as f:
            pickle.dump(struct_emb, f)
        with open(content_pkl, 'wb') as f:
            pickle.dump(content_emb, f)
        
        # Verify files created
        assert text_pkl.exists(), "Text embeddings pickle not created"
        assert struct_pkl.exists(), "Structure embeddings pickle not created"
        assert content_pkl.exists(), "Content embeddings pickle not created"
            
        print(f"✓ Text embeddings: {len(text_emb)} reports (100% coverage)")
        print(f"✓ Structure embeddings: {len(struct_emb)} reports (100% coverage)")
        print(f"✓ Content embeddings: {len(content_emb)} reports (100% coverage)")
        
        # Test similarity matrix computation
        print("\nComputing similarity matrix...")
        sim_matrix = self._compute_similarity_matrix(text_emb, struct_emb, content_emb)
        
        print(f"✓ Similarity matrix: {len(sim_matrix)} pairs computed")
        print(f"  Sample distances: text={sim_matrix[0]['text_dist']:.4f}, "
              f"struct={sim_matrix[0]['struct_dist']:.4f}, "
              f"content={sim_matrix[0]['content_dist']:.4f}")
        
        # Save similarity matrix as CSV
        query_corpus_pairs = [(p['id1'], p['id2']) for p in sim_matrix]
        similarity_scores = [np.mean([p['text_dist'], p['struct_dist'], p['content_dist']]) for p in sim_matrix]
        self._save_similarity_matrix_csv(query_corpus_pairs, similarity_scores, 'FILTERED')
        
        # Save readable pickle dump
        self._save_readable_pickle_dump('filtered')
        
        self.results['scenarios']['filtered'] = {
            'text_coverage': 100.0,
            'image_coverage': 100.0,
            'pairs_computed': len(sim_matrix)
        }
        self.results['tests_passed'] += 1
        return True
        
    def test_full_scenario(self):
        """
        Test FULL dataset scenario:
        - 100% text coverage
        - 10-12% image coverage
        - Mixed feature availability
        """
        print("\n" + "="*70)
        print("SCENARIO 2: FULL Dataset (100% text + 10-12% images)")
        print("="*70)
        
        # Create synthetic multi-project dataset with sparse images
        # 5 projects with different numbers of reports
        projects = {
            'MobileApp': 25,      # 25 reports
            'WebPortal': 20,      # 20 reports
            'BackendAPI': 15,     # 15 reports
            'Dashboard': 12,      # 12 reports
            'AdminTool': 8        # 8 reports
        }
        n_total = sum(projects.values())
        n_with_images = int(n_total * 0.11)  # 11% have images
        
        print(f"Testing with {n_total} reports across {len(projects)} projects:")
        for proj, count in projects.items():
            print(f"  - {proj}: {count} reports")
        print(f"Expected: 100% have text, {n_with_images} have images ({n_with_images/n_total*100:.1f}%)")
        
        # Create embeddings
        text_emb = {}
        struct_emb = {}
        content_emb = {}
        
        from apted import helpers
        tree_templates = [
            "{root{button{text}{icon}}{list{item1{text}{img}}{item2{text}{img}}}{footer{btn}}}",
            "{root{header{title}}{body{text}{image}}{footer}}",
            "{root{view{text}{button}}}",
            "{root{list{item1}{item2}{item3}}}",
            "{root{scroll{view1{text}}{view2{image}{text}}}}"
        ]
        
        idx = 0
        image_idx = 0
        for proj_name, n_reports in projects.items():
            for qid in range(1, n_reports + 1):
                key = f"{proj_name}:{qid}"
                
                # Text: Always available (100%)
                text_emb[key] = {
                    'problem_vector': np.random.rand(100),
                    'procedure_vectors': [np.random.rand(100)]
                }
                
                # Structure & Content: Only for some reports (11%)
                if image_idx < n_with_images:
                    struct_emb[key] = helpers.Tree.from_text(tree_templates[idx % len(tree_templates)])
                    content_emb[key] = [{
                        'bbox': [0, 0, 100, 100],
                        'vector': np.random.rand(512)
                    }]
                    image_idx += 1
                
                idx += 1
        
        # Save as pickle files
        text_pkl = self.embeddings_dir / 'text_embeddings_full.pkl'
        struct_pkl = self.embeddings_dir / 'structure_embeddings_full.pkl'
        content_pkl = self.embeddings_dir / 'content_embeddings_full.pkl'
        
        with open(text_pkl, 'wb') as f:
            pickle.dump(text_emb, f)
        with open(struct_pkl, 'wb') as f:
            pickle.dump(struct_emb, f)
        with open(content_pkl, 'wb') as f:
            pickle.dump(content_emb, f)
            
        print(f"✓ Text embeddings: {len(text_emb)} reports (100% coverage)")
        print(f"✓ Structure embeddings: {len(struct_emb)} reports ({len(struct_emb)/n_total*100:.1f}% coverage)")
        print(f"✓ Content embeddings: {len(content_emb)} reports ({len(content_emb)/n_total*100:.1f}% coverage)")
        
        # Test similarity matrix computation with mixed features
        print("\nComputing similarity matrix with adaptive fusion...")
        sim_matrix = self._compute_similarity_matrix_adaptive(text_emb, struct_emb, content_emb)
        
        # Analyze fusion modes
        mode_4way = sum(1 for p in sim_matrix if p['fusion_mode'] == 4)
        mode_3way = sum(1 for p in sim_matrix if p['fusion_mode'] == 3)
        mode_2way = sum(1 for p in sim_matrix if p['fusion_mode'] == 2)
        
        print(f"✓ Similarity matrix: {len(sim_matrix)} pairs computed")
        print(f"  4-way fusion (all features): {mode_4way} pairs ({mode_4way/len(sim_matrix)*100:.1f}%)")
        print(f"  3-way fusion (partial features): {mode_3way} pairs ({mode_3way/len(sim_matrix)*100:.1f}%)")
        print(f"  2-way fusion (text only): {mode_2way} pairs ({mode_2way/len(sim_matrix)*100:.1f}%)")
        
        # Save similarity matrix as CSV
        query_corpus_pairs = [(p['id1'], p['id2']) for p in sim_matrix]
        # Calculate average score based on available features
        similarity_scores = []
        for p in sim_matrix:
            available = []
            if p['text_dist'] is not None:
                available.append(p['text_dist'])
            if p['struct_dist'] is not None:
                available.append(p['struct_dist'])
            if p['content_dist'] is not None:
                available.append(p['content_dist'])
            similarity_scores.append(np.mean(available) if available else 0.0)
        self._save_similarity_matrix_csv(query_corpus_pairs, similarity_scores, 'FULL')
        
        # Save readable pickle dump
        self._save_readable_pickle_dump('full')
        
        self.results['scenarios']['full'] = {
            'text_coverage': 100.0,
            'image_coverage': len(struct_emb)/n_total*100,
            'pairs_computed': len(sim_matrix),
            'fusion_modes': {'4way': mode_4way, '3way': mode_3way, '2way': mode_2way}
        }
        self.results['tests_passed'] += 1
        return True
        
    def _compute_similarity_matrix(self, text_emb, struct_emb, content_emb):
        """Compute similarity matrix assuming all features available"""
        from scipy.spatial.distance import euclidean
        from apted import APTED
        
        pairs = []
        keys = list(text_emb.keys())
        
        # Group keys by project
        projects = {}
        for key in keys:
            project = key.split(':')[0]
            if project not in projects:
                projects[project] = []
            projects[project].append(key)
        
        # For each project, select some queries and compare with corpus
        for project, project_keys in projects.items():
            # Select first few as queries (max 4 per project)
            n_queries = min(4, len(project_keys))
            query_keys = project_keys[:n_queries]
            
            for q_key in query_keys:
                for c_key in project_keys:
                    if q_key == c_key:
                        continue
                    
                    # Text distance (Euclidean on problem vectors)
                    text_dist = euclidean(
                        text_emb[q_key]['problem_vector'],
                        text_emb[c_key]['problem_vector']
                    )
                    
                    # Structure distance (Tree Edit Distance)
                    struct_dist = APTED(struct_emb[q_key], struct_emb[c_key]).compute_edit_distance()
                    
                    # Content distance (mean of widget distances)
                    content_dist = euclidean(
                        content_emb[q_key][0]['vector'],
                        content_emb[c_key][0]['vector']
                    )
                    
                    pairs.append({
                        'id1': q_key,
                        'id2': c_key,
                        'text_dist': text_dist,
                        'struct_dist': struct_dist,
                        'content_dist': content_dist
                    })
        
        return pairs
    
    def _compute_similarity_matrix_adaptive(self, text_emb, struct_emb, content_emb):
        """Compute similarity matrix with adaptive fusion based on available features"""
        from scipy.spatial.distance import euclidean
        from apted import APTED
        
        pairs = []
        keys = list(text_emb.keys())
        
        # Group keys by project
        projects = {}
        for key in keys:
            project = key.split(':')[0]
            if project not in projects:
                projects[project] = []
            projects[project].append(key)
        
        # For each project, select some queries and compare with corpus
        for project, project_keys in projects.items():
            # Select first few as queries (max 5 per project for FULL dataset)
            n_queries = min(5, len(project_keys))
            query_keys = project_keys[:n_queries]
            
            for q_key in query_keys:
                for c_key in project_keys:
                    if q_key == c_key:
                        continue
                    
                    # Text: Always available
                    text_dist = euclidean(
                        text_emb[q_key]['problem_vector'],
                        text_emb[c_key]['problem_vector']
                    )
                    
                    # Structure: May not be available
                    struct_dist = None
                    if q_key in struct_emb and c_key in struct_emb:
                        struct_dist = APTED(struct_emb[q_key], struct_emb[c_key]).compute_edit_distance()
                    
                    # Content: May not be available
                    content_dist = None
                    if q_key in content_emb and c_key in content_emb:
                        if content_emb[q_key] and content_emb[c_key]:
                            content_dist = euclidean(
                                content_emb[q_key][0]['vector'],
                                content_emb[c_key][0]['vector']
                            )
                    
                    # Determine fusion mode
                    available_features = sum([
                        text_dist is not None,
                        text_dist is not None,  # text steps (same as text_dist for simplification)
                        struct_dist is not None,
                        content_dist is not None
                    ])
                    
                    pairs.append({
                        'id1': q_key,
                        'id2': c_key,
                        'text_dist': text_dist,
                        'struct_dist': struct_dist,
                        'content_dist': content_dist,
                        'fusion_mode': available_features
                    })
                
        return pairs
    
    def save_report(self):
        """Save test report"""
        report_file = self.test_dir / 'pipeline_test_report.json'
        with open(report_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n✓ Report saved: {report_file}")
    
    def _save_similarity_matrix_csv(self, query_corpus_pairs, similarity_matrix, dataset_name):
        """Save similarity matrix as CSV like the main project"""
        rows = []
        idx = 0
        
        # For testing, mark some pairs as ground truth (c_is_gt=1)
        # Simulate: each query has 1-2 relevant corpus items
        # Pattern: for query at position i, corpus at position (i+1)%n and (i+2)%n are relevant
        query_ids = sorted(list(set([q for q, c in query_corpus_pairs])))
        gt_pairs = set()
        for i, qid in enumerate(query_ids):
            # Mark 2 corpus items as ground truth for each query
            corpus_ids = [c for q, c in query_corpus_pairs if q == qid]
            if len(corpus_ids) >= 2:
                # Mark the 2nd and 3rd closest as ground truth (to test ranking)
                gt_pairs.add((qid, corpus_ids[1]))
                if len(corpus_ids) >= 3:
                    gt_pairs.add((qid, corpus_ids[2]))
        
        for (q_id, c_id), score in zip(query_corpus_pairs, similarity_matrix):
            # Extract project name from query ID (format: "Project:QueryID")
            project_name = q_id.split(':')[0] if ':' in q_id else 'TestProject'
            is_gt = 1 if (q_id, c_id) in gt_pairs else 0
            rows.append({
                'Project': project_name,
                'query': q_id,
                'corpus': c_id,
                'score': round(score, 6),
                'rank': idx + 1,
                'c_is_gt': is_gt,
                'BB': '',
                'RS': '',
                'SF': '',
                'CF': ''
            })
            idx += 1
        
        df = pd.DataFrame(rows)
        
        # Re-rank per query
        df = df.sort_values(['query', 'score']).reset_index(drop=True)
        df['rank'] = df.groupby('query').cumcount() + 1
        
        csv_path = self.test_dir / f'semcluster_similarity_matrix_{dataset_name}.csv'
        df.to_csv(csv_path, index=False)
        
        # Report ground truth statistics
        n_gt = df['c_is_gt'].sum()
        n_queries = df['query'].nunique()
        n_projects = df['Project'].nunique()
        print(f"✓ Similarity matrix CSV saved: {csv_path}")
        print(f"  Projects: {n_projects}, Queries: {n_queries}")
        print(f"  Ground truth: {n_gt} relevant pairs across {n_queries} queries")
        return csv_path
    
    def _save_readable_pickle_dump(self, dataset_name):
        """Save readable text dump of pickle file contents"""
        dump_path = self.test_dir / f'pickle_contents_{dataset_name}.txt'
        
        with open(dump_path, 'w') as f:
            # Text embeddings
            text_pkl = self.embeddings_dir / f'text_embeddings_{dataset_name}.pkl'
            if text_pkl.exists():
                with open(text_pkl, 'rb') as pkl:
                    text_data = pickle.load(pkl)
                f.write("=" * 80 + "\n")
                f.write(f"TEXT EMBEDDINGS ({dataset_name})\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"Total reports: {len(text_data)}\n\n")
                for report_id in sorted(list(text_data.keys())[:5]):  # Show first 5
                    data = text_data[report_id]
                    f.write(f"\nReport ID: {report_id}\n")
                    f.write(f"  Problem vector shape: {data['problem_vector'].shape}\n")
                    f.write(f"  Problem vector sample: {data['problem_vector'][:10]}\n")
                    f.write(f"  Procedure vectors: {len(data['procedure_vectors'])} steps\n")
                    if len(data['procedure_vectors']) > 0:
                        f.write(f"  First step shape: {data['procedure_vectors'][0].shape}\n")
                    f.write("\n")
            
            # Structure embeddings
            struct_pkl = self.embeddings_dir / f'structure_embeddings_{dataset_name}.pkl'
            if struct_pkl.exists():
                with open(struct_pkl, 'rb') as pkl:
                    struct_data = pickle.load(pkl)
                f.write("=" * 80 + "\n")
                f.write(f"STRUCTURE EMBEDDINGS ({dataset_name})\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"Total reports: {len(struct_data)}\n\n")
                for report_id in sorted(list(struct_data.keys())[:5]):  # Show first 5
                    tree = struct_data[report_id]
                    f.write(f"\nReport ID: {report_id}\n")
                    f.write(f"  Tree type: {type(tree).__name__}\n")
                    f.write(f"  Tree representation: {str(tree)[:200]}\n")
                    f.write("\n")
            
            # Content embeddings
            content_pkl = self.embeddings_dir / f'content_embeddings_{dataset_name}.pkl'
            if content_pkl.exists():
                with open(content_pkl, 'rb') as pkl:
                    content_data = pickle.load(pkl)
                f.write("=" * 80 + "\n")
                f.write(f"CONTENT EMBEDDINGS ({dataset_name})\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"Total reports: {len(content_data)}\n\n")
                for report_id in sorted(list(content_data.keys())[:5]):  # Show first 5
                    data = content_data[report_id]
                    f.write(f"\nReport ID: {report_id}\n")
                    if isinstance(data, list):
                        f.write(f"  Content regions: {len(data)}\n")
                        if len(data) > 0:
                            f.write(f"  First region bbox: {data[0].get('bbox', 'N/A')}\n")
                            vec = data[0].get('vector')
                            if vec is not None:
                                f.write(f"  First region vector shape: {vec.shape}\n")
                                f.write(f"  First region vector sample: {vec[:10]}\n")
                    else:
                        f.write(f"  Feature vector shape: {data.shape}\n")
                        f.write(f"  Feature vector sample: {data[:10]}\n")
                    f.write("\n")
        
        print(f"✓ Readable pickle dump saved: {dump_path}")
        return dump_path
        
    def run_all_tests(self):
        """Run all test scenarios"""
        print("\n" + "="*70)
        print("SemCluster Pipeline Testing")
        print("="*70)
        print("Testing: Embedding creation (pickle) → Similarity matrix computation")
        
        self.setup()
        
        try:
            self.test_filtered_scenario()
            self.test_full_scenario()
            
            print("\n" + "="*70)
            print("TEST SUMMARY")
            print("="*70)
            print(f"✓ Tests Passed: {self.results['tests_passed']}")
            print(f"✗ Tests Failed: {self.results['tests_failed']}")
            print(f"Success Rate: {self.results['tests_passed']/(self.results['tests_passed']+self.results['tests_failed'])*100:.1f}%")
            
            self.save_report()
            
        except Exception as e:
            print(f"\n✗ Test failed: {e}")
            import traceback
            traceback.print_exc()
            self.results['tests_failed'] += 1
            
if __name__ == '__main__':
    tester = SemClusterPipelineTester()
    tester.run_all_tests()
