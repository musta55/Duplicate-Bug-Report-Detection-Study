import jieba
import gensim
import numpy as np
import pandas as pd
import pickle
import os

from text.text_feature_extraction import dtw as dtw
import text.text_feature_extraction.text_feature_extraction as tfe

# pre-trained word2ve model
word2vec_model = gensim.models.Word2Vec.load('text/text_feature_extraction/bugdata_format_model_100')
# initial widget categories vector of blank report
initial_widget_categories = np.zeros(15)
# initial reproduction steps list of blank report
initial_reproduction_procedures = ['']
# initial bug descriptions list of blank report
initial_bug_descriptions = ['']


# sim between two short sentences (measured by euclidean distance)
def sentence_sim(model, s1, s2):
    size = model.layer1_size

    def sentence_vector(s):
        words = []
        try:
            words = [x for x in jieba.cut(s, cut_all=True) if x != '']
        except:
            return np.zeros(size)
        v = np.zeros(size)
        length = len(words)
        for word in words:
            try:
                v += model.wv[word]
            except:
                length -= 1
        if length == 0:
            return np.zeros(size)
        v /= length
        return v

    v1, v2 = sentence_vector(s1), sentence_vector(s2)
    return eucli_distance(v1, v2)


def eucli_distance(a, b):
    dis = np.sqrt(sum(np.power((a - b), 2)))
    return round(dis, 4)


# normalize distance to [0,1] by max_dis and min_dis
def normalize_dis(dis, max_dis, min_dis):
    if max_dis <= min_dis:
        return 0.0
    return (dis - min_dis) / (max_dis - min_dis)


# SR = gamma * SB + delta * SC
def cal_report_similarity(text_feature1, text_feature2, index1, index2, max_problem_dis, min_problem_dis,
                          max_procedure_dis, min_procedure_dis):
    print('---------begin cal {}th & {}th report sim---------'.format(index1, index2))
    print('---{}th text_feature{}---'.format(index1, text_feature1))
    print('---{}th text_feature{}---'.format(index2, text_feature2))
    sp_dis = cal_bug_similarity(text_feature1, text_feature2, max_problem_dis, min_problem_dis)
    # if SB == -1 or SB == -2:
    #     return SB
    sr_dis = cal_context_similarity(text_feature1, text_feature2, max_procedure_dis, min_procedure_dis)
    # SR = gamma * SB + delta * SC
    # whole_dis = 0.25 * sp_dis + 0.25 * swp_dis + 0.25 * sr_dis + 0.25 * swc_dis
    print('---------finish cal {}th & {}th report sim---------'.format(index1, index2))
    print(sp_dis)
    print(sr_dis)
    return round(sp_dis, 4), round(sr_dis, 4)


# SB = alpha1 * SP + beta1 * SWP
def cal_bug_similarity(text_feature1, text_feature2, max_problem_dis, min_problem_dis):
    problem_list1 = text_feature1['problems_list']
    problem1 = ' '.join(problem_list1)
    problem2 = ''
    if text_feature2 is not None:
        problem_list2 = text_feature2['problems_list']
        problem2 = ' '.join(problem_list2)
    # dis = sift_similarity(pwidget_img1, pwidget_img2)
    # SWP = normalize_dis(dis, max_sift_fea, 0)
    # swp_dis = 1 - SWP
    # if SWP == -1 or SWP == -2:
    #     return SWP
    dis = sentence_sim(word2vec_model, problem1, problem2)
    sp_dis = sentence_sim(word2vec_model, problem1, problem2)
    sp_dis = normalize_dis(sp_dis, max_problem_dis, min_problem_dis)
    # SB = alpha1 * SP + beta1 * SWP
    # print('SP:{}'.format(SP))
    return sp_dis


# SC = alpha2 * SR + beta2 * SWC
def cal_context_similarity(text_feature1, text_feature2, max_procedure_dis, min_procedure_dis):
    procedures_list1 = text_feature1['procedures_list']
    procedures_list2 = initial_reproduction_procedures
    if text_feature2 is not None:
        procedures_list2 = text_feature2['procedures_list']
    # sim = 1 - dis
    sr_dis = dtw.dtw_distance(procedures_list1, procedures_list2, min_procedure_dis, max_procedure_dis)
    # SR = 1.0 - sr_dis
    # cal dis between two widget category vec
    # print('$$$$$$$$$$$$$$$$$$$$$')
    # print(category_vec1)
    # print(category_vec2)
    # print('$$$$$$$$$$$$$$$$$$$$$')
    # dis = normalize_dis(eucli_distance(category_vec1, category_vec2), max_category_dis, min_category_dis)
    # swc_dis = round(dis, 4)
    # SWC = 1.0 - swc_dis
    # SC = alpha2 * SR + beta2 * SWC
    print(procedures_list1, procedures_list2)
    sr_dis = normalize_dis(sr_dis, max_procedure_dis, min_procedure_dis)
    # print('SR:{}'.format(SR))
    return sr_dis


# get max and min value of four types of different distance - OPTIMIZED VERSION
def cal_normalize_dis(number, text_feature_list, sample_df=None, query_to_valid_corpus=None, seq_to_id=None, parquet_df=None):
    # the max num of sift feature points of screenshots (used to normalize dis)
    max_sift_fea = 0.
    # the max val of Euclidean distance of two bug descriptions (used to normalize dis)
    max_problem_dis = 0.
    # the min val of Euclidean distance of two bug descriptions (used to normalize dis)
    min_problem_dis = 100000.
    # the max val of Euclidean distance of two reproduction steps (used to normalize dis)
    max_procedure_dis = 0.
    # the min val of Euclidean distance of two reproduction steps (used to normalize dis)
    min_procedure_dis = 100000.
    # the max val of Euclidean distance of two widget categories vectors (used to normalize dis)
    max_category_dis = 0.
    # the min val of Euclidean distance of two widget categories vectors (used to normalize dis)
    min_category_dis = 100000.
    
    # Check if optimization parameters are provided
    use_optimization = (sample_df is not None and query_to_valid_corpus is not None and 
                       seq_to_id is not None and parquet_df is not None and len(query_to_valid_corpus) > 0)
    
    if use_optimization:
        print(f"  → TEXT NORMALIZATION: OPTIMIZED MODE - Computing only needed pairs")
        import sys
        sys.stdout.flush()
        
        # Build ID to repo mapping
        id_to_repo = {}
        for idx, row in parquet_df.iterrows():
            id_to_repo[row['id']] = row['repo_name']
        
        # Build seq_idx to list index mapping
        seq_idx_to_list_idx = {number[i]: i for i in range(len(number))}
        
        # Build (repo, report_id) to seq_idx mapping
        repo_id_to_seq_idx = {}
        if seq_to_id:
             for seq_idx, composite_id in seq_to_id.items():
                 if ':' in str(composite_id):
                     repo_name, real_id_str = str(composite_id).split(':', 1)
                     real_id = int(real_id_str)
                 else:
                     real_id = int(composite_id)
                     repo_name = id_to_repo.get(real_id, 'Unknown')
                 repo_id_to_seq_idx[(repo_name, real_id)] = seq_idx
        
        # Collect all needed pairs
        needed_pairs = set()
        for (repo_name, query_id), corpus_tuples in query_to_valid_corpus.items():
            if (repo_name, query_id) not in repo_id_to_seq_idx:
                continue
            query_seq_idx = repo_id_to_seq_idx[(repo_name, query_id)]
            if query_seq_idx not in seq_idx_to_list_idx:
                continue
            query_list_idx = seq_idx_to_list_idx[query_seq_idx]
            
            for corpus_repo, corpus_id in corpus_tuples:
                if (corpus_repo, corpus_id) in repo_id_to_seq_idx:
                    corpus_seq_idx = repo_id_to_seq_idx[(corpus_repo, corpus_id)]
                    if corpus_seq_idx in seq_idx_to_list_idx:
                        corpus_list_idx = seq_idx_to_list_idx[corpus_seq_idx]
                        needed_pairs.add((min(query_list_idx, corpus_list_idx), max(query_list_idx, corpus_list_idx)))
        
        print(f"  → TEXT NORMALIZATION: Computing {len(needed_pairs)} pairs (not {len(number)*(len(number)-1)//2})")
        sys.stdout.flush()
        
        # First pass: blank report comparisons (always needed for normalization)
        for i in range(0, len(number)):
            text_feature = text_feature_list[i]
            problem_list = text_feature['problems_list']
            problem = ' '.join(problem_list)
            problem_sim = sentence_sim(word2vec_model, problem, '')
            if problem_sim > max_problem_dis:
                max_problem_dis = problem_sim
            if problem_sim < min_problem_dis:
                min_problem_dis = problem_sim
            procedures_list = text_feature['procedures_list']
            min_dtw_dis, max_dtw_dis = dtw.dtw_distance(procedures_list, initial_reproduction_procedures, 0, 0)
            if min_dtw_dis < min_procedure_dis:
                min_procedure_dis = min_dtw_dis
            if max_dtw_dis > max_procedure_dis:
                max_procedure_dis = max_dtw_dis
        
        # Second pass: only compute needed query-corpus pairs
        pair_count = 0
        for (i, j) in needed_pairs:
            if pair_count % 100 == 0:
                print(f"  → Progress: {pair_count}/{len(needed_pairs)} pairs computed")
                sys.stdout.flush()
            text_feature1 = text_feature_list[i]
            problem_list1 = text_feature1['problems_list']
            problem1 = ' '.join(problem_list1)
            text_feature2 = text_feature_list[j]
            problem_list2 = text_feature2['problems_list']
            problem2 = ' '.join(problem_list2)
            problem_sim = sentence_sim(word2vec_model, problem1, problem2)
            if problem_sim > max_problem_dis:
                max_problem_dis = problem_sim
            if problem_sim < min_problem_dis:
                min_problem_dis = problem_sim
            procedures_list1 = text_feature1['procedures_list']
            procedures_list2 = text_feature2['procedures_list']
            min_dtw_dis, max_dtw_dis = dtw.dtw_distance(procedures_list1, procedures_list2, 0, 0)
            if min_dtw_dis < min_procedure_dis:
                min_procedure_dis = min_dtw_dis
            if max_dtw_dis > max_procedure_dis:
                max_procedure_dis = max_dtw_dis
            pair_count += 1
    else:
        print("  → TEXT NORMALIZATION: LEGACY MODE - Computing all N×N pairs")
        # Original unoptimized code
        for i in range(0, len(number)):
            # cal dis between each report and blank report
            print('-------------start cal sim between {}th & -1 report-----------'.format(number[i]))
            text_feature = text_feature_list[i]
            problem_list = text_feature['problems_list']
            print('-----------{}th report problem_list:{}'.format(number[i], problem_list))
            problem = ' '.join(problem_list)
            problem_sim = sentence_sim(word2vec_model, problem, '')
            if problem_sim > max_problem_dis:
                max_problem_dis = problem_sim
            if problem_sim < min_problem_dis:
                min_problem_dis = problem_sim
            procedures_list = text_feature['procedures_list']
            print('-----------{}th report procedure_list:{}'.format(number[i], procedures_list))
            min_dtw_dis, max_dtw_dis = dtw.dtw_distance(procedures_list, initial_reproduction_procedures, 0, 0)
            if min_dtw_dis < min_procedure_dis:
                min_procedure_dis = min_dtw_dis
            if max_dtw_dis > max_procedure_dis:
                max_procedure_dis = max_dtw_dis
            print('-------------end cal sim between {}th & -1 report-----------'.format(number[i]))
        # cal dis between every two reports in report list
        for i in range(0, len(number) - 1):
            for j in range(i + 1, len(number)):
                print('-------------start cal sim between {}th & {}th report-----------'.format(number[i], number[j]))
                text_feature1 = text_feature_list[i]
                problem_list1 = text_feature1['problems_list']
                problem1 = ' '.join(problem_list1)
                text_feature2 = text_feature_list[j]
                problem_list2 = text_feature2['problems_list']
                print('-----------{}th report problem_list:{}'.format(number[i], problem_list1))
                print('-----------{}th report problem_list:{}'.format(number[j], problem_list2))
                problem2 = ' '.join(problem_list2)
                problem_sim = sentence_sim(word2vec_model, problem1, problem2)
                if problem_sim > max_problem_dis:
                    max_problem_dis = problem_sim
                if problem_sim < min_problem_dis:
                    min_problem_dis = problem_sim
                procedures_list1 = text_feature1['procedures_list']
                procedures_list2 = text_feature2['procedures_list']
                print('-----------{}th report procedure_list:{}'.format(number[i], procedures_list1))
                print('-----------{}th report procedure_list:{}'.format(number[i], procedures_list2))
                min_dtw_dis, max_dtw_dis = dtw.dtw_distance(procedures_list1, procedures_list2, 0, 0)
                if min_dtw_dis < min_procedure_dis:
                    min_procedure_dis = min_dtw_dis
                if max_dtw_dis > max_procedure_dis:
                    max_procedure_dis = max_dtw_dis
                print('-------------end cal sim between {}th & {}th report-----------'.format(number[i], number[j]))
    print('max_problem={},min_problem={},max_procedure={},min_procedure={}'.format(max_problem_dis, min_problem_dis,
                                                                                   max_procedure_dis,
                                                                                   min_procedure_dis))
    return max_problem_dis, min_problem_dis, max_procedure_dis, min_procedure_dis


# calculate and record sim between every two reports in report list
def cal_sim_matrix(file_path, text_feature_list, sample_df=None, query_to_valid_corpus=None, seq_to_id=None, parquet_df=None):
    """
    Calculate similarity matrix - OPTIMIZED to only compute query-corpus pairs.
    
    Args:
        file_path: Path to evaluation CSV
        text_feature_list: List of extracted text features
        sample_df: DataFrame with query information (for optimization)
        query_to_valid_corpus: Dict mapping (repo, query_id) -> set of (repo, corpus_id) tuples
        seq_to_id: Mapping from sequential index to real report ID
        parquet_df: DataFrame with repo_name for ID lookup
    """
    # CACHING IMPLEMENTATION
    # Derive cache filename from input CSV path to separate FILTERED vs FULL caches
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    cache_file = f"text_features_cache_{base_name}.pkl"
    
    if os.path.exists(cache_file):
        print(f"  → [CACHE] Found {cache_file}, loading cached text features...")
        try:
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            # Verify cache validity (simple check: length of features)
            # Note: This is a basic check. Ideally we'd hash the inputs.
            if len(cached_data) == 4:
                p_list, r_list, p_pairs, r_pairs = cached_data
                # Check if dimensions match current input
                # text_feature_list length corresponds to number of reports
                # p_list has header + rows, so len should be len(text_feature_list) + 1
                if len(p_list) == len(text_feature_list) + 1:
                    print(f"  → [CACHE] Cache loaded successfully!")
                    return p_list, r_list, p_pairs, r_pairs
                else:
                    print(f"  → [CACHE] Cache size mismatch (expected {len(text_feature_list)+1}, got {len(p_list)}). Recomputing...")
        except Exception as e:
            print(f"  → [CACHE] Failed to load cache: {e}. Recomputing...")

    reports = pd.read_csv(file_path, engine='python')
    number = list(reports['index'])
    # get max and min value of distance (used for distance normalization)
    max_problem_dis, min_problem_dis, max_procedure_dis, min_procedure_dis = \
        cal_normalize_dis(number, text_feature_list, sample_df, query_to_valid_corpus, seq_to_id, parquet_df)

    header = ['index']
    for n in number:
        header.append(str(n))

    all_dis_p_list = []
    all_dis_r_list = []
    all_dis_p_list.append(header)
    all_dis_r_list.append(header)
    sp_dis_matrix = np.zeros((len(number) + 1, len(number) + 1))
    sr_dis_matrix = np.zeros((len(number) + 1, len(number) + 1))
    sp_dis_matrix[0][0] = 0
    sr_dis_matrix[0][0] = 0
    pair_p_distances = {}
    pair_r_distances = {}
    
    # OPTIMIZED: Only compute query-corpus pairs if mapping provided
    print(f"  → DEBUG cal_sim_matrix: sample_df={sample_df is not None}, query_to_valid_corpus={query_to_valid_corpus is not None}, seq_to_id={seq_to_id is not None}, parquet_df={parquet_df is not None}")
    if query_to_valid_corpus is not None and len(query_to_valid_corpus) > 0:
        print(f"  → DEBUG cal_sim_matrix: query_to_valid_corpus has {len(query_to_valid_corpus)} entries")
    if seq_to_id is not None:
        print(f"  → DEBUG cal_sim_matrix: seq_to_id has {len(seq_to_id)} entries")
    
    if sample_df is not None and query_to_valid_corpus is not None and seq_to_id is not None and parquet_df is not None and len(query_to_valid_corpus) > 0:
        print(f"  → OPTIMIZED MODE: Computing only query-corpus text similarities")
        import sys
        sys.stdout.flush()
        
        # Build repo lookup from parquet_df
        id_to_repo = {}
        for idx, row in parquet_df.iterrows():
            id_to_repo[row['id']] = row['repo_name']
        
        # Build seq_idx to list index mapping
        seq_idx_to_list_idx = {number[i]: i for i in range(len(number))}
        
        # Build (repo, report_id) to seq_idx mapping
        repo_id_to_seq_idx = {}
        seq_idx_to_repo = {}
        
        if seq_to_id:
             for seq_idx, composite_id in seq_to_id.items():
                 if ':' in str(composite_id):
                     repo_name, real_id_str = str(composite_id).split(':', 1)
                     real_id = int(real_id_str)
                 else:
                     real_id = int(composite_id)
                     repo_name = id_to_repo.get(real_id, 'Unknown')
                 repo_id_to_seq_idx[(repo_name, real_id)] = seq_idx
                 seq_idx_to_repo[seq_idx] = (repo_name, real_id)

        def record_pair(i_idx, j_idx, sp_val, sr_val):
            # i_idx and j_idx are indices in the number list (0..N-1)
            # number[i_idx] gives the seq_idx
            seq_idx_i = number[i_idx]
            seq_idx_j = number[j_idx]
            
            repo_info_i = seq_idx_to_repo.get(seq_idx_i)
            repo_info_j = seq_idx_to_repo.get(seq_idx_j)
            
            if not repo_info_i or not repo_info_j:
                return
            repo_i, id_i = repo_info_i
            repo_j, id_j = repo_info_j
            if repo_i == 'Unknown' or repo_j == 'Unknown':
                return
            pair_p_distances[(repo_i, id_i, id_j)] = round(sp_val, 4)
            pair_r_distances[(repo_i, id_i, id_j)] = round(sr_val, 4)
        
        # Collect all needed pairs using composite (repo, id) keys
        needed_pairs = set()
        for (repo_name, query_id), corpus_tuples in query_to_valid_corpus.items():
            if (repo_name, query_id) not in repo_id_to_seq_idx:
                continue
            query_seq_idx = repo_id_to_seq_idx[(repo_name, query_id)]
            if query_seq_idx not in seq_idx_to_list_idx:
                continue
            query_list_idx = seq_idx_to_list_idx[query_seq_idx]
            
            for corpus_repo, corpus_id in corpus_tuples:
                if (corpus_repo, corpus_id) in repo_id_to_seq_idx:
                    corpus_seq_idx = repo_id_to_seq_idx[(corpus_repo, corpus_id)]
                    if corpus_seq_idx in seq_idx_to_list_idx:
                        corpus_list_idx = seq_idx_to_list_idx[corpus_seq_idx]
                        needed_pairs.add((query_list_idx, corpus_list_idx))
                        needed_pairs.add((corpus_list_idx, query_list_idx))
        
        # Add self-similarities (diagonal)
        for i in range(len(number)):
            needed_pairs.add((i, i))
        
        print(f"  → Computing {len(needed_pairs)} text comparisons (instead of {len(number)*len(number)})")
        sys.stdout.flush()
        
        # Initialize all to maximum distance (completely dissimilar)
        sp_dis_matrix.fill(1.0)
        sr_dis_matrix.fill(1.0)
        sp_dis_matrix[0][0] = 0
        sr_dis_matrix[0][0] = 0
        
        # Compute only needed pairs
        processed = 0
        for (i, j) in needed_pairs:
            if i == j:
                sp_dis_matrix[i + 1][j + 1] = 0
                sr_dis_matrix[i + 1][j + 1] = 0
            else:
                sp_dis, sr_dis = cal_report_similarity(text_feature_list[i], text_feature_list[j], number[i],
                                                       number[j], max_problem_dis, min_problem_dis,
                                                       max_procedure_dis, min_procedure_dis)
                sp_dis_matrix[i + 1][j + 1] = sp_dis
                sr_dis_matrix[i + 1][j + 1] = sr_dis
                record_pair(i, j, sp_dis, sr_dis)
            
            processed += 1
            if processed % 1000 == 0:
                print(f"  → Text: Processed {processed}/{len(needed_pairs)} pairs ({processed/len(needed_pairs)*100:.1f}%)")
                sys.stdout.flush()
        
        # Set first row/column to max distance for consistency
        for i in range(len(number)):
            sp_dis, sr_dis = cal_report_similarity(text_feature_list[i], None, number[i], -1, max_problem_dis,
                                                   min_problem_dis, max_procedure_dis, min_procedure_dis)
            sp_dis_matrix[i + 1][0] = sp_dis
            sp_dis_matrix[0][i + 1] = sp_dis
            sr_dis_matrix[i + 1][0] = sr_dis
            sr_dis_matrix[0][i + 1] = sr_dis
        
        print(f"  → Text features complete (optimized)!")
        sys.stdout.flush()
    else:
        # LEGACY MODE: Compute full N×N matrix
        print(f"  → LEGACY MODE: Computing full N×N text matrix ({len(number)*len(number)} comparisons)")
        sys.stdout.flush()
        
        for i in range(0, len(number)):
            sp_dis, sr_dis = cal_report_similarity(text_feature_list[i], None, number[i], -1, max_problem_dis,
                                                   min_problem_dis, max_procedure_dis, min_procedure_dis)
            sp_dis_matrix[i + 1][0] = sp_dis
            sp_dis_matrix[0][i + 1] = sp_dis
            sr_dis_matrix[i + 1][0] = sr_dis
            sr_dis_matrix[0][i + 1] = sr_dis
        for i in range(0, len(number)):
            for j in range(i, len(number)):
                if i == j:
                    sp_dis_matrix[i + 1][j + 1] = 0
                    sr_dis_matrix[i + 1][j + 1] = 0
                else:
                    sr_dis, sp_dis = cal_report_similarity(text_feature_list[i], text_feature_list[j], number[i],
                                                           number[j], max_problem_dis, min_problem_dis,
                                                           max_procedure_dis, min_procedure_dis)
                    sp_dis_matrix[i + 1][j + 1] = sp_dis
                    sp_dis_matrix[j + 1][i + 1] = sp_dis
                    sr_dis_matrix[i + 1][j + 1] = sr_dis
                    sr_dis_matrix[j + 1][i + 1] = sr_dis
                    repo_i = id_to_repo.get(number[i], 'Unknown')
                    repo_j = id_to_repo.get(number[j], 'Unknown')
                    if repo_i != 'Unknown' and repo_j != 'Unknown':
                        pair_p_distances[(repo_i, number[i], number[j])] = round(sp_dis, 4)
                        pair_p_distances[(repo_j, number[j], number[i])] = round(sp_dis, 4)
                        pair_r_distances[(repo_i, number[i], number[j])] = round(sr_dis, 4)
                        pair_r_distances[(repo_j, number[j], number[i])] = round(sr_dis, 4)
    for i in range(0, len(number) + 1):
        index = 0
        if i == 0:
            index = -1
        else:
            index = number[i - 1]
        sp_dis = [index]
        sr_dis = [index]
        for j in range(0, len(number) + 1):
            sp_dis.append(sp_dis_matrix[i][j])
            sr_dis.append(sr_dis_matrix[i][j])
        all_dis_p_list.append(sp_dis)
        all_dis_r_list.append(sr_dis)
    
    # Save to cache
    print(f"  → [CACHE] Saving text features to {cache_file}...")
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump((all_dis_p_list, all_dis_r_list, pair_p_distances, pair_r_distances), f)
        print(f"  → [CACHE] Saved successfully!")
    except Exception as e:
        print(f"  → [CACHE] Failed to save cache: {e}")

    return all_dis_p_list, all_dis_r_list, pair_p_distances, pair_r_distances


# the main process of extract report feature
def extract_report_feature(file_path):
    return tfe.text_feature_extraction(pd.read_csv(file_path, engine='python')['description'])
