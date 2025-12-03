import os
import cv2
import pandas as pd
import xml.dom.minidom as xmldom

import image.vgg16 as vgg16
# import image.vgg16_stub as vgg16

threshold = 0.7

# Global caches for performance optimization
_xml_cache = {}
_image_cache = {}

def get_parsed_xml(xmlpath):
    """Cache and parse XML files to avoid repeated disk I/O"""
    if xmlpath in _xml_cache:
        return _xml_cache[xmlpath]
    
    try:
        dom_obj = xmldom.parse(xmlpath)
        element_obj = dom_obj.documentElement
        sub_elements = element_obj.getElementsByTagName("col")
        
        parsed_list = []
        for i in range(len(sub_elements)):
            # Convert to int immediately to save processing time later
            # Handle potential float strings or empty strings if necessary
            try:
                x = int(float(sub_elements[i].getAttribute("x")))
                y = int(float(sub_elements[i].getAttribute("y")))
                w = int(float(sub_elements[i].getAttribute("w")))
                h = int(float(sub_elements[i].getAttribute("h")))
                parsed_list.append([x, y, w, h])
            except ValueError:
                continue
                
        _xml_cache[xmlpath] = parsed_list
        return parsed_list
    except Exception as e:
        print(f"Error parsing XML {xmlpath}: {e}")
        return []

def get_loaded_image(imgpath):
    """Cache loaded images to avoid repeated disk I/O and decoding"""
    if imgpath in _image_cache:
        return _image_cache[imgpath]
    
    img = cv2.imread(imgpath)
    if img is not None:
        _image_cache[imgpath] = img
    return img


def cal_iou_ok(list1, list2, bias=0):
    # list1/2 are [x, y, w, h]
    # x is col, y is row
    col_min_a, row_min_a = list1[0], list1[1]
    col_max_a, row_max_a = list1[0] + list1[2], list1[1] + list1[3]
    
    col_min_b, row_min_b = list2[0], list2[1]
    col_max_b, row_max_b = list2[0] + list2[2], list2[1] + list2[3]
    
    if col_min_a > col_max_b or col_min_b > col_max_a or row_min_a > row_max_b or row_min_b > row_max_a:
        return False
    col_min_s = max(col_min_a - bias, col_min_b - bias)
    row_min_s = max(row_min_a - bias, row_min_b - bias)
    col_max_s = min(col_max_a + bias, col_max_b + bias)
    row_max_s = min(row_max_a + bias, row_max_b + bias)
    w = max(0, col_max_s - col_min_s)
    h = max(0, row_max_s - row_min_s)
    inter = w * h
    area_a = (col_max_a - col_min_a) * (row_max_a - row_min_a)
    area_b = (col_max_b - col_min_b) * (row_max_b - row_min_b)
    if area_a + area_b - inter == 0:
        return False
    iou = inter / (area_a + area_b - inter)
    return iou >= threshold


def is_pure_color(com):
    if com.size == 0: return True
    baseline = com[0, 0]
    base_b = baseline[0]
    base_g = baseline[1]
    base_r = baseline[2]
    
    # Optimization: Use numpy for faster check
    # Check if all pixels are equal to the first pixel
    return (com == baseline).all()


def process(xmlpath1, xmlpath2, imgpath1, imgpath2):
    # Use cached XML data
    list_file1_all = get_parsed_xml(xmlpath1)
    list_file2_all = get_parsed_xml(xmlpath2)

    isChange = False
    
    # Reference lists (do not modify in place)
    list1 = list_file1_all
    list2 = list_file2_all

    if len(list1) < len(list2):
        list1, list2 = list2, list1
        isChange = True
        
    # list1 is now the longer one (or equal)
    # We want to match elements in list2 (shorter) to list1 (longer)
    
    count = 0
    flags = [False] * len(list2)
    match_pool = []

    for i in range(len(list2)):
        for j in range(len(list1)):
            if cal_iou_ok(list1[j], list2[i]) and flags[i] == False:
                count += 1
                flags[i] = True
                # Store matched pair
                match_pool.append((list1[j], list2[i]))
                break

    list_match_com = []
    
    # Use cached images
    if isChange:
        img1 = get_loaded_image(imgpath2)
        img2 = get_loaded_image(imgpath1)
    else:
        img1 = get_loaded_image(imgpath1)
        img2 = get_loaded_image(imgpath2)
        
    if img1 is None or img2 is None:
        return 0.0

    reduce_count = count

    for i in range(count):
        list_pairs = match_pool[i]
        list_pair1 = list_pairs[0] # [x, y, w, h]
        list_pair2 = list_pairs[1] # [x, y, w, h]
        
        # Already ints from cache
        
        if list_pair1[2] == 0 or list_pair1[3] == 0 or list_pair2[2] == 0 or list_pair2[3] == 0:
            continue
            
        # Crop images: img[y:y+h, x:x+w]
        # list_pair is [x, y, w, h]
        com1 = img1[list_pair1[1]:list_pair1[1] + list_pair1[3],
               list_pair1[0]:list_pair1[0] + list_pair1[2]]
               
        if com1.size == 0:
            reduce_count -= 1
            continue

        if is_pure_color(com1):
            reduce_count -= 1
            continue
            
        com1 = cv2.resize(com1, (224, 224))
        
        com2 = img2[list_pair2[1]:list_pair2[1] + list_pair2[3],
               list_pair2[0]:list_pair2[0] + list_pair2[2]]
               
        if com2.size == 0:
            reduce_count -= 1
            continue

        com2 = cv2.resize(com2, (224, 224))
        
        list_match_com.append([com1, com2])

    # Pass normalize=True for cosine-like similarity
    distance_list = vgg16.getdistance(list_match_com, normalize=True)

    result = 0
    for i in range(len(distance_list)):
        result += distance_list[i]
    if reduce_count == 0:
        result = 1
    if reduce_count != 0:
        result /= reduce_count
    if imgpath1 == imgpath2:
        result = 0.0
    return result


def getCTdis(xml_dir, img_dir, label_csv, sample_df=None, query_to_valid_corpus=None, seq_to_id=None, parquet_df=None):
    """
    Get content distance matrix - OPTIMIZED to only compute query-corpus pairs.
    """
    import sys
    import pickle
    print("  → [DEBUG] Entered getCTdis")
    sys.stdout.flush()
    
    data = pd.read_csv(label_csv, header=None, engine='python').drop([0])
    title = ["index"]
    xml_list = sorted([f for f in os.listdir(xml_dir) if f.endswith('.xml')])
    img_list = sorted([f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])

    for i in range(len(xml_list)):
        title.append(str(data.iloc[i][0]))

    # Create mapping from (repo_name, report_id) to index - CRITICAL for avoiding ID collisions!
    # With 8,124 duplicate IDs across 29 repos, we MUST use composite keys
    repo_id_to_idx = {}
    id_to_idx = {}  # Keep simple mapping for backwards compatibility in legacy mode
    idx_to_repo_id = {}
    
    # Build repo_name lookup from parquet_df if available
    id_to_repo = {}
    if parquet_df is not None:
        for idx, row in parquet_df.iterrows():
            id_to_repo[row['id']] = row['repo_name']
    
    # ROBUST ID EXTRACTION: Get ID directly from filename instead of relying on label_csv order
    # This fixes the bug where label_csv might not match os.listdir order or content
    for i in range(len(xml_list)):
        filename = xml_list[i]
        try:
            # Extract ID from filename: layout123.xml -> 123
            # Assuming filename format is "layout{id}.xml"
            seq_idx = int(filename.replace('layout', '').replace('.xml', ''))
        except ValueError:
            # Fallback to label_csv if filename parsing fails
            if i < len(data):
                seq_idx = int(data.iloc[i][0])
            else:
                continue

        # Resolve Real ID and Repo using seq_to_id if available
        real_id = seq_idx
        repo_name = 'Unknown'
        
        if seq_to_id is not None and seq_idx in seq_to_id:
            composite_id = seq_to_id[seq_idx]
            if ':' in str(composite_id):
                repo_name, real_id_str = str(composite_id).split(':', 1)
                real_id = int(real_id_str)
            else:
                real_id = int(composite_id)
                repo_name = id_to_repo.get(real_id, 'Unknown')
        else:
            # Legacy fallback: assume seq_idx IS the real ID (or look it up directly)
            real_id = seq_idx
            repo_name = id_to_repo.get(real_id, 'Unknown')

        id_to_idx[real_id] = i  # Simple mapping for legacy mode
        
        if repo_name != 'Unknown':
            repo_id_to_idx[(repo_name, real_id)] = i
            idx_to_repo_id[i] = (repo_name, real_id)
        else:
            idx_to_repo_id[i] = ('Unknown', real_id)
    
    # OPTIMIZED: Only compute query-corpus pairs if mapping provided
    print(f"  → DEBUG getCTdis: sample_df={sample_df is not None}, query_to_valid_corpus={query_to_valid_corpus is not None}, seq_to_id={seq_to_id is not None}, parquet_df={parquet_df is not None}")
    if query_to_valid_corpus is not None:
        print(f"  → DEBUG getCTdis: query_to_valid_corpus has {len(query_to_valid_corpus)} entries")
    if seq_to_id is not None:
        print(f"  → DEBUG getCTdis: seq_to_id has {len(seq_to_id)} entries")
    print(f"  → DEBUG getCTdis: repo_id_to_idx has {len(repo_id_to_idx)} entries")
    print(f"  → DEBUG getCTdis: id_to_repo has {len(id_to_repo)} entries")
    sys.stdout.flush()
    if sample_df is not None and query_to_valid_corpus is not None and seq_to_id is not None and parquet_df is not None and len(query_to_valid_corpus) > 0:
        print(f"  → OPTIMIZED MODE: Computing only query-corpus pairs (using repo-aware ID matching)")
        sys.stdout.flush()
        
        # Collect all needed pairs using composite (repo, id) keys
        needed_pairs = set()
        for (repo_name, query_id), corpus_tuples in query_to_valid_corpus.items():
            if (repo_name, query_id) not in repo_id_to_idx:
                continue
            query_idx = repo_id_to_idx[(repo_name, query_id)]
            
            for corpus_repo, corpus_id in corpus_tuples:
                if (corpus_repo, corpus_id) in repo_id_to_idx:
                    corpus_idx = repo_id_to_idx[(corpus_repo, corpus_id)]
                    needed_pairs.add((query_idx, corpus_idx))
                    needed_pairs.add((corpus_idx, query_idx))
        
        # Add self-similarities (diagonal)
        for idx in range(len(xml_list)):
            needed_pairs.add((idx, idx))
        
        print(f"  → Computing {len(needed_pairs)} VGG16 comparisons (instead of {len(xml_list)*len(xml_list)})")
        sys.stdout.flush()
        
        distance_cache = {}
        pair_distances = {}
        global_content_distance_max = 0
        processed = 0
        
        for (idx1, idx2) in needed_pairs:
            if (idx1, idx2) in distance_cache:
                continue
            
            xml_cur1 = xml_dir + xml_list[idx1]
            xml_cur2 = xml_dir + xml_list[idx2]
            img_cur1 = img_dir + img_list[idx1]
            img_cur2 = img_dir + img_list[idx2]
            dis_cur = process(xml_cur1, xml_cur2, img_cur1, img_cur2)
            dis_cur = round(dis_cur, 4)
            
            distance_cache[(idx1, idx2)] = dis_cur
            if dis_cur > global_content_distance_max:
                global_content_distance_max = dis_cur
            
            processed += 1
            if processed % 500 == 0:
                print(f"  → Content: Processed {processed}/{len(needed_pairs)} pairs ({processed/len(needed_pairs)*100:.1f}%)")
                sys.stdout.flush()
        
        # Ensure we have at least some distances
        if global_content_distance_max == 0:
            print(f"  ⚠ WARNING: global_content_distance_max is 0! Setting to 1.0")
            global_content_distance_max = 1.0
            
        print(f"  → Maximum content distance: {global_content_distance_max}")
        sys.stdout.flush()
        
        all_dist_list = [title]
        for i in range(len(xml_list)):
            report_id_i = int(data.iloc[i][0])
            dist_list = [report_id_i]
            
            for j in range(len(xml_list)):
                report_id_j = int(data.iloc[j][0])
                
                # Use indices for cache lookup
                computed = False
                if (i, j) in distance_cache:
                    raw_dist = distance_cache[(i, j)]
                    computed = True
                elif (j, i) in distance_cache:
                    raw_dist = distance_cache[(j, i)]
                    computed = True
                else:
                    raw_dist = global_content_distance_max # Max distance for unknown pairs

                dist_list.append(raw_dist)
                
                if computed and global_content_distance_max > 0:
                    norm_val = round(raw_dist / global_content_distance_max, 4)
                    repo_i, id_i = idx_to_repo_id.get(i, ('Unknown', report_id_i))
                    repo_j, id_j = idx_to_repo_id.get(j, ('Unknown', report_id_j))
                    pair_distances[(repo_i, id_i, id_j)] = norm_val
            
            all_dist_list.append(dist_list)
            
        # Normalize entire matrix
        for i in range(1, len(all_dist_list)):
            for j in range(1, len(all_dist_list[0])):
                value = all_dist_list[i][j]
                if global_content_distance_max > 0:
                    all_dist_list[i][j] = round(value / global_content_distance_max, 4)
                else:
                    all_dist_list[i][j] = 0.0
            all_dist_list[i][0] = int(all_dist_list[i][0])
        
        print(f"  → Content features complete (optimized)!")
        sys.stdout.flush()
        return all_dist_list, pair_distances
    
    # LEGACY MODE: Compute full N×N matrix
    print(f"  → LEGACY MODE: Computing full N×N matrix ({len(xml_list)*len(xml_list)} comparisons)")
    sys.stdout.flush()

    # Check for checkpoint file
    checkpoint_file = 'content_features_checkpoint.pkl'
    start_idx = 0
    all_dist_list = [title]
    global_content_distance_max = 0
    
    if os.path.exists(checkpoint_file):
        print(f"  → Found checkpoint file, resuming from saved progress...")
        sys.stdout.flush()
        with open(checkpoint_file, 'rb') as f:
            checkpoint_data = pickle.load(f)
            start_idx = checkpoint_data['last_completed_idx'] + 1
            all_dist_list = checkpoint_data['all_dist_list']
            global_content_distance_max = checkpoint_data.get('global_max', 0)
        print(f"  → Resuming from image {start_idx+1}/{len(xml_list)}")
        sys.stdout.flush()
    
    total_comparisons = len(xml_list) * len(xml_list)
    print(f"  → Computing content features for {len(xml_list)} images ({total_comparisons} comparisons)")
    sys.stdout.flush()

    for i in range(start_idx, len(xml_list)):
        if i % 10 == 0 or i == start_idx:
            print(f"  → Processing image {i+1}/{len(xml_list)} ({(i*len(xml_list))/total_comparisons*100:.1f}% complete)")
            sys.stdout.flush()
        dist_list = [data.iloc[i][0]]
        for j in range(len(xml_list)):
            xml_cur1 = xml_dir + xml_list[i]
            xml_cur2 = xml_dir + xml_list[j]
            img_cur1 = img_dir + img_list[i]
            img_cur2 = img_dir + img_list[j]
            dis_cur = process(xml_cur1, xml_cur2, img_cur1, img_cur2)
            dis_cur = round(dis_cur, 4)
            if dis_cur > global_content_distance_max:
                global_content_distance_max = dis_cur
            dist_list.append(dis_cur)
        all_dist_list.append(dist_list)
        
        # Save checkpoint every 5 images
        if (i + 1) % 5 == 0:
            with open(checkpoint_file, 'wb') as f:
                pickle.dump({
                    'last_completed_idx': i,
                    'all_dist_list': all_dist_list,
                    'global_max': global_content_distance_max
                }, f)

    # Remove checkpoint file on completion
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        
    # Normalize distances
    for i in range(1, len(all_dist_list)):
        for j in range(1, len(all_dist_list[0])):
            # Avoid division by zero
            if global_content_distance_max > 0:
                all_dist_list[i][j] /= global_content_distance_max
                all_dist_list[i][j] = round(all_dist_list[i][j], 4)
            else:
                all_dist_list[i][j] = 0.0
        all_dist_list[i][0] = int(all_dist_list[i][0])
    
    print(f"  → Content features complete!")
    sys.stdout.flush()
    return all_dist_list, {}
