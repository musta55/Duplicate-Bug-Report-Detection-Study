import os
import cv2
import json
import numpy as np
import pandas as pd
from apted import APTED, helpers
import xml.dom.minidom as minidom

import image.widget as widget


def flat_bounding(image, canny_sigma=0.33, dilate_count=4):
    """
    Find external bounding of widgets in a screenshot.

    :param image: Input image of type `ndarray`.
    :param canny_sigma: Sigma parameter for canny to control the thresholds.
    :param dilate_count: Number of iterations to perform dilation.
    :return: A list of widget bounding.
        Each bounding is represented as (x, y, w, h) where (x, y) stands for the position
        of the top-left vertex of the bounding, and (w, h) stands for the width and height of the bounding.
    """
    v = np.median(image)
    img_binary = cv2.Canny(image, int(max(0, (1 - canny_sigma) * v)), int(min(255, (1 + canny_sigma) * v)))
    img_dilated = cv2.dilate(img_binary, None, iterations=dilate_count)
    contours, _ = cv2.findContours(img_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding = []
    for c in contours:
        bounding.append(cv2.boundingRect(c))
    return bounding


def process_bounding(img, bounding_list):
    non_blanks = []
    for bounding in bounding_list:
        x, y, w, h = bounding
        node = img[y:y + h, x:x + w, :]
        if not np.count_nonzero(node) == 0 and not np.count_nonzero(255 - node) == 0:
            non_blanks.append(bounding)

    enlarge_width = 5
    img_h, img_w, _ = img.shape
    enlarged_bounding = []
    for x, y, w, h in bounding_list:
        enlarged_x = max(0, x - enlarge_width)
        enlarged_y = max(0, y - enlarge_width)
        enlarged_w = min(w + 2 * enlarge_width, img_w - enlarged_x)
        enlarged_h = min(h + 2 * enlarge_width, img_h - enlarged_y)
        enlarged_bounding.append((enlarged_x, enlarged_y, enlarged_w, enlarged_h))
    return enlarged_bounding


def layout(bounding, resolution):
    """
        Data Structure Generation
    """
    # Group Generation
    groups = group_generation(basic_row_generation(bounding), resolution)

    # Row & Column Generation.

    line_merge_threshold = 1.5
    column_merge_threshold = 1.5

    for group in groups:
        group[0][0][1] = group[1]
        group[0][-1][2] = group[2]
        nodes = [(y, h) for node_row in group[0] for _, y, _, h in node_row[0]]
        lines = sorted(set([y for y, _ in nodes] + [y + h for y, h in nodes] + [group[1], group[2]]))
        merge_close_lines(lines, line_merge_threshold * resolution[1] / 100)
        lines[0] = group[1]
        lines[-1] = group[2]

        rows = []
        for top, bottom in zip(lines[:-1], lines[1:]):
            filtered_basic_rows = [row for row in group[0] if not (bottom <= row[1] or top >= row[2])]
            filtered_nodes = [(x, w) for row in filtered_basic_rows
                              for x, y, w, h in row[0] if not (y + h <= top or y >= bottom)]
            cols = sorted(set([x for x, _ in filtered_nodes] + [x + w for x, w in filtered_nodes]))
            if len(cols) == 0 or not cols[0] == 0:
                cols = [0] + cols
            if not cols[-1] == resolution[0]:
                cols.append(resolution[0])
            if len(cols) > 0:
                merge_close_lines(cols, column_merge_threshold * resolution[0] / 100)
                cols[0] = 0
                cols[-1] = resolution[0]
                cols = [[left, right] for left, right in zip(cols[:-1], cols[1:])]
            rows.append([cols, top, bottom])
        group[0] = rows

    return groups


def basic_row_generation(bounding):
    basic_rows = []
    for bounding in sorted(bounding, key=lambda b: b[1] + b[3] / 2):
        x, y, w, h = bounding
        center_y = y + h / 2
        found = False
        for row in basic_rows:
            ceiling = row[1]
            ground = row[2]
            if ceiling <= center_y <= ground:
                row[0].append(bounding)
                row[1] = min(ceiling, y)
                row[2] = max(ground, y + h)
                found = True
                break
        if not found:
            basic_rows.append([[bounding], y, y + h])
    return basic_rows


def group_generation(basic_rows, resolution):
    # Initial Group generation.
    groups = [[[row], row[1], row[2]] for row in basic_rows]
    surviving = [True] * len(groups)
    group_count = 0
    while not len(groups) == group_count:
        group_count = len(groups)
        for i, group_i in enumerate(groups):
            for j, group_j in enumerate(groups):
                if not i == j and surviving[j] and \
                        group_j[1] <= (group_i[1] + group_i[2]) / 2 <= group_j[2]:
                    group_j[0] += group_i[0]
                    group_j[1] = min(group_j[1], group_i[1])
                    group_j[2] = max(group_j[2], group_i[2])
                    surviving[i] = False
                    break
    groups = [group for i, group in enumerate(groups) if surviving[i]]

    # Group separation.
    for i, group_i in enumerate(groups):
        for group_j in groups[i + 1:]:
            if group_j[1] < group_i[2] < group_j[2]:
                group_i[2] = int((group_i[2] + group_j[1]) / 2)
                group_j[1] = group_i[2]
            elif group_j[1] < group_i[1] < group_j[2]:
                group_i[1] = int((group_i[1] + group_j[2]) / 2)
                group_j[2] = group_i[1]

    # Group simplification.
    if len(groups) > 0:
        groups[0][1] = 0
        groups[-1][2] = resolution[1]
    for prev, cur in zip(groups[:-1], groups[1:]):
        if prev[2] < cur[1]:
            cur[1] = int((prev[2] + cur[1]) / 2)
            prev[2] = cur[1]
    g_threshold = 1.5 * resolution[1] / 100
    surviving = [True] * len(groups)
    for i in range(len(groups)):
        if groups[i][2] - groups[i][1] < g_threshold:
            if i - 1 < 0 and i + 1 < len(groups):
                groups[i + 1][0] += groups[i][0]
                groups[i + 1][1] = groups[i][1]
            elif i + 1 >= len(groups) and i - 1 >= 0:
                groups[i - 1][0] += groups[i][0]
                groups[i - 1][2] = groups[i][2]
            elif i - 1 >= 0 and i + 1 < len(groups):
                height_a = groups[i - 1][2] - groups[i - 1][1]
                height_b = groups[i + 1][2] - groups[i + 1][1]
                if height_a < height_b:
                    groups[i - 1][0] += groups[i][0]
                    groups[i - 1][2] = groups[i][2]
                else:
                    groups[i + 1][0] += groups[i][0]
                    groups[i + 1][1] = groups[i][1]
            surviving[i] = False
    return [group for i, group in enumerate(groups) if surviving[i]]


def merge_close_lines(lines, threshold=5):
    i = 0
    if len(lines) < 2:
        return
    first = lines[0]
    last = lines[-1]
    while i + 1 < len(lines):
        if lines[i + 1] - lines[i] < threshold:
            lines[i] = int((lines[i] + lines[i + 1]) / 2)
            lines.pop(i + 1)
        else:
            i += 1
    if len(lines) < 2:
        lines.clear()
        lines.extend([first, last])


def exec(path):
    img = cv2.imread(path)
    groups = layout(process_bounding(img, widget.extract(path)), (img.shape[1], img.shape[0]))
    return groups


def group_json(groups):
    gson = [{
        'rows': [{
            'cols': [{
                'left': col[0],
                'right': col[1]
            } for col in row[0]],
            'top': row[1],
            'bottom': row[2]
        } for row in group[0]],
        'top': group[1],
        'bottom': group[2]
    } for group in groups]
    return json.dumps(gson, indent=2)


def GenerateXML(list_group_line, list_row_line, list_col_line, width, height, xml_path):
    impl = minidom.getDOMImplementation()
    dom = impl.createDocument(None, 'groups', None)
    root = dom.documentElement
    len_g = len(list_group_line)
    counter = 0

    ans = ""
    list_g_str = []
    for i in range(len_g):

        str_g = ""

        group = dom.createElement("group")
        x = 0 if i == 0 else list_group_line[i - 1]
        y = 0
        w = width
        h = list_group_line[i] if i == 0 else list_group_line[i] - list_group_line[i - 1]
        group.setAttribute("x", str(x))
        group.setAttribute("y", str(y))
        group.setAttribute("w", str(w))
        group.setAttribute("h", str(h))
        root.appendChild(group)

        len_r = len(list_row_line[i]) - 1

        list_r_str = []
        for j in range(1, len_r + 1):

            row = dom.createElement("row")
            x = list_row_line[i][j - 1]
            y = 0
            w = width
            h = list_row_line[i][j] - list_row_line[i][j - 1]
            if h < 0:
                h = 0
            row.setAttribute("x", str(x))
            row.setAttribute("y", str(y))
            row.setAttribute("w", str(w))
            row.setAttribute("h", str(h))
            group.appendChild(row)

            len_c = len(list_col_line[counter])
            str_c = "{1}" * len_c
            str_r = "{1" + str_c + "}"
            list_r_str.append(str_r)
            for k in range(len_c):
                col = dom.createElement("col")
                x = list_row_line[i][j - 1]
                y = 0 if k == 0 else list_col_line[counter][k - 1]
                w = list_col_line[counter][k] if k == 0 else list_col_line[counter][k] - list_col_line[counter][k - 1]
                h = list_row_line[i][j] - list_row_line[i][j - 1]
                if h < 0:
                    h = 0
                col.setAttribute("x", str(x))
                col.setAttribute("y", str(y))
                col.setAttribute("w", str(w))
                col.setAttribute("h", str(h))
                row.appendChild(col)

            counter += 1
        for s in list_r_str:
            str_g += s
        str_g = "{1" + str_g + "}"
        list_g_str.append(str_g)

    f = open(xml_path, 'w')
    dom.writexml(f, addindent='    ', newl='\n')
    f.close()
    for s in list_g_str:
        ans += s
    ans = "{1" + ans + "}"
    return ans

def load_layout_string_from_xml(xml_path):
    """Return the cached layout string if the XML file already exists."""
    if not os.path.exists(xml_path):
        return None
    try:
        dom = minidom.parse(xml_path)
    except Exception:
        return None

    def _child_elements(node, tag_name):
        return [child for child in node.childNodes
                if child.nodeType == child.ELEMENT_NODE and child.tagName == tag_name]

    groups = _child_elements(dom.documentElement, 'group')
    group_strings = []
    for group in groups:
        rows = _child_elements(group, 'row')
        row_strings = []
        for row in rows:
            cols = _child_elements(row, 'col')
            row_strings.append('{1' + ('{1}' * len(cols)) + '}')
        group_strings.append('{1' + ''.join(row_strings) + '}')

    return '{1' + ''.join(group_strings) + '}' if group_strings else '{1}'


def getStrs(file_dir, xml_dir="file/xml_file/"):
    import sys
    # Ensure xml_dir ends with /
    if not xml_dir.endswith('/'):
        xml_dir = xml_dir + '/'
    # Create xml_dir if it doesn't exist
    os.makedirs(xml_dir, exist_ok=True)
    
    pic_files = sorted([f for f in os.listdir(file_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    tree_str_list = ["" for _ in range(len(pic_files))]
    for pic_idx, pic in enumerate(pic_files):
        if pic_idx % 500 == 0:
            print(f"  → [DEBUG] getStrs processing image {pic_idx}/{len(pic_files)}: {pic}")
            sys.stdout.flush()
        pic_num = pic.split("_")
        num = pic_num[2].split(".")[0]

        int_num = int(num)
        xml_path = os.path.join(xml_dir, f"layout{num}.xml")
        cached_str = load_layout_string_from_xml(xml_path)
        if cached_str:
            tree_str_list[int_num] = cached_str
            continue

        img = cv2.imread(file_dir + pic)
        w, h = img.shape[1], img.shape[0]
        groups = exec(file_dir + pic)
        list_group_line = []
        list_row_line = []
        list_col_line = []
        for group in groups:
            list_group_line.append(group[2])
            cv2.line(img, (0, group[2]), (w, group[2]), (255, 0, 0), thickness=3)

        for group in groups:
            list_row_temp = []
            for i in range(len(group[0])):
                if i == 0:
                    list_row_temp.append(group[0][i][1])
                list_row_temp.append(group[0][i][2])
            list_row_line.append(list_row_temp)

        for group in groups:
            for i in range(len(group[0])):
                col_len = len(group[0][i][0])
                list_col_temp = []
                for j in range(col_len):
                    col_index = group[0][i][0][j][1]
                    list_col_temp.append(col_index)
                list_col_line.append(list_col_temp)

        tree_str_list[int_num] = GenerateXML(list_group_line, list_row_line, list_col_line, w, h, xml_path)


    return tree_str_list


def getSTdis(pic_dir, label_csv, xml_dir="file/xml_file/", sample_df=None, query_to_valid_corpus=None, seq_to_id=None, parquet_df=None):
    """
    Get structure distance matrix - OPTIMIZED to only compute query-corpus pairs.
    
    If sample_df and query_to_valid_corpus are provided, only computes distances for
    query-corpus pairs. Otherwise, computes the full N×N matrix.
    """
    import sys
    print("  → [DEBUG] Entered getSTdis")
    sys.stdout.flush()
    try:
        # Use python engine to avoid C parser buffer overflows on large fields
        # on_bad_lines='skip' to ignore malformed rows
        data = pd.read_csv(label_csv, header=None, engine='python', on_bad_lines='skip', encoding='utf-8').drop([0])
    except Exception as e:
        print(f"  ⚠ Error reading CSV {label_csv}: {e}")
        # Fallback: try reading with default engine if python engine fails (unlikely)
        data = pd.read_csv(label_csv, header=None).drop([0])
    title = ["index"]
    pic_list = sorted([f for f in os.listdir(pic_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    for i in range(len(pic_list)):
        title.append(str(data.iloc[i][0]))
    str_list = getStrs(pic_dir, xml_dir)
    
    # Create mapping from (repo_name, report_id) to sequential index - CRITICAL for avoiding ID collisions!
    repo_id_to_idx = {}
    id_to_seq_idx = {}  # Keep simple mapping for backwards compatibility
    id_to_str = {}
    idx_to_repo_id = {}
    
    # Build repo_name lookup from parquet_df
    id_to_repo = {}
    if parquet_df is not None:
        for idx, row in parquet_df.iterrows():
            id_to_repo[row['id']] = row['repo_name']
    
    for i in range(len(str_list)):
        # i is the sequential index (seq_idx)
        seq_idx = i
        
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
            # Legacy fallback
            real_id = seq_idx
            repo_name = id_to_repo.get(real_id, 'Unknown')

        id_to_seq_idx[real_id] = i  # Map Real ID -> Seq Index
        id_to_str[real_id] = str_list[i] # Map Real ID -> Content (Wait, original code mapped Seq ID -> Content)
        
        # For compatibility with existing code that uses data.iloc[i][0] (which is seq_idx)
        # We should also map seq_idx -> content if needed, but str_list[i] does that.
        # The issue is later code uses id_to_str[report_id1] where report_id1 comes from CSV (seq_idx).
        # So id_to_str MUST be keyed by seq_idx for legacy parts, OR we update legacy parts.
        # Let's key it by seq_idx to be safe for the loop below.
        id_to_str[seq_idx] = str_list[i] 
        
        if repo_name != 'Unknown':
            repo_id_to_idx[(repo_name, real_id)] = i
            idx_to_repo_id[i] = (repo_name, real_id)
        else:
            idx_to_repo_id[i] = ('Unknown', real_id)
    
    # OPTIMIZED: Only compute query-corpus pairs if mapping provided
    print(f"  → DEBUG getSTdis: sample_df={sample_df is not None}, query_to_valid_corpus={query_to_valid_corpus is not None}, seq_to_id={seq_to_id is not None}, parquet_df={parquet_df is not None}")
    if query_to_valid_corpus is not None:
        print(f"  → DEBUG getSTdis: query_to_valid_corpus has {len(query_to_valid_corpus)} entries")
    if seq_to_id is not None:
        print(f"  → DEBUG getSTdis: seq_to_id has {len(seq_to_id)} entries")
    print(f"  → DEBUG getSTdis: repo_id_to_idx has {len(repo_id_to_idx)} entries")
    print(f"  → DEBUG getSTdis: id_to_repo has {len(id_to_repo)} entries")
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
        for idx in range(len(str_list)):
            needed_pairs.add((idx, idx))
        
        print(f"  → Computing {len(needed_pairs)} tree edit distances (instead of {len(str_list)*len(str_list)})")
        sys.stdout.flush()
        
        # Compute distances only for needed pairs
        distance_cache = {}
        pair_distances = {}
        global_edit_distance_max = 0
        processed = 0
        
        for (idx1, idx2) in needed_pairs:
            # Skip already computed pairs
            if (idx1, idx2) in distance_cache or (idx2, idx1) in distance_cache:
                continue

            # Use seq indices directly (robust against CSV parsing issues)
            report_id1 = idx1
            report_id2 = idx2

            # id_to_str is keyed by seq_idx for legacy compatibility
            src = id_to_str.get(report_id1, "")
            tgt = id_to_str.get(report_id2, "")

            try:
                tree1 = helpers.Tree.from_text(src)
                tree2 = helpers.Tree.from_text(tgt)
                apted = APTED(tree1, tree2)
                ted = apted.compute_edit_distance()
            except Exception as e:
                # If parsing fails for any pair, log and set to max placeholder; continue
                if processed < 5:
                    print(f"  ⚠ Structure distance computation failed for pair ({idx1},{idx2}): {e}")
                ted = global_edit_distance_max if global_edit_distance_max > 0 else 1

            distance_cache[(idx1, idx2)] = ted
            if ted >= global_edit_distance_max:
                global_edit_distance_max = ted

            processed += 1
            if processed % 1000 == 0:
                print(f"  → Structure: Processed {processed}/{len(needed_pairs)} pairs ({processed/len(needed_pairs)*100:.1f}%)")
                sys.stdout.flush()
        
        # Ensure we have at least some distances
        if global_edit_distance_max == 0:
            print(f"  ⚠ WARNING: global_edit_distance_max is 0! Setting to 1.0")
            global_edit_distance_max = 1.0
        
        print(f"  → Maximum edit distance: {global_edit_distance_max}")
        sys.stdout.flush()
        
        # Build matrix with computed distances (set others to max distance)
        all_dist_list = [title]
        for i in range(len(str_list)):
            # Use seq index as report id (robust)
            report_id_i = i
            dist_list = [report_id_i]
            
            for j in range(len(str_list)):
                report_id_j = j
                
                # Use indices for cache lookup
                computed = False
                if (i, j) in distance_cache:
                    raw_dist = distance_cache[(i, j)]
                    computed = True
                elif (j, i) in distance_cache:
                    raw_dist = distance_cache[(j, i)]
                    computed = True
                else:
                    # Not a query-corpus pair - set to maximum distance (completely dissimilar)
                    raw_dist = global_edit_distance_max

                dist_list.append(raw_dist)

                if computed and global_edit_distance_max > 0:
                    norm_val = round(raw_dist / global_edit_distance_max, 4)
                    repo_i, id_i = idx_to_repo_id.get(i, ('Unknown', report_id_i))
                    repo_j, id_j = idx_to_repo_id.get(j, ('Unknown', report_id_j))
                    pair_distances[(repo_i, id_i, id_j)] = norm_val
            
            all_dist_list.append(dist_list)
    
        # Normalize entire matrix
        for i in range(1, len(all_dist_list)):
            for j in range(1, len(all_dist_list[0])):
                value = all_dist_list[i][j]
                if global_edit_distance_max > 0:
                    all_dist_list[i][j] = round(value / global_edit_distance_max, 4)
                else:
                    all_dist_list[i][j] = 0.0
            all_dist_list[i][0] = int(all_dist_list[i][0])

        return all_dist_list, pair_distances

    else:
        # LEGACY MODE: Compute full N×N matrix (slow!)
        print(f"  → LEGACY MODE: Computing full N×N matrix ({len(str_list)*len(str_list)} comparisons)")
        sys.stdout.flush()
        
        global_edit_distance_max = 0
        all_dist_list = [title]
        
        for i in range(len(str_list)):
            if i % 10 == 0:
                print(f"  → Structure: Processing image {i+1}/{len(str_list)}")
                sys.stdout.flush()
            
            dist_list = [data.iloc[i][0]]
            for j in range(len(str_list)):
                src = str_list[i]
                tgt = str_list[j]
                tree1 = helpers.Tree.from_text(src)
                tree2 = helpers.Tree.from_text(tgt)
                apted = APTED(tree1, tree2)
                ted = apted.compute_edit_distance()
                if ted > global_edit_distance_max:
                    global_edit_distance_max = ted
                dist_list.append(ted)
            all_dist_list.append(dist_list)

    # Normalize distances
    for i in range(1, len(all_dist_list)):
        for j in range(1, len(all_dist_list[0])):
            # Avoid division by zero
            if global_edit_distance_max > 0:
                all_dist_list[i][j] /= global_edit_distance_max
                all_dist_list[i][j] = round(all_dist_list[i][j], 4)
            else:
                all_dist_list[i][j] = 0.0
        all_dist_list[i][0] = int(all_dist_list[i][0])

    return all_dist_list, {}
