import os
import sys
import pickle
import pandas as pd
import numpy as np
import cv2
import jieba
import gensim
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
import xml.dom.minidom as xmldom
from apted import helpers

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

import image.vgg16 as vgg16

# Configuration
# Dataset Coverage:
#   - FILTERED dataset: 100% text + 100% images (all reports have screenshots)
#   - FULL dataset: 100% text + 10-12% images (only some reports have screenshots)
TEXT_MODEL_PATH = os.path.join(project_root, 'text/text_feature_extraction/bugdata_format_model_100')
PARQUET_FILE = os.path.join(project_root, 'Dataset/bug_reports_with_images.parquet')
GT_CSV = os.path.join(project_root, 'Dataset/Overall - FULL_trimmed_year_1_corpus_with_gt.csv')
IMG_DIR = os.path.join(project_root, 'file/pic_file_parquet_full/')
XML_DIR = os.path.join(project_root, 'file/xml_file_parquet_full/')
OUTPUT_DIR = os.path.join(project_root, 'embeddings/')

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load Word2Vec model
print("Loading Word2Vec model...")
try:
    word2vec_model = gensim.models.Word2Vec.load(TEXT_MODEL_PATH)
    print("Word2Vec model loaded.")
except Exception as e:
    print(f"Error loading Word2Vec model: {e}")
    sys.exit(1)

def parse_id_list(id_string):
    if pd.isna(id_string) or id_string == '[]': return []
    cleaned = id_string.strip('[]').strip()
    if not cleaned: return []
    return [int(x.strip()) for x in cleaned.split('|') if x.strip()]

def get_relevant_ids(filter_by_images=False):
    """
    Get set of relevant IDs from GT CSV.
    
    Args:
        filter_by_images: If True, only return IDs that have images.
                         For text embeddings, use False (all reports).
                         For structure/content embeddings, use True (only reports with images).
    
    Returns:
        set: Composite IDs (repo:id) of relevant reports
    """
    print(f"Loading GT CSV from {GT_CSV}...")
    df = pd.read_csv(GT_CSV)
    relevant_ids = set()
    
    for idx, row in df.iterrows():
        repo = row['Repository_Name']
        
        if filter_by_images:
            # Only include reports that have images
            # Query with image
            if row.get('query_has_image', False):
                relevant_ids.add(f"{repo}:{row['query']}")
            
            # GT issues with images
            if 'ground_truth_issues_with_images' in row:
                for gid in parse_id_list(row['ground_truth_issues_with_images']):
                    relevant_ids.add(f"{repo}:{gid}")
            
            # Corpus issues with images
            if 'corpus_issues_with_images' in row:
                for cid in parse_id_list(row['corpus_issues_with_images']):
                    relevant_ids.add(f"{repo}:{cid}")
        else:
            # Include all reports (for text embeddings)
            # Query
            relevant_ids.add(f"{repo}:{row['query']}")
            
            # GT
            for gid in parse_id_list(row['ground_truth']):
                relevant_ids.add(f"{repo}:{gid}")
                
            # Corpus
            for cid in parse_id_list(row['corpus']):
                relevant_ids.add(f"{repo}:{cid}")
            
    filter_str = " with images" if filter_by_images else ""
    print(f"Found {len(relevant_ids)} relevant reports{filter_str}.")
    return relevant_ids

def get_sentence_vector(model, s):
    """Compute sentence vector by averaging word vectors"""
    size = model.layer1_size
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

def parse_xml_widgets(xmlpath):
    """Parse XML to get widget bounding boxes [x, y, w, h]"""
    try:
        dom_obj = xmldom.parse(xmlpath)
        element_obj = dom_obj.documentElement
        sub_elements = element_obj.getElementsByTagName("col")
        
        parsed_list = []
        for i in range(len(sub_elements)):
            try:
                x = int(float(sub_elements[i].getAttribute("x")))
                y = int(float(sub_elements[i].getAttribute("y")))
                w = int(float(sub_elements[i].getAttribute("w")))
                h = int(float(sub_elements[i].getAttribute("h")))
                if w > 0 and h > 0:
                    parsed_list.append([x, y, w, h])
            except ValueError:
                continue
        return parsed_list
    except Exception as e:
        # print(f"Error parsing XML {xmlpath}: {e}")
        return []

def is_pure_color(img):
    """Check if image is single color"""
    if img.size == 0: return True
    if img.shape[0] < 2 or img.shape[1] < 2: return True
    
    # Fast check using std dev
    std = np.std(img, axis=(0,1))
    return np.all(std < 1.0) # Low variance means pure color

from text.text_feature_extraction.text_feature_extraction import text_feature_extraction

def generate_text_embeddings(parquet_df):
    """Generate and save text embeddings
    
    Note: All reports (100%) in both FILTERED and FULL datasets have text descriptions.
    """
    output_file = os.path.join(OUTPUT_DIR, 'text_embeddings.pkl')
    
    embeddings = {}
    if os.path.exists(output_file):
        print(f"Loading existing text embeddings from {output_file}")
        with open(output_file, 'rb') as f:
            embeddings = pickle.load(f)
    
    print(f"Generating text embeddings for {len(parquet_df)} reports...")
    
    # Collect descriptions for batch processing
    ids_to_process = []
    descriptions = []
    
    for idx, row in parquet_df.iterrows():
        report_id = row['id']
        repo_name = row['repo_name']
        composite_id = f"{repo_name}:{report_id}"
        
        if composite_id in embeddings:
            continue
            
        description = str(row.get('description', ''))
        # Original code uses description column.
        # Some reports might have title/comments too, but tfe uses description.
        ids_to_process.append(composite_id)
        descriptions.append(description)
        
    if not ids_to_process:
        print("All text embeddings already exist.")
        return

    print(f"Running TextCNN extraction for {len(descriptions)} descriptions...")
    
    # Process in chunks to allow saving progress
    CHUNK_SIZE = 1000
    total_chunks = (len(descriptions) + CHUNK_SIZE - 1) // CHUNK_SIZE
    
    for i in range(0, len(descriptions), CHUNK_SIZE):
        chunk_descriptions = descriptions[i:i+CHUNK_SIZE]
        chunk_ids = ids_to_process[i:i+CHUNK_SIZE]
        
        print(f"Processing chunk {i//CHUNK_SIZE + 1}/{total_chunks} ({len(chunk_descriptions)} reports)...")
        
        try:
            # Run TextCNN on chunk Core Semcluster
            results = text_feature_extraction(chunk_descriptions)
            
            print(f"Computing Word2Vec vectors for {len(results)} results...")
            for j, res in enumerate(results):
                cid = chunk_ids[j]
                
                # 5 embeddings and tests -> decode the currect embeddings or not

                # Problem Vector
                problems = res.get('problems_list', [])
                problem_text = ' '.join(problems)
                prob_vec = get_sentence_vector(word2vec_model, problem_text)
                
                # Procedure Vectors
                procedures = res.get('procedures_list', [])
                proc_vecs = []
                for step in procedures:
                    proc_vecs.append(get_sentence_vector(word2vec_model, step))
                    
                embeddings[cid] = {
                    'problem_vector': prob_vec,
                    'procedure_vectors': proc_vecs,
                    'raw_problems': problems,
                    'raw_procedures': procedures
                }
            
            # Save after each chunk
            print(f"Saving progress... ({len(embeddings)} total)")
            with open(output_file, 'wb') as f:
                pickle.dump(embeddings, f)
                
        except Exception as e:
            print(f"Error in text feature extraction chunk {i}: {e}")
            # Fallback to simple whole-text vector for this chunk
            print("Falling back to simple text vectors for this chunk...")
            for j, desc in enumerate(chunk_descriptions):
                cid = chunk_ids[j]
                prob_vec = get_sentence_vector(word2vec_model, desc)
                embeddings[cid] = {
                    'problem_vector': prob_vec,
                    'procedure_vectors': [],
                    'raw_problems': [desc],
                    'raw_procedures': []
                }
            # Save fallback
            with open(output_file, 'wb') as f:
                pickle.dump(embeddings, f)

    print(f"Saved {len(embeddings)} text embeddings.")

def generate_structure_embeddings(parquet_df):
    """Generate and save structure embeddings (Trees)
    
    Note: Structure features require XML layout files (from screenshots).
    - FILTERED dataset: 100% coverage (all have images)
    - FULL dataset: 10-12% coverage (only reports with screenshots)
    """
    output_file = os.path.join(OUTPUT_DIR, 'structure_embeddings.pkl')
    
    # Start with empty embeddings dict - only generate for reports in parquet_df
    # Do NOT load existing embeddings to avoid mixing old unfiltereddata with new filtered data
    embeddings = {}
            
    print(f"Generating structure embeddings...")
    count = 0
    
    # We need to map composite_id to XML file
    # The XML files are named layout{id}.xml or similar.
    # We need to know the mapping.
    # In the original code, it used sequential IDs.
    # Here we have the parquet_df which has 'id' and 'repo_name'.
    # We assume the XMLs are in XML_DIR and named by ID? 
    # Actually, the resume script extracts images to `report_img_{seq_idx}.png`.
    # And `image_main` calls `getSTdis` which calls `getStrs`.
    # `getStrs` reads `layout{num}.xml`.
    # We need to ensure we are looking at the right files.
    
    # Issue: The XML files in `file/xml_file_parquet_full/` might be named `layout{seq_idx}.xml` 
    # or `layout{real_id}.xml`.
    # Let's check the directory content.
    
    xml_files = os.listdir(XML_DIR)
    xml_map = {} # map ID -> filename
    for f in xml_files:
        if f.endswith('.xml'):
            # Assuming format layout{id}.xml
            try:
                fid = int(f.replace('layout', '').replace('.xml', ''))
                xml_map[fid] = f
            except:
                pass
                
    for idx, row in parquet_df.iterrows():
        report_id = row['id']
        repo_name = row['repo_name']
        composite_id = f"{repo_name}:{report_id}"
        
        if composite_id in embeddings:
            continue
            
        # Try to find XML
        # We try looking up by report_id
        if report_id in xml_map:
            xml_path = os.path.join(XML_DIR, xml_map[report_id])
            
            # Parse to Tree string (using logic from structure_feature.py)
            # We can't easily import the layout generation logic without OpenCV and image.
            # But if the XML exists, we can just read it and convert to the bracket string format.
            
            try:
                # We need the bracket string format "{1{1}{1}}" for APTED
                # structure_feature.py has `load_layout_string_from_xml`
                from image.structure_feature import load_layout_string_from_xml
                tree_str = load_layout_string_from_xml(xml_path)
                
                if tree_str:
                    # Parse to Tree object
                    tree_obj = helpers.Tree.from_text(tree_str)
                    embeddings[composite_id] = tree_obj
            except Exception as e:
                # print(f"Error processing structure for {composite_id}: {e}")
                pass
        
        count += 1
        if count % 1000 == 0:
            print(f"Processed {count} structure records...")
            with open(output_file, 'wb') as f:
                pickle.dump(embeddings, f)

    with open(output_file, 'wb') as f:
        pickle.dump(embeddings, f)
    coverage = (len(embeddings) / len(parquet_df) * 100) if len(parquet_df) > 0 else 0
    print(f"Saved {len(embeddings)} structure embeddings ({coverage:.1f}% coverage).")

def generate_content_embeddings(parquet_df):
    """Generate and save content embeddings (VGG16 vectors for widgets)
    
    Note: Content features require screenshots with extractable widgets.
    - FILTERED dataset: 100% coverage (all have images)
    - FULL dataset: 10-12% coverage (only reports with screenshots)
    """
    output_file = os.path.join(OUTPUT_DIR, 'content_embeddings.pkl')
    
    # Start with empty embeddings dict - only generate for reports in parquet_df
    # Do NOT load existing embeddings to avoid mixing old unfiltered data with new filtered data
    embeddings = {}
            
    print(f"Generating content embeddings...")
    
    # Map IDs to files
    xml_files = {int(f.replace('layout', '').replace('.xml', '')): f 
                 for f in os.listdir(XML_DIR) if f.endswith('.xml') and 'layout' in f}
    
    # Image files: report_img_{seq_idx}.png OR report_img_{id}.png?
    # The resume script saves as `report_img_{seq_idx}.png`.
    # But we need to map Real ID -> Image File.
    # We need the `id_to_seq` mapping from the extraction phase.
    # Since we don't have that easily, let's look at the image directory.
    # If images are named by seq_idx, we have a problem unless we reconstruct the mapping.
    
    # Alternative: The parquet file HAS the image binary!
    # We can load the image directly from the dataframe row without relying on the file on disk.
    # This is much safer and robust.
    
    # Batch processing
    BATCH_SIZE = 32
    batch_widgets = [] # list of (composite_id, bbox, crop_img)
    
    model = vgg16.get_model()
    
    count = 0
    
    for idx, row in parquet_df.iterrows():
        report_id = row['id']
        repo_name = row['repo_name']
        composite_id = f"{repo_name}:{report_id}"
        
        if composite_id in embeddings:
            continue
            
        # Check if we have XML (widgets)
        if report_id not in xml_files:
            continue
            
        xml_path = os.path.join(XML_DIR, xml_files[report_id])
        widgets = parse_xml_widgets(xml_path)
        
        if not widgets:
            continue
            
        # Load Image
        image_data = row.get('image')
        if image_data is None:
            continue
            
        try:
            if isinstance(image_data, dict):
                image_bytes = image_data.get('bytes')
            else:
                image_bytes = image_data
                
            if not image_bytes:
                continue
                
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                continue
                
            # Process widgets
            for bbox in widgets:
                x, y, w, h = bbox
                # Crop
                crop = img[y:y+h, x:x+w]
                
                if crop.size == 0: continue
                if is_pure_color(crop): continue
                
                # Resize for VGG16
                crop = cv2.resize(crop, (224, 224))
                
                batch_widgets.append({
                    'id': composite_id,
                    'bbox': bbox,
                    'img': crop
                })
                
        except Exception as e:
            print(f"Error processing image {composite_id}: {e}")
            continue
            
        # Process batch if full
        if len(batch_widgets) >= BATCH_SIZE:
            process_batch(model, batch_widgets, embeddings)
            batch_widgets = []
            
        count += 1
        if count % 100 == 0:
            print(f"Processed {count} content records...")
            if count % 500 == 0:
                with open(output_file, 'wb') as f:
                    pickle.dump(embeddings, f)
    
    # Process remaining
    if batch_widgets:
        process_batch(model, batch_widgets, embeddings)
        
    with open(output_file, 'wb') as f:
        pickle.dump(embeddings, f)
        
    reports_with_content = len(set(cid for cid in embeddings.keys()))
    coverage = (reports_with_content / len(parquet_df) * 100) if len(parquet_df) > 0 else 0
    print(f"Saved content embeddings for {reports_with_content} reports ({coverage:.1f}% coverage)")
    print(f"Saved {len(embeddings)} content embeddings.")

def process_batch(model, batch, embeddings):
    """Run VGG16 on batch and store results"""
    if not batch: return
    
    images = np.array([item['img'] for item in batch])
    
    # Preprocess
    images = preprocess_input(images)
    
    # Predict
    features = model.predict(images, verbose=0)
    
    # Global Average Pooling: (N, 14, 14, 512) -> (N, 512)
    if len(features.shape) == 4:
        features = np.mean(features, axis=(1, 2))
    
    # Store
    for i, item in enumerate(batch):
        cid = item['id']
        bbox = item['bbox']
        vec = features[i]
        
        if cid not in embeddings:
            embeddings[cid] = []
            
        embeddings[cid].append({
            'bbox': bbox,
            'vector': vec
        })

def main():
    print("Loading Parquet data...")
    import pyarrow.parquet as pq
    table = pq.read_table(PARQUET_FILE)
    df = table.to_pandas()
    print(f"Loaded {len(df)} reports.")
    
    # Create composite ID column
    df['composite_id'] = df['repo_name'] + ':' + df['id'].astype(str)
    
    # 1. TEXT EMBEDDINGS - ALL reports
    print("\n" + "="*60)
    print("GENERATING TEXT EMBEDDINGS (100% expected)")
    print("="*60)
    relevant_ids_all = get_relevant_ids(filter_by_images=False)
    df_text = df[df['composite_id'].isin(relevant_ids_all)].copy()
    print(f"Processing {len(df_text)} reports for text embeddings")
    generate_text_embeddings(df_text)
    
    # 2. STRUCTURE EMBEDDINGS - Only reports with images
    print("\n" + "="*60)
    print("GENERATING STRUCTURE EMBEDDINGS (only reports with images)")
    print("Expected coverage: FILTERED=100%, FULL=10-12%")
    print("="*60)
    relevant_ids_with_images = get_relevant_ids(filter_by_images=True)
    df_structure = df[df['composite_id'].isin(relevant_ids_with_images)].copy()
    print(f"Processing {len(df_structure)} reports for structure embeddings")
    generate_structure_embeddings(df_structure)
    
    # 3. CONTENT EMBEDDINGS - Only reports with images
    print("\n" + "="*60)
    print("GENERATING CONTENT EMBEDDINGS (VGG16, only reports with images)")
    print("Expected coverage: FILTERED=100%, FULL=10-12%")
    print("="*60)
    df_content = df[df['composite_id'].isin(relevant_ids_with_images)].copy()
    print(f"Processing {len(df_content)} reports for content embeddings")
    generate_content_embeddings(df_content)
    
    print("\n" + "="*60)
    print("EMBEDDING GENERATION COMPLETE")
    print("="*60)
    
    print("Done generating embeddings.")

if __name__ == '__main__':
    main()
