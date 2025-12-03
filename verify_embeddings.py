import pickle
import sys
import os

files = [
    'embeddings/text_embeddings.pkl',
    'embeddings/structure_embeddings.pkl',
    'embeddings/content_embeddings.pkl'
]

for fpath in files:
    if not os.path.exists(fpath):
        print(f"{fpath}: Not found")
        continue
        
    try:
        size_mb = os.path.getsize(fpath) / (1024 * 1024)
        print(f"Checking {fpath} ({size_mb:.2f} MB)...")
        
        with open(fpath, 'rb') as f:
            data = pickle.load(f)
            
        print(f"  Count: {len(data)}")
        if len(data) > 0:
            first_key = next(iter(data))
            print(f"  Sample Key: {first_key}")
            print(f"  Sample Type: {type(data[first_key])}")
            
            if 'content' in fpath:
                # Check size of one entry
                import sys
                sample = data[first_key]
                print(f"  Sample Entry Length: {len(sample)}")
                if len(sample) > 0:
                    print(f"  Sample Item Keys: {sample[0].keys()}")
                    if 'vector' in sample[0]:
                        print(f"  Vector Shape: {sample[0]['vector'].shape}")
                        
    except Exception as e:
        print(f"  Error reading {fpath}: {e}")
