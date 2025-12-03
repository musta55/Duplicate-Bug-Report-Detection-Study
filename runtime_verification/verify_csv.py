import pandas as pd

def parse_custom_list(list_str):
    """Parse list string in format [123| 456| 789]"""
    if not isinstance(list_str, str):
        return []
    # Remove brackets
    clean_str = list_str.strip('[]')
    if not clean_str:
        return []
    # Split by pipe
    parts = clean_str.split('|')
    # Convert to int, stripping whitespace
    return [int(p.strip()) for p in parts if p.strip()]

def verify_csv_claims(file_path, dataset_type):
    print(f"\n{'='*50}")
    print(f"Verifying: {file_path} ({dataset_type})")
    print(f"{'='*50}")
    
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # Print columns to map letters to names
    # Assuming 0-indexed: A=0, B=1, C=2, D=3, E=4, ... L=11, M=12
    cols = df.columns.tolist()
    print("Columns found:")
    for i, col in enumerate(cols):
        print(f"  {chr(65+i)} ({i}): {col}")

    # Verify Column B (query)
    col_b_name = cols[1]
    print(f"\n[B] Query Column: '{col_b_name}'")
    
    # Verify Column C (query_has_image)
    col_c_name = cols[2]
    print(f"\n[C] Query Has Image Column: '{col_c_name}'")
    unique_flags = df[col_c_name].unique()
    print(f"  Unique values: {unique_flags}")
    if dataset_type == "FILTERED":
        all_true = all(df[col_c_name] == True)
        print(f"  All TRUE for FILTERED? {all_true}")

    # Verify List Columns (D, E, L, M)
    # D=3, E=4, L=11, M=12
    col_d_name = cols[3] if len(cols) > 3 else None
    col_e_name = cols[4] if len(cols) > 4 else None
    col_l_name = cols[11] if len(cols) > 11 else None
    col_m_name = cols[12] if len(cols) > 12 else None

    print(f"\n[D] Corpus Column: '{col_d_name}'")
    print(f"[E] Ground Truth Column: '{col_e_name}'")
    print(f"[L] GT with Images Column: '{col_l_name}'")
    print(f"[M] Corpus with Images Column: '{col_m_name}'")

    # Sample check on first 5 rows
    print("\nChecking logic on first 5 rows:")
    for idx, row in df.head(5).iterrows():
        try:
            # Parse lists using custom parser
            d_list = parse_custom_list(str(row[col_d_name]))
            e_list = parse_custom_list(str(row[col_e_name]))
            l_list = parse_custom_list(str(row[col_l_name]))
            m_list = parse_custom_list(str(row[col_m_name]))
            
            print(f"  Row {idx}: Query {row[col_b_name]}")
            
            # Check subsets
            l_subset_e = set(l_list).issubset(set(e_list))
            m_subset_d = set(m_list).issubset(set(d_list))
            
            print(f"    L subset of E? {l_subset_e} (L={len(l_list)}, E={len(e_list)})")
            print(f"    M subset of D? {m_subset_d} (M={len(m_list)}, D={len(d_list)})")
            
            if not l_subset_e:
                print(f"      VIOLATION: L items not in E: {set(l_list) - set(e_list)}")
            
        except Exception as e:
            print(f"    Error parsing row {idx}: {e}")

if __name__ == "__main__":
    verify_csv_claims("Overall - FILTERED_trimmed_year_1_corpus_with_gt.csv", "FILTERED")
    verify_csv_claims("Overall - FULL_trimmed_year_1_corpus_with_gt.csv", "FULL")
