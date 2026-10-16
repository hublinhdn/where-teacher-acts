import pandas as pd
import os

def load_epill_full_data():
    """
    Load the ePillID data and assign fixed labels. 
    The first 4902 labels match the ePillID benchmark label list exactly.
    """
    root_path = 'data/raw/ePillID'
    folds_dir = os.path.join(root_path, 'folds/pilltypeid_nih_sidelbls0.01_metric_5folds/base')
    label_path = os.path.join(folds_dir, 'pill_classes.txt')
    
    # 1. Load the MASTER file
    all_csv = os.path.join(folds_dir, 'pilltypeid_nih_sidelbls0.01_metric_5folds_all.csv')
    if not os.path.exists(all_csv):
        raise FileNotFoundError(f"File not found: {all_csv}")
    
    # Read pilltype_id as a string from the start
    df_all = pd.read_csv(all_csv, dtype={'pilltype_id': str})
    # Clean the strings (in case pandas appended .0 to the end of an ID)
    df_all['pilltype_id'] = df_all['pilltype_id'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
    
    # 2. LABEL ENCODING
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            # Strip extra whitespace and newline characters
            classes = [line.strip() for line in f.readlines() if line.strip()]
        
        # The labels actually present in the CSV
        actual_labels = df_all['pilltype_id'].unique().tolist()

        # Split them into matched labels and extra labels
        matched_labels = [l for l in actual_labels if l in classes]
        extra_labels = [l for l in actual_labels if l not in classes]

        full_classes = classes + sorted(extra_labels)
        df_all['label_idx'] = pd.Categorical(df_all['pilltype_id'], categories=full_classes).codes

        # ACCURATE LOG: report how many labels actually match
        print(f"Matched {len(matched_labels)}/{len(classes)} labels of the ePillID benchmark list.")
        if len(extra_labels) > 0:
            print(f"Added {len(extra_labels)} new labels found in the CSV.")
        
        # # Find the labels that are in the CSV but missing from the .txt file
        # extra_labels = [l for l in actual_labels if l not in classes]
        
        # # Merge: the benchmark labels first, then the new labels
        # full_classes = classes + sorted(extra_labels)
        
        # Assign label_idx
        # df_all['label_idx'] = pd.Categorical(df_all['label_code_id'], categories=full_classes).codes
        
        # print(f"Matched {len(classes)} labels of the ePillID benchmark list.")
        # if len(extra_labels) > 0:
        #     print(f"Added {len(extra_labels)} new labels found in the CSV.")
    else:
        print(f"Not found: {label_path}. Creating labels with factorize.")
        df_all['label_idx'], _ = pd.factorize(df_all['pilltype_id'])

    # 3. ASSIGN FOLDS (the standard ePillID fold logic, unchanged)
    df_all['fold'] = -1 
    for i in range(5):
        fold_csv_path = os.path.join(folds_dir, f'pilltypeid_nih_sidelbls0.01_metric_5folds_{i}.csv')
        if os.path.exists(fold_csv_path):
            df_f = pd.read_csv(fold_csv_path, dtype={'image_path': str})
            # Assign each fold from the image list in its standard fold file
            df_all.loc[df_all['image_path'].isin(df_f['image_path']), 'fold'] = i
            
    # 4. NORMALIZE is_ref
    df_all['is_ref'] = df_all['is_ref'].map({True: 1, False: 0, 1: 1, 0: 0, 'True': 1, 'False': 0}).astype(int)
    df_all['is_front'] = df_all['is_front'].map({True: 1, False: 0, 1: 1, 0: 0, 'True': 1, 'False': 0}).astype(int)

    # 1. Build a string that combines the pill ID and the front/back side (for example "105_1", "105_0")
    df_all['sub_label_raw'] = df_all['label_idx'].astype(str) + "_" + df_all['is_front'].astype(str)
    # 2. pd.factorize renumbers these strings as consecutive integers: 0, 1, 2, ..., N-1
    df_all['sub_label_idx'] = pd.factorize(df_all['sub_label_raw'])[0]
    
    # Check the final result
    total_samples = len(df_all)
    ref_count = (df_all['is_ref'] == 1).sum()
    cons_count = (df_all['is_ref'] == 0).sum()

    total_classes = df_all['label_idx'].nunique()
    total_sub_classes = df_all['sub_label_idx'].nunique()
    
    print(f"Loaded successfully: Total {total_samples} | Ref: {ref_count} | Cons: {cons_count} | total_classes: {total_classes} | total_sub_classes:{total_sub_classes}")
    return df_all

if __name__ == "__main__":
    df = load_epill_full_data()
    if len(df) > 0:
        print(df[['image_path', 'pilltype_id', 'label_idx', 'is_ref', 'fold']].head())
    else:
        print("Error: the DataFrame is still empty!")