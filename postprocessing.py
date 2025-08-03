import pandas as pd
from datetime import datetime

# Get current date in YYMMDD format
current_date = datetime.now().strftime('%y%m%d')

# Load CSV
df = pd.read_csv('data/out/250802_weave_export.csv')

# Select and rename columns
columns_to_keep = {
    'output.metadata.key_identity_id': 'key_identity_id',
    'output.metadata.company_name': 'company_name',
    'output.label_value_predicted': 'min_share_capital',
    'output.metadata.date': 'assembly_date',
    'output.metadata.ordentlich': 'assembly_type',
    'inputs.file_path': 'file_path'
}

# Create new dataframe with selected columns
processed_df = df[list(columns_to_keep.keys())].rename(columns=columns_to_keep)

# Convert assembly_type: True -> 1, False -> 2
processed_df['assembly_type'] = processed_df['assembly_type'].map({True: 1, False: 2})

# Add ascending ID column
processed_df['id'] = range(1, len(processed_df) + 1)

# Add empty source column
processed_df['source'] = ''

# Set source to "MANUAL" for specific min_share_capital values
manual_condition = (processed_df['min_share_capital'] == -2) | \
                   (processed_df['min_share_capital'] == -1) | \
                   (processed_df['min_share_capital'] > 97) | \
                   (processed_df['min_share_capital'].isna())
processed_df.loc[manual_condition, 'source'] = 'MANUAL'

# Set source to "LLM" for all other rows
processed_df.loc[processed_df['source'] == '', 'source'] = 'LLM'

# Reorder columns to put id first
column_order = ['id'] + [col for col in processed_df.columns if col != 'id']
processed_df = processed_df[column_order]

# Load manual annotations and merge them
try:
    manual_annotations = pd.read_csv('data/out/250803_manual_annotations.csv')
    
    # Merge the manual annotations with the processed dataframe
    merged_df = processed_df.merge(manual_annotations[['id', 'min_share_capital', 'diff_stamm_vs_vorzug']], 
                                   on='id', how='left', suffixes=('', '_manual'))
    
    # Update min_share_capital based on manual annotations:
    # - If manual annotation has numeric value: use that value
    # - If manual annotation has NaN: set to -3 (indicating explicitly no value)
    # - If no manual annotation: keep original value
    has_manual_annotation = merged_df['min_share_capital_manual'].notna() | merged_df['diff_stamm_vs_vorzug'].notna()
    
    # For rows with manual annotations but NaN min_share_capital, set to NaN
    nan_annotation_mask = has_manual_annotation & merged_df['min_share_capital_manual'].isna()
    merged_df.loc[nan_annotation_mask, 'min_share_capital'] = pd.NA
    
    # For rows with manual annotations and numeric min_share_capital, use the manual value
    numeric_annotation_mask = merged_df['min_share_capital_manual'].notna()
    merged_df.loc[numeric_annotation_mask, 'min_share_capital'] = merged_df.loc[numeric_annotation_mask, 'min_share_capital_manual']
    
    # Set diff_stamm_vs_vorzug to -1 for rows without manual annotations
    merged_df['diff_stamm_vs_vorzug'] = merged_df['diff_stamm_vs_vorzug'].fillna(-1)
    
    # Drop the temporary manual annotation column
    processed_df = merged_df.drop('min_share_capital_manual', axis=1)
    
    print(f"Applied {numeric_annotation_mask.sum()} numeric manual annotations")
    print(f"Set {nan_annotation_mask.sum()} rows to NaN (manual NaN annotations)")
    
except FileNotFoundError:
    print("Warning: 250803_manual_annotations.csv not found. Adding diff_stamm_vs_vorzug column with -1 values.")
    # If manual annotations file doesn't exist, just add the column with -1 values
    processed_df['diff_stamm_vs_vorzug'] = -1

# Apply patch file before saving
try:
    patch_df = pd.read_csv('data/out/250803_patch_file.csv')
    
    # Get IDs that are in the patch file
    patch_ids = patch_df['id'].tolist()
    
    # Remove rows from processed_df that have IDs matching the patch file
    processed_df = processed_df[~processed_df['id'].isin(patch_ids)]
    
    # Append the patch file rows
    processed_df = pd.concat([processed_df, patch_df], ignore_index=True)
    
    # Sort by ID to maintain order
    processed_df = processed_df.sort_values('id').reset_index(drop=True)
    
    print(f"Applied patch file: replaced {len(patch_ids)} rows")
    
except FileNotFoundError:
    print("Warning: 250803_patch_file.csv not found. Skipping patch application.")

# Convert key_identity_id to int
processed_df['key_identity_id'] = processed_df['key_identity_id'].astype(int)

# Save processed file
processed_df.to_csv(f'data/out/{current_date}_results.csv', index=False)

print(f"Processed {len(processed_df)} rows. Saved to data/{current_date}_results.csv")
