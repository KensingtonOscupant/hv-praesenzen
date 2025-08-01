import pandas as pd
from sklearn.model_selection import train_test_split

# ---- 1. load the full dataset ------------------------------------------------
df = pd.read_csv("data/250730_full_set.csv")          # <- your file name here

# Drop specified columns
df = df.drop(columns=['correct', 'error', 'explanation', 'comment'])

# Set label_predicted to 0 for all rows
df['label_predicted'] = ""


# ---- 2. choose the column to stratify on -------------------------------------
# If you have a label column (e.g. "kpi_present"), stratify on it; otherwise set
# stratify_col = None and the split will be purely random.
stratify_col = None  # change or set to None

# ---- 3. 60 / 20 / 20 split ---------------------------------------------------
train_df, temp_df = train_test_split(
    df,
    test_size=0.40,                # leave 40 % for dev+test
    stratify=stratify_col,
    random_state=42,
)

# Re-use the same stratify column for the smaller split
stratify_temp = temp_df["kpi_present"] if stratify_col is not None else None

dev_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,                # 50 % of 40 %  →  20 % of total
    stratify=stratify_temp,
    random_state=42,
)

# ---- 4. save to disk ---------------------------------------------------------
train_df.to_csv("data/250730_train_set.csv", index=False)
dev_df.to_csv("data/250730_dev_set.csv",   index=False)
test_df.to_csv("data/250730_test_set.csv", index=False)

print(f"train: {len(train_df)} docs | dev: {len(dev_df)} | test: {len(test_df)}")

