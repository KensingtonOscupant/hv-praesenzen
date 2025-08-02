# script to get from 250707_test_set.csv to 250707_test_set.csv

import pandas as pd

# Loaded variable 'df' from URI: /Users/felixringe/Documents/Projects/hv-praesenzen/data/250707_test_set.csv
import pandas as pd
df = pd.read_csv(r'/Users/felixringe/Documents/Projects/hv-praesenzen/data/250707_test_set.csv', sep=";")
df.rename(columns={'Presence_enhanced': 'label_value'}, inplace=True)
df.rename(columns={'Presence_predicted': 'label_predicted'}, inplace=True)
df.rename(columns={'ID_Key_original': 'id_key'}, inplace=True)
df.rename(columns={'Year_original': 'year'}, inplace=True)
df['label_present'] = True
df.drop(columns=['price'], inplace=True)
df = df[['id_key', 'year', 'label_present', 'label_value', 'label_predicted', 'correct', 'error', 'explanation', 'comment', 'file_path']]

df.to_csv("test_set.csv", index=False)