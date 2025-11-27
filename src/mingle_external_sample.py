#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: LL
# @File：mingle_external_sample.py
"""
 In-silico mingle samples from external literature data
"""
import pandas as pd

file_path = '../data/ext_val_data_literature.CSV'

df = pd.read_csv(file_path)

pairs = [
    (1, 4), (1, 12), (4, 12),
    (2, 3), (2, 6), (2, 10), (2, 11), (2, 14),
    (3, 6), (3, 10), (3, 11), (3, 14),
    (6, 10), (6, 11), (6, 14),
    (10, 11), (10, 14), (11, 14),
    (13, 15)
]

df_indexed = df.set_index('No.')
new_rows = []
next_no = df['No.'].max() + 1
for a, b in pairs:
    if a not in df_indexed.index or b not in df_indexed.index:
        continue
    row_a = df_indexed.loc[a]
    row_b = df_indexed.loc[b]

    feature_cols = df.columns[1:-1]
    source_col = df.columns[-1]  # 'Source'
    combined_features = row_a[feature_cols].combine(row_b[feature_cols], func=max)
    combined_source = f"{row_a[source_col]}"
    new_row = {'No.': next_no}
    new_row.update(combined_features.to_dict())
    new_row[source_col] = combined_source

    new_rows.append(new_row)
    next_no += 1


new_df = pd.DataFrame(new_rows)

final_df = pd.concat([df, new_df], ignore_index=True)

output_path = '../data/ext_val_data_literature_mingled.CSV'
final_df.to_csv(output_path, index=False)

print(f"Mingled data saved to {output_path}")