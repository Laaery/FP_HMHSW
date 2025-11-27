#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: LL
# @File：transpose_phase_vector.py
import pandas as pd


def count_and_transpose(input_file, output_file):
    df = pd.read_csv(input_file)
    source_col = df['Source']
    phase_cols = df.drop(['Source', 'Index'], axis=1).columns
    result_data = []
    for phase in phase_cols:
        phase_counts = df[df[phase] != 0].groupby('Source')[phase].count()
        row = {'Phase': phase}
        for source in source_col.unique():
            row[source] = phase_counts.get(source, 0)
        result_data.append(row)
    result_df = pd.DataFrame(result_data)
    result_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

count_and_transpose('../data/phase_vector.csv', '../data/phase_vector_transposed.csv')
