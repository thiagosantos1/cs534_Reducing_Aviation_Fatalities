#!/usr/bin/python3

import pandas as pd
import numpy as np
import sys

for i in range(1, len(sys.argv)):
    data = pd.read_csv(sys.argv[i])
    head = sys.argv[i].rsplit('.', 1)[0]
    print(data.shape)
    uq = data[['crew', 'experiment', 'seat']].drop_duplicates()
    for ix, row in uq.iterrows():
        crew = row['crew']
        experiment = row['experiment']
        seat = row['seat']
        filename = '{}_{:02}_{}_{}.csv'.format(head, crew, experiment, seat)
        print(filename)
        f1 = data[data['crew'] == crew]
        f2 = f1[f1['experiment'] == experiment]
        f3 = f2[f2['seat'] == seat]
        f3.sort_values('time').to_csv(filename)

