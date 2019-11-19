#!/usr/bin/python3

import pandas as pd
import numpy as np
import sys

for i in range(1, len(sys.argv)):
    data = pd.read_csv(sys.argv[i])
    head = sys.argv[i].rsplit('.', 1)[0]
    print(data.shape)
    uq = data[['crew', 'experiment']].drop_duplicates()
    for ix, row in uq.iterrows():
        crew = row['crew']
        experiment = row['experiment']
        filename = '{}_{:02}_{}.csv'.format(head, crew, experiment)
        print(filename)
        f1 = data[data['crew'] == crew]
        f2 = f1[f1['experiment'] == experiment]
        f2.sort_values('time').to_csv(filename)

