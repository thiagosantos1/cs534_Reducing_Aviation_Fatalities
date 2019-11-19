#!/usr/bin/python3

import pandas as pd
import numpy as np

data = pd.read_csv('train.csv')
print(data.shape)
uq = data[['crew','experiment']].drop_duplicates()
print(uq)
for ix,row in uq.iterrows():
    crew=row['crew']
    experiment=row['experiment']
    f1=data[data['crew']==crew]
    f2=f1[f1['experiment']==experiment]
    f2.sort_values('time').to_csv('train_{:02}_{}.csv'.format(crew, experiment))

