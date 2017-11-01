from pandas import Series, DataFrame
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# load test set
res = pd.read_csv('./test/result.csv', header=0, low_memory=False)
res.columns = ['row_id', 'shop_id']

origin = pd.read_csv('./raw_data/evaluation_public.csv', low_memory=False)

part_origin = origin[['row_id']]

res = pd.merge(part_origin, res, on='row_id')

res.to_csv("./test/result", index=None)





