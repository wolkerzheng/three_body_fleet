# -*- coding: utf-8 -*-

from pandas import Series, DataFrame
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# user_shop_behavior
usb = pd.read_csv('./raw_data/train-ccf_first_round_user_shop_behavior.csv', header=0, low_memory=False)
# shop_info
si = pd.read_csv('./raw_data/train-ccf_first_round_shop_info.csv', header=0, low_memory=False)

usb.columns = ['uid', 'sid', 'time', 'jing', 'wei', 'wifi']
si.columns = ['sid', 'cid', 'jing', 'wei', 'price', 'mid']

# [1.0] add mall id for every record

sid_cid_mid = si[['sid', 'cid', 'mid']]
# extended_user_shop_behavior (mall_id added)
eusb = pd.merge(usb, sid_cid_mid, on='sid')

# eusb.to_csv('i:tianchi/eusb.csv', index=None)

# [2.0] generate subset, one subset for one mall, sorted by time
# extended_user_shop_behavior
# eusb = pd.read_csv('i:tianchi/eusb.csv', header=0, low_memory=False)

mall_group = eusb.groupby('mid')
mall_si = si.groupby('mid')

i = 0
for k, v in mall_group:
    i += 1
    sorted_v = v.sort_values(by=['time'])
    print(k)
    sorted_v.to_csv('./subset/' + k + '.csv', index=None)


#record the shop_infos in different mall
for k, v in mall_si:
	v.to_csv('./mall_shop/' + k + '.csv', index = None)





