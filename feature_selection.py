from pandas import Series, DataFrame
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from functions import *

processed_test_data = None
flag = True
wifi_encoder_dict = {}
data_type = {'jing': np.float64, 'wei': np.float64}

# get all files in subset, handle them independently
subsets = os.listdir("./subset")

# load test set
test_data = pd.read_csv('./raw_data/evaluation_public.csv', header=0, low_memory=False)
test_data.columns = ['rid', 'uid', 'mid', 'time', 'jing', 'wei', 'wifi']


#according to shops in different mall to build the KNN tree of each mall
leafsize = 2
topK = 10
# for loop, handle subset independently
for file_name in subsets:  # file_name = m_xxxx.csv
	file_path = "./subset/" + file_name
	current_mall_id = file_name.split(".")[0] #m_xxxx
	data = pd.read_csv(file_path, header=0, low_memory=False)
	
	#read the file in dir mall_shop
	mall_shop_path = "./mall_shop/" + file_name
	shop_data = pd.read_csv(mall_shop_path, header = 0, dtype = data_type, low_memory = False)
	shop_data.columns = ['sid', 'cid', 'jing', 'wei', 'price', 'mid']
	lat_lng = np.array(shop_data[['wei', 'jing']])
	radiansX = latlng2radians(lat_lng)
	tree = build_tree(radiansX, leafsize)

	# [1.0] encode wifi_names [for each mall, which means different mall have different encodings]
	wifi_dict = get_all_wifi(data)  # key=wifi_name, value=show up count

	# [1.1] before encoding wifi names, filter out some wifi below certain threshold
	fixed_wifi_dict = wifi_filter(wifi_dict)
	fixed_wifi_names = list(fixed_wifi_dict.keys())
	label_encoder = LabelEncoder()
	label_encoder.fit(fixed_wifi_names)

	# [2.0] handle wifi infos. two feature: strongest wifi and connected wifi
	wifi_infos = data.wifi
	# choose test data according to mall id
	current_test_data = test_data[test_data['mid'] == current_mall_id]
	current_test_data['s_wifi'] = current_test_data.wifi.apply(strongest_wifi)
	current_test_data['c_wifi'] = current_test_data.wifi.apply(connected_wifi)
	current_test_data['is_conn'] = current_test_data.wifi.apply(is_connnected)
	

	data['s_wifi'] = data.wifi.apply(strongest_wifi)
	data['c_wifi'] = data.wifi.apply(connected_wifi)

	#query the KNN tree
	testX = np.array(current_test_data[['wei', 'jing']])
	radians_testX = latlng2radians(testX)
	index = tree.query(radians_testX, k = topK, return_distance = False)
	topK_sid = np.array(shop_data['sid'])[index]
	new_topK_sid = []
	for topKx in topK_sid:
		new_topK_sid.append(";".join(topKx))
	current_test_data['topK_shopid'] = new_topK_sid

	topK_cid = np.array(shop_data['cid'])[index]
	new_topK_cid = []
	for topKx in topK_cid:
		new_topK_cid.append(";".join(topKx))
	current_test_data['topK_cid'] = new_topK_cid

	# [3.0] use previously fitted label encoder to encode two newly extracted wifi feature
	data.s_wifi = label_encoder.transform(data.s_wifi)
	data.c_wifi = label_encoder.transform(data.c_wifi)

	current_test_data.s_wifi = label_encoder.transform(current_test_data.s_wifi)
	current_test_data.c_wifi = label_encoder.transform(current_test_data.c_wifi)

	# [4.0] connected to wifi or not
	data['is_conn'] = data.wifi.apply(is_connnected)
	current_test_data['is_conn'] = current_test_data.wifi.apply(is_connnected)

	# [5.0] time feature
	data['time_block'] = data.time.apply(get_time_block)
	current_test_data['time_block'] = current_test_data.time.apply(get_time_block)

	feature_path = "./feature/" + current_mall_id
	data.to_csv(feature_path, index=None)
	
	# concat processed test data
	if flag:
		processed_test_data = current_test_data
		flag = False
	else:
		processed_test_data = pd.concat([processed_test_data, current_test_data], axis=0)

	# shop id encoder should be stored here
	# after prediction, re-inverse is needed

# finally store processed test data
processed_test_data.to_csv("./test/test.csv", index=None)

