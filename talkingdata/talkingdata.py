#!  /usr/bin/env python
# -*- coding: utf-8 -*- 

import pandas as pd
import numpy as np
import sys
import gc
from preprocess import * 
from Tools.estimators import *
from sklearn.preprocessing import LabelEncoder
from shapely.geometry import MultiPoint

df = load_train(0.0005)
df = merge_installed_active_apps(df)
df.to_csv("output/train_FIXED.csv")

df = load_test(0.0005)
df.to_csv("output/test_FIXED.csv", index=True, index_label='device_id')

encoder = LabelEncoder()
Y = train["group"]
Y = encoder.fit_transform(Y)

#feature_selection(train, ['number_events', 'phone_brand', 'device_model', 'installed', 'active'], train['group'])
#results = perform_random_forest_proba(train, train['group'], test, ['number_events', 'phone_brand'])
results = perform_gradient_boosting_proba(train, train['group'], test, ['number_events', 'phone_brand', 'installed'])
df = pd.DataFrame(results, columns=encoder.classes_)

df["device_id"] = test['device_id']
df = df.set_index("device_id")
df.to_csv('output/results.csv', index=True,index_label='device_id')

#print(df.head(5))



