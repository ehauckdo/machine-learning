#!  /usr/bin/env python
# -*- coding: utf-8 -*- 

import pandas as pd
import numpy as np
import sys
from preprocess import translate_phone_brands
import gc
from talkingdata import *
from Tools.estimators import *
from sklearn.preprocessing import LabelEncoder
from shapely.geometry import MultiPoint

if False:
    df = generate_subset()

    devices = df['device_id'].value_counts().keys()
    events = []

    print(devices)
    for device in devices:
        device_ocurrences = df.loc[df['device_id'] == device]
        number_events = len(device_ocurrences['event_id'].value_counts())
        events.append(number_events)

    new_df = pd.DataFrame()
    new_df['device_id'] = devices
    new_df['number_events'] = events
    print(new_df.head(5))
    new_df.to_csv("output/number_events_for_device.csv", index=False)
    #new_df.to_csv("teste.csv", index=False)

#=== merges centroid ====

df = pd.read_csv("output/train_subset6.csv", dtype={'device_id':np.str})
#df = pd.read_csv("input/gender_age_train.csv", dtype={'device_id':np.str})
merge_device_centroid(df)
sys.exit()
df['lat_long'] = df[['latitude', 'longitude']].apply(tuple, axis=1)
#df.to_csv("output/train_subset7.csv", index=False)
#df = df.dropna(subset=['lat_long'])
df = df.drop_duplicates(subset=df.columns[15])
df = df[5:]
df = df['lat_long'].tolist()
#print(df)
points = MultiPoint(df)
print points.centroid #True centroid, not necessarily an existing point
print points.representative_point() #A represenative point, not centroid,
                                    #that is guarnateed to be with the geometry
sys.exit()

train = pd.read_csv("output/train_NE_PB.csv", dtype={'device_id':np.str})
test = pd.read_csv("output/test_NE_PB.csv", dtype={'device_id':np.str})

events_small = pd.read_csv("output/events_small.csv", dtype={'device_id':np.str, "counts":np.str})
events_small = events_small[['device_id', 'installed', 'active']].drop_duplicates('device_id', keep='first')

train = pd.merge(train, events_small, how='left', on='device_id')
train.fillna(-1, inplace=True)
test = test.drop_duplicates('device_id', keep='first')
test = pd.merge(test, events_small, how='left', on='device_id', suffixes=['','_'])
test.fillna(-1, inplace=True)

encoder = LabelEncoder()
train["phone_brand"] = encoder.fit_transform(train['phone_brand'])
train["device_model"] = encoder.fit_transform(train['device_model'])
test["phone_brand"] = encoder.fit_transform(test['phone_brand'])
test["device_model"] = encoder.fit_transform(test['device_model'])

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



