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

def infer_gender(train, test):

    encoder = LabelEncoder()
    Y = encoder.fit_transform(train['gender'])

    results = perform_random_forest_proba(train, train['gender'], test, ['number_events', 'phone_brand'])
    df = pd.DataFrame(results, columns=encoder.classes_)

    df['device_id'] = test['device_id']
    df.to_csv('output/results_gender.csv', index=True,index_label='device_id')

    return df

def infer_age_group(train, test):
    
    encoder = LabelEncoder()
    Y = train["group"]
    Y = encoder.fit_transform(Y)

    results = perform_random_forest_proba(train, train['group'], test, ['number_events', 'phone_brand'])
    df = pd.DataFrame(results, columns=encoder.classes_)

    df['device_id'] = test['device_id']
    df.to_csv('output/results_group.csv', index=True,index_label='device_id')

    return df

def prepare_submission_sex_age(df, device_id):

    result = pd.DataFrame()
    result["device_id"] = df["device_id"]  
    
    result['F23-'] = df.apply(lambda row: (row['F']*row[0]), axis=1)
    result['F24-26'] = df.apply(lambda row: (row['F']*row[1]), axis=1)
    result['F27-28'] = df.apply(lambda row: (row['F']*row[2]), axis=1)
    result['F29-32'] = df.apply(lambda row: (row['F']*row[3]), axis=1)
    result['F33-42'] = df.apply(lambda row: (row['F']*row[4]), axis=1)
    result['F43+'] = df.apply(lambda row: (row['F']*row[5]), axis=1)
    result['M22-'] = df.apply(lambda row: (row['M']*row[0]), axis=1)
    result['M23-26'] = df.apply(lambda row: (row['M']*row[1]), axis=1)
    result['M27-28'] = df.apply(lambda row: (row['M']*row[2]), axis=1)
    result['M29-31'] = df.apply(lambda row: (row['M']*row[3]), axis=1)
    result['M32-38'] = df.apply(lambda row: (row['M']*row[4]), axis=1)
    result['M39+'] = df.apply(lambda row: (row['M']*row[5]), axis=1)

    result = result.set_index("device_id")

    return result

train = load_train(1)
#df = merge_installed_active_apps(df)
train.to_csv("output/train_FIXED.csv")

test = load_test(1)
test.to_csv("output/test_FIXED.csv", index=True, index_label='device_id')
test = test.reset_index()

#feature_selection(train, ['number_events', 'phone_brand', 'device_model', 'installed', 'active'], train['group'])
#results = perform_random_forest_proba(train, train['group'], test, ['number_events', 'phone_brand'])

df_gender = infer_gender(train, test)

#df_men = df[df['M'] >= 0.5]
#df_men.drop(['F'], axis=1, inplace=True)

#df_women = df[df['F'] > 0.5]
#df_women.drop(['M'], axis=1, inplace=True)

df_age = infer_age_group(train, test)

#df_men = pd.merge(df_men, df_age, how="left")
#df_women = pd.merge(df_women, df_age, how="left")
df_gender = pd.merge(df_gender, df_age, how="left")

df = prepare_submission_sex_age(df_gender, test["device_id"])

df.to_csv("output/results.csv")

print(df.head(10))

sys.exit()

