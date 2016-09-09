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

    results = perform_gaussianNB_proba(train, train['gender'], test, ['number_events', 'phone_brand'])
    df = pd.DataFrame(results, columns=encoder.classes_)

    df['device_id'] = test['device_id']
    df.to_csv('output/results_gender.csv', index=True,index_label='device_id')

    return df

def infer_age_group(train, test):
    
    encoder = LabelEncoder()
    Y = train["group"]
    Y = encoder.fit_transform(Y)

    results = perform_gaussianNB_proba(train, train['group'], test, ['number_events', 'phone_brand'])
    df = pd.DataFrame(results, columns=encoder.classes_)

    df['device_id'] = test['device_id']
    df.to_csv('output/results_group.csv', index=True,index_label='device_id')

    return df

def infer_age_group_male(train, test):
    
    train_male = train[train['gender'] == 'M']
    train_male.loc[train_male["gender"] == 'M', "gender"] = 0
    print("(male) Training on:")
    print(train_male.head(10))

    encoder = LabelEncoder()
    Y = encoder.fit_transform(train_male["group"])
   
    test_male = test
    test_male['gender'] = 0
    
    results = perform_gaussianNB_proba(train_male, train_male['group'], test, ['number_events', 'phone_brand', 'gender'])
    df = pd.DataFrame(results, columns=encoder.classes_)
    
    df['device_id'] = test_male['device_id']
    df.to_csv('output/results_group_male.csv', index=True,index_label='device_id')

    return df

def infer_age_group_female(train, test):
    
    train_female = train[train['gender'] == 'F']
    train_female.loc[train_female["gender"] == 'F', "gender"] = 1
    print("(female) Training on:")
    print(train_female.head(10))
    
    encoder = LabelEncoder()
    Y = encoder.fit_transform(train_female["group"])
    
    test_female = test
    test_female['gender'] = 1 
    
    results = perform_gaussianNB_proba(train_female, train_female['group'], test, ['number_events', 'phone_brand', 'gender'])
    df = pd.DataFrame(results, columns=encoder.classes_)
    
    df['device_id'] = test_female['device_id']
    df.to_csv('output/results_group_female.csv', index=True,index_label='device_id')
    
    return df

def prepare_submission_gender_age(df, device_id):

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

def prepare_submission_gender_age_separated(df, device_id):

    result = pd.DataFrame()
    result["device_id"] = df["device_id"]  
    
    result['F23-'] =   df.apply(lambda row: (row['F']*row['F23-']), axis=1)
    result['F24-26'] = df.apply(lambda row: (row['F']*row['F24-26']), axis=1)
    result['F27-28'] = df.apply(lambda row: (row['F']*row['F27-28']), axis=1)
    result['F29-32'] = df.apply(lambda row: (row['F']*row['F29-32']), axis=1)
    result['F33-42'] = df.apply(lambda row: (row['F']*row['F33-42']), axis=1)
    result['F43+'] =   df.apply(lambda row: (row['F']*row['F43+']), axis=1)
    result['M22-'] =   df.apply(lambda row: (row['M']*row['M22-']), axis=1)
    result['M23-26'] = df.apply(lambda row: (row['M']*row['M23-26']), axis=1)
    result['M27-28'] = df.apply(lambda row: (row['M']*row['M27-28']), axis=1)
    result['M29-31'] = df.apply(lambda row: (row['M']*row['M29-31']), axis=1)
    result['M32-38'] = df.apply(lambda row: (row['M']*row['M32-38']), axis=1)
    result['M39+'] =   df.apply(lambda row: (row['M']*row['M39+']), axis=1)

    result = result.set_index("device_id")

    return result
   
train = load_train(1)
train.to_csv("output/train_merged.csv")

test = load_test(1)
test.to_csv("output/test_merged.csv", index=True, index_label='device_id')
test = test.reset_index()

# get feature relevance
#feature_selection(train, ['number_events', 'phone_brand', 'device_model', 'installed', 'active'], train['group'])

# perform regular rf/gb/etc
#encoder = LabelEncoder()
#Y = encoder.fit_transform(train['group'])
#results = perform_gaussianNB_proba(train, train['group'], test, ['number_events', 'phone_brand'])
#df = pd.DataFrame(results, columns=encoder.classes_)
#df['device_id'] = test['device_id']
#df = df.set_index("device_id")
#df.to_csv('output/results.csv')

# perform gender probability x age probability separately for each gender
#df_gender = infer_gender(train, test)

#df_age_male = infer_age_group_male(train, test)
#df_age_female = infer_age_group_female(train, test)

#df = pd.merge(df_gender, df_age_male, how="left")
#df = pd.merge(df, df_age_female, how="left")

#df = prepare_submission_gender_age_separated(df, df['device_id'])
#df.to_csv("output/results.csv")

# perform gender probability x age probability altogether
df_gender = infer_gender(train, test)
df_age = infer_age_group(train, test)

df_gender = pd.merge(df_gender, df_age, how="left")
print(df_gender.head(5))
df = prepare_submission_gender_age(df_gender, test["device_id"])
df.to_csv("output/results.csv")

