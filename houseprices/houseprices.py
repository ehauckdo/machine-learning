#!  /usr/bin/env python

import pandas as pd
import numpy as np
import sys
import gc
from Tools.estimators import *
from sklearn.preprocessing import LabelEncoder

def encode_features(df, features):
    encoder = LabelEncoder()
    for feature in features:
        df[feature] = encoder.fit_transform(df[feature])
    return df

df = pd.read_csv("input/train.csv", keep_default_na=False)
print(df.head(5))
test = pd.read_csv("input/test.csv", keep_default_na=False)


predictors = list(df.columns.values)
predictors.remove("Id")
predictors.remove("SalePrice")
#print(predictors)

top_features = ["OverallQual", "YearBuilt", "MasVnrArea", "ExterQual", "BsmtQual", "TotalBsmtSF", "GrLivArea", "FullBath", "KitchenQual", "GarageCars", "GarageArea"]
df = encode_features(df, top_features)
test = encode_features(test, top_features)

#feature_selection(df, predictors, df["SalePrice"])

results = perform_random_forest_regressor(df, df['SalePrice'], test, top_features)

submission =  pd.DataFrame()
submission["Id"] = test["Id"]
submission["SalePrice"] = results

submission.to_csv("output/kaggle.csv", index=False)
print(submission.head(5))
