#!  /usr/bin/env python

import pandas as pd
import numpy as np
import sys
import gc
from Tools.estimators import *
from sklearn.preprocessing import LabelEncoder
from sklearn import cross_validation

def encode_features(df, features):
    encoder = LabelEncoder()
    for feature in features:
        df[feature] = encoder.fit_transform(df[feature])
    return df

def drop_useless_features(df):
    df.drop("PoolQC", axis=1, inplace=True)
    df.drop("Fence", axis=1, inplace=True)
    df.drop("MiscFeature", axis=1, inplace=True)
    df.drop("Alley", axis = 1, inplace=True)
    return df

# For these features, the selected values are predominant in each column
# so we can opt to turn them into 0/1 instead of creating a bunch of dummy vars
def norm_features(df):
    df.MSZoning = (df.MSZoning == "RL").astype(int)
    df.LotShape = (df.LotShape == "Reg").astype(int) 
    df.LandContour = (df.LandContour == "Lvl").astype(int) 
    df.LotConfig = (df.LotConfig == "Inside").astype(int) 
    df.LandSlope = (df.LandSlope == "Gtl").astype(int) 
    df.Condition1 = (df.Condition1 == "Norm").astype(int) 
    df.Condition2 = (df.Condition2 == "Norm").astype(int) 
    df.BldgType = (df.BldgType == "1Fam").astype(int) 
    df.RoofStyle = (df.RoofStyle == "Gable").astype(int) 
    df.RoofMatl = (df.RoofMatl == "CompShg").astype(int) 
    df.ExterCond = (df.ExterCond == "TA").astype(int)
    df.BsmtCond = (df.BsmtCond == "TA").astype(int)
    df.BsmtFinType2 = (df.BsmtFinType2 == "Unf").astype(int)
    df.Heating = (df.Heating == "GasA").astype(int)
    df.Electrical = (df.Electrical == "SBrkr").astype(int)
    df.Functional = (df.Functional == "Typ").astype(int)
    df.GarageQual = (df.GarageQual == "TA").astype(int)
    df.GarageCond = (df.GarageCond == "TA").astype(int)
    df.PavedDrive = (df.PavedDrive == "Y").astype(int)
    df.SaleType = (df.SaleType == "WD").astype(int)
    df.SaleCondition = (df.SaleCondition == "Normal").astype(int)
    return df

def create_dummy_variables(df):
    #all the quantitave variables are collect with the describe function
    quant_variable = df.describe().columns.values
    column = df.columns.values
    df = pd.get_dummies(df)
    return df
    
    print("DUMMY:")
    print(dummy_variable.head(5))
    dummy_variable.to_csv("output/train_fixed.csv", dummy_na=True)

    for i in column:
        if i not in quant_variable:
            #we are with qualitative variable
            df[i].fillna("no_present", inplace=True)
            dummy_variable = pd.get_dummies(df[i], prefix=i)
            print("DUMMY:")
            print(dummy_variable.head(5))
            print("COLUMN: ", i)
            print(dummy_variable.info()) 
            for dummy in dummy_variable:
                #for value in dummy_variable[dummy]:
                #    print(value) 
                #df.loc[dummy] = dummy_variable[dummy]
                df[dummy] = dummy_variable[dummy]
            #df = df.join(dummy_variable)
            df.drop(i, axis=1, inplace=True)
            #df.reindex(columns = dummy_variable.columns)
            #print(test.info())
    df.to_csv("output/train_fixed2.csv")
    sys.exit()
    return df

def replace_NAs(df):
    df = df.apply(lambda x: x.fillna(x.mean()),axis=0)
    return df

train = pd.read_csv("input/train.csv", index_col=0)
test = pd.read_csv("input/test.csv", index_col=0)
#print(train.shape, test.shape)

# let's combine train and test to make manipulations easier
all_df = pd.concat((train, test), axis=0)
all_df.to_csv("output/all_df.csv", index=False)
all_df = drop_useless_features(all_df)
all_df = create_dummy_variables(all_df)
all_df = replace_NAs(all_df)

# separate train and test back
train = all_df.loc[train.index]
test = all_df.loc[test.index]

train.to_csv("output/train_df.csv", index=False)
test.to_csv("output/test_df.csv", index=False)
#print(train.shape, test.shape)

predictors = list(train.columns.values)
predictors.remove("SalePrice")
for predictor in predictors:
    print(predictor)

top_features = ["OverallQual", "GrLivArea", "TotalBsmtSF", "2ndFlrSF", "BsmtFinSF1", "LotArea", "LotFrontage", "GarageCars", "1stFlrSF", "FullBath", "TotRmsAbvGrd"]
#train = encode_features(train, predictors)
#test = encode_features(test, predictors)

#feature_selection(all_df, predictors, all_df["SalePrice"])
#grid_search(RandomForestRegressor(), df, predictors, 'SalePrice')

results = perform_gradient_boosting_regressor(train, train['SalePrice'], test, predictors)

submission =  pd.DataFrame()
submission["Id"] = test.index
submission["SalePrice"] = results

submission.to_csv("output/kaggle.csv", index=False)
print(submission.head(5))
