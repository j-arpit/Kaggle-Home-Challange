# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 01:10:32 2021

@author: arpit
"""


from sklearn.preprocessing import OneHotEncoder as OHE
import pandas as pd

def one_encoder(X_train, X_valid, object_cols):
    encoder = OHE(handle_unknown='ignore', sparse=False)
    OH_cols_train = pd.DataFrame(encoder.fit_transform(X_train[object_cols]))
    OH_cols_valid = pd.DataFrame(encoder.transform(X_valid[object_cols]))
    OH_cols_train.index = X_train.index
    OH_cols_valid.index = X_valid.index
    num_X_train = X_train.drop(object_cols, axis=1)
    num_X_valid = X_valid.drop(object_cols, axis=1)
    OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
    OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)
    return OH_X_train, OH_X_valid
"""
Initially i was using ONEHOTENCODER for mapping, but there were 2 problem with that 
1) It is creating col with colname a numerical value which was causing problem in Backward Elimination, 
2) It is creating one extra dummy variable.
"""
def one_hot_encode(df):
    df.MSSubClass = df.MSSubClass.astype('str')
    df.MoSold = df.MoSold.astype('str')
    categorical_cols = df.select_dtypes(include=['object']).columns
    dummies = pd.get_dummies(df[categorical_cols], columns = categorical_cols).columns
    df = pd.get_dummies(df, columns = categorical_cols)

    print("Total Columns:",len(df.columns))
    print(df.info())
    
    return df, dummies
