# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 21:19:11 2021

@author: arpit
"""

import pandas as pd
import Quality
import mapper 
import missing_data


df = pd.read_csv("train.csv", index_col="Id")

#Handling the missing data
df = missing_data.deal_with_missing_values(df)

##Checking the Quality of the data
quality = Quality.quality_check(df, "SalePrice")

## Mapping the category data
df, mapp = mapper.map_ordinals(df)

#Convert date type data into Age
def age_Conversion(X):
    return X.apply(lambda x: 0 if x == 0 else (2020-x) )
df["GarageYrBlt"] = age_Conversion(df["GarageYrBlt"])
df["YearRemodAdd"] = age_Conversion(df["YearRemodAdd"])
df["YearBuilt"] = age_Conversion(df["YearBuilt"])
df["YrSold"] = age_Conversion(df["YrSold"])

#Removing Outliers
df = df.drop(df[(df.GrLivArea>4000) & (df.SalePrice<300000)].index)
df = df[df.GrLivArea * df.TotRmsAbvGrd < 45000]
df = df[df.GarageArea * df.GarageCars < 3700]
df = df[(df.FullBath + (df.HalfBath*0.5) + df.BsmtFullBath + (df.BsmtHalfBath*0.5))<5]
df = df.loc[~(df.SalePrice==392500.0)]
df = df.loc[~((df.SalePrice==275000.0) & (df.Neighborhood=='Crawfor'))]


#Feature Enigneering
df["Garage_Area_Car"] = df["GarageCars"] * df["GarageArea"]
df.drop(["GarageCars"], axis=1, inplace=True)

df['TotalBsmtSF_x_Bsm'] = df.TotalBsmtSF * df['1stFlrSF']
df.drop(["1stFlrSF"], axis=1, inplace=True)

TotalArea = ["GrLivArea", "TotalBsmtSF", "WoodDeckSF", "MasVnrArea", "GarageArea", "OpenPorchSF", "3SsnPorch", "ScreenPorch", "EnclosedPorch", "PoolArea" ]
df["TotalArea"] = df.GrLivArea + df.TotalBsmtSF + df.WoodDeckSF + df.MasVnrArea + df.GarageArea + df.OpenPorchSF + df["3SsnPorch"] + df.ScreenPorch + df.EnclosedPorch + df.PoolArea
df.drop(TotalArea, axis=1, inplace=True)

df['LotAreaMultSlope'] = df.LotArea * df.LandSlope
df.drop(["LotArea", "LandSlope"], axis=1, inplace=True)
df.drop([ "MiscVal"], axis=1, inplace=True)

quality2 = Quality.quality_check(df, "SalePrice")

#returns a list of cols where |skewd| > 1
#return a list of cols where |kurt| > 3
def get_skewd_kurts_cols(X):
    L1 = X[(X['skew'] > 1) | (X['skew'] < -1) ].index.tolist()
    L2 = X[(X['kurt'] > 3) | (X['kurt'] < -3) ].index.tolist()
    return L1,L2
skewd, kurts = get_skewd_kurts_cols(quality2)


## Handling the skewd data 
import numpy as np
def handle_skewd_data(L, X):
    for col in L:
        X[col] = np.log1p(X[col])
            
    return X

temp = df[mapp]
df = handle_skewd_data(skewd, df) ##Uncomment if you want to handle the skewd data
df[mapp] = temp

quality3 = Quality.quality_check(df, "SalePrice")

y = df[['SalePrice']]
df.drop(['SalePrice'], axis=1, inplace=True)
X = df

from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold 
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel

X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.15, random_state=0)

cols = X.columns
num_cols = X._get_numeric_data().columns
cat_cols = list(set(cols) - set(num_cols))



##############################FEATURE_SELECTION################################
import Encoder
X_train, X_test = Encoder.one_encoder(X_train, X_test, cat_cols)
features = SelectFromModel(GradientBoostingRegressor())
features.fit(X_train, y_train)
val = features.get_support()
fet = X_train.columns[val]

from sklearn.metrics import roc_auc_score, r2_score
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
sfs = SFS(RandomForestRegressor(n_estimators=10, n_jobs=4, random_state=10), 
           k_features=20, 
           forward=True, 
           floating=False, 
           verbose=2,
           scoring='r2',
           cv=2)

sfs = sfs.fit(np.array(X_train), y_train)


###############################################################################



Regressor = RandomForestRegressor(n_estimators=1000, random_state=10)

from sklearn.model_selection import cross_val_predict
model.fit(X_train[val], y_train)
y_pred_test=model.predict(X_test[val])


y_inv_test = np.expm1(y_test)
y_inv_pred = np.expm1(y_pred_test)

mean_absolute_error(y_inv_pred, y_inv_test)