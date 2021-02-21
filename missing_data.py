# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 16:40:51 2021

@author: arpit
"""
## returns the List of the cols with missing data
def find_col_with_missing_data(X_full):
    return [col for col in X_full.columns if X_full[col].isnull().sum()>0]

#Handling the missing data
def deal_with_missing_values(X):
    #List of NaN including columns where NaN's mean none.
    none_cols = [
        'Alley', 'PoolQC', 'MiscFeature', 'Fence', 'FireplaceQu', 'GarageType',
        'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtQual', 'BsmtCond',
        'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'MasVnrType'
    ]
    #List of NaN including columns where NaN's mean 0.
    zero_cols = [
        'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath',
        'BsmtHalfBath', 'GarageYrBlt', 'GarageArea', 'GarageCars', 'MasVnrArea'
    ]
    
    #List of NaN including columns where NaN's actually missing gonna replaced 
    #with mode.
    freq_cols = [
        'Electrical', 'Exterior1st', 'Exterior2nd', 'Functional', 'KitchenQual',
        'SaleType', 'Utilities'
    ]
    for col in none_cols:
        X[col].fillna(value='none', inplace=True)
    for col in zero_cols:
        X[col].fillna(value=0, inplace=True)
    for col in freq_cols:
        X[col].fillna(value = X[col].mode()[0], inplace=True)
    # Now only one left LotFrontage
    X['LotFrontage'].fillna(value = X['LotFrontage'].mean(), inplace=True)
    return X