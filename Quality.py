# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 16:45:13 2021

@author: arpit
"""
import pandas as pd
##Checking the Quality of the data
def quality_check(X, target):
    X = X[list(X._get_numeric_data().columns)]
    corr = X.corr()[target]
    corr = corr.sort_values(ascending=False)
    skew = X.skew()
    kurts = X.kurt()
    quality = pd.concat([corr, skew, kurts], axis=1)
    quality.rename(columns={0:'skew'}, inplace=True)
    quality.rename(columns={1:'kurt'}, inplace=True)
    return quality

