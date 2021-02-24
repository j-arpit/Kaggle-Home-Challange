#!/usr/bin/env python
# coding: utf-8

# # Importing libraries and Dataset

# In[1]:


import numpy as np
import pandas as pd
import statsmodels.api  as sm # For p value calculations

#import external functions 
import Quality # To Check the quality of the data
import mapper # For Label encoding
import missing_data # To Handle Missing data
import Encoder # For OneHotEncoding


# In[2]:


#sklearn libraries
from sklearn.model_selection import train_test_split


# In[3]:


# Import Data set
test = pd.read_csv("test.csv", index_col="Id")
df = pd.read_csv("train.csv", index_col="Id")

# Lets CheckOut the Total No of Features
print("Length of Test = {}".format(len(test.columns)))
print("Length of Train =  {}".format(len(df.columns)))

print(df.columns)
print("\n")
print(df._get_numeric_data().columns)


# # Data Processing

# In[4]:


#Handling the missing data
df = missing_data.deal_with_missing_values(df)
test = missing_data.deal_with_missing_values(test)

##Checking the Quality of the data
quality = Quality.quality_check(df, "SalePrice")
print(quality)

## Label Encoding
df, mapp = mapper.mapping(df)
test, mapp = mapper.mapping(test)
print(mapp)


# In[5]:


##ONE Hot encoding using dummy
df, dummies = Encoder.one_hot_encode(df)
test, dummies_test = Encoder.one_hot_encode(test)

miss = missing_data.find_col_with_missing_data(df)
print(miss) # just for assurance


# #### Note that the size of test and df is different.  The reason behind this is that some categorical features in testset doesnot contains all the unique values. So here i am creating all those cols with 0 value.

# In[6]:


dummies = dummies.values
dummies_test = dummies_test.values
M = list(set(dummies) - set(dummies_test))
for i in M:
    test[i] = 0
print("Length of Test = {}".format(len(test.columns)))
print("Length of Train =  {}".format(len(df.columns)))


# # CONVERTING YEAR DATA INTO AGE

# In[7]:


def age_Conversion(X):
    return X.apply(lambda x: 0 if x == 0 else (2020-x) )
df["GarageYrBlt"] = age_Conversion(df["GarageYrBlt"])
df["YearRemodAdd"] = age_Conversion(df["YearRemodAdd"])
df["YearBuilt"] = age_Conversion(df["YearBuilt"])
df["YrSold"] = age_Conversion(df["YrSold"])


# In[8]:


test["GarageYrBlt"] = age_Conversion(test["GarageYrBlt"])
test["YearRemodAdd"] = age_Conversion(test["YearRemodAdd"])
test["YearBuilt"] = age_Conversion(test["YearBuilt"])
test["YrSold"] = age_Conversion(test["YrSold"])


# # FEATURE ENGINEERING

# In[9]:


def add_fet(X):
    X["Remod"] = 2
    X.loc[(X.YearBuilt==X.YearRemodAdd), ['Remod']] = 0
    X.loc[(X.YearBuilt!=X.YearRemodAdd), ['Remod']] = 1
    X.Remod
    X["Age"] = X.YearRemodAdd - X.YrSold # sice I convert both to age
    X["IsNew"] = 2
    X.loc[(X.YearBuilt==X.YrSold), ['IsNew']] = 1
    X.loc[(X.YearBuilt!=X.YrSold), ['IsNew']] = 0
    return X
df = add_fet(df)
test = add_fet(test)


# In[10]:


def fet_Engineering(X):
    X["Garage_Area_Car"] = X["GarageCars"] * X["GarageArea"]

    X['TotalBsmtSF_x_Bsm'] = X.TotalBsmtSF * X['1stFlrSF']

    #TotalArea = ["GrLivArea", "TotalBsmtSF", "WoodDeckSF", "MasVnrArea", "GarageArea", "OpenPorchSF", "3SsnPorch", "ScreenPorch", "EnclosedPorch", "PoolArea" ]
    X["TotalArea"] = X.GrLivArea + X.TotalBsmtSF + X.WoodDeckSF + X.MasVnrArea + X.GarageArea + X.OpenPorchSF + X["3SsnPorch"] + X.ScreenPorch + X.EnclosedPorch + X.PoolArea
    #df.drop(TotalArea, axis=1, inplace=True)

    X['LotAreaMultSlope'] = X.LotArea * X.LandSlope
    #df.drop([ "MiscVal", "GarageCars", "1stFlrSF", "LotArea", "LandSlope", "Utilities"], axis=1, inplace=True)
    
    return X

df = fet_Engineering(df)
test = fet_Engineering(test)


# # Handling Skewd data

# What is skewd data?
# A data is called as skewed when curve appears distorted or skewed either to the left or to the right, in a statistical distribution. In a normal distribution, the graph appears symmetry meaning that there are about as many data values on the left side of the median as on the right side.

# How does it affects the model?
# So in skewed data, the tail region may act as an outlier for the statistical model and we know that outliers adversely affect the model’s performance especially regression-based models.

# In[11]:


quality2 = Quality.quality_check(df, "SalePrice")
mapp.remove("Utilities") ## this col is useless 

#returns a list of cols where |skewd| > 1
#return a list of cols where |kurt| > 3
def get_skewd_kurts_cols(X):
    L1 = X[(X['skew'] > 1) | (X['skew'] < -1) ].index.tolist()
    L2 = X[(X['kurt'] > 3) | (X['kurt'] < -3) ].index.tolist()
    return L1,L2
skewd, _ = get_skewd_kurts_cols(quality2)


## Handling the skewd data 
def handle_skewd_data(L, X):
    for col in L:
        try: 
            X[col] = np.log1p(X[col])
        except:
            pass
            
    return X

temp_df = df[mapp]
temp_test = test[mapp]
df = handle_skewd_data(skewd, df)
test = handle_skewd_data(skewd, test)##Uncomment if you want to handle the skewd data
df[mapp] = temp_df
test[mapp] = temp_test

quality3 = Quality.quality_check(df, "SalePrice")
print(quality3)


# # Split the data

# In[12]:


y = df[['SalePrice']]
df.drop(['SalePrice'], axis=1, inplace=True)
X = df

X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.15, random_state=0)

cols = X.columns
num_cols = X._get_numeric_data().columns
cat_cols = list(set(cols) - set(num_cols))


# In[13]:


type(X_train)
print(X_train._get_numeric_data().columns)
X_train.shape


# ### Feature Elimination using P Value

# What is p value?
# The p-value is used as an alternative to rejection points to provide the smallest level of significance at which the null hypothesis would be rejected. A smaller p-value means that there is stronger evidence in favor of the alternative hypothesis.

# In[14]:


def Backward_Elimination(y, X, sl):
    cols = X.columns.values
    ini = len(cols)
    col_vars = X.shape[1]
    for i in range (0, col_vars):
        Regressor = sm.OLS(y, X).fit()
        maxVar = max(Regressor.pvalues)
        if maxVar > sl:
            for j in range(0, col_vars-i):
                if (Regressor.pvalues[j].astype(float) == maxVar):
                    cols = np.delete(cols, j)
                    X = X.loc[:, cols]
        
    print('\nSelect {:d} features from {:d} by best p-values.'.format(len(cols), ini))
    print(Regressor.summary())
    return cols


# In[15]:


colum = Backward_Elimination(y_train, X_train, 0.051)
len(colum)


# ###  Converting DataFrames into Numpy arrays and saving it in .npy format

# In[16]:


X_train =X_train[colum].values
y_train = y_train.values
X_val =X_test[colum].values
y_val = y_test.values
test = test[colum].values


# In[17]:


np.save('X_train.npy', X_train)
np.save('y_train.npy', y_train)
np.save('X_val.npy', X_val)
np.save('y_val.npy', y_val)
np.save('test.npy', test)

