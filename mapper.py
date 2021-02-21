# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 20:33:29 2021

@author: arpit
"""


def mapping(data):
    
    
    LandSlope = {}
    LandSlope['Gtl'] = 3 
    LandSlope['Mod'] = 2 
    LandSlope['Sev'] = 1 

    data.LandSlope = data.LandSlope.map(LandSlope)
        
    
    ExterQual = {}
    ExterQual['Ex'] = 5 
    ExterQual['Gd'] = 4 
    ExterQual['TA'] = 3 
    ExterQual['Fa'] = 2 
    ExterQual['Po'] = 1 
    ExterQual['none'] = 0 

    data.ExterQual = data.ExterQual.map(ExterQual)

    
    data.ExterCond = data.ExterCond.map(ExterQual)

    
    data.HeatingQC = data.HeatingQC.map(ExterQual)

    
    data.KitchenQual = data.KitchenQual.map(ExterQual)

    
    data.FireplaceQu = data.FireplaceQu.map(ExterQual)

    
    data.GarageCond = data.GarageCond.map(ExterQual)

    PavedDrive = {}
    PavedDrive['Y'] = 3 
    PavedDrive['P'] = 2 
    PavedDrive['N'] = 1 

    data.PavedDrive = data.PavedDrive.map(PavedDrive)

    
    LotShape = {}
    LotShape['Reg'] = 4 
    LotShape['IR1'] = 3 
    LotShape['IR2'] = 2 
    LotShape['IR3'] = 1 

    data.LotShape = data.LotShape.map(LotShape)

    
    BsmtQual = {}
    BsmtQual['Ex'] = 5 
    BsmtQual['Gd'] = 4 
    BsmtQual['TA'] = 3 
    BsmtQual['Fa'] = 2 
    BsmtQual['Po'] = 1 
    BsmtQual['none'] =  0

    data.BsmtQual = data.BsmtQual.map(BsmtQual)

    
    data.BsmtCond = data.BsmtCond.map(BsmtQual)

    
    data.GarageQual = data.GarageQual.map(BsmtQual)

    
    data.PoolQC = data.PoolQC.map(BsmtQual)
    
    
    BsmtExposure = {}
    BsmtExposure['Gd'] = 4 
    BsmtExposure['Av'] = 3 
    BsmtExposure['Mn'] = 2 
    BsmtExposure['No'] = 1 
    BsmtExposure['none'] = 0 

    data.BsmtExposure = data.BsmtExposure.map(BsmtExposure)

    
    BsmtFinType1 = {}
    BsmtFinType1['GLQ'] = 6 
    BsmtFinType1['ALQ'] = 5 
    BsmtFinType1['BLQ'] = 4 
    BsmtFinType1['Rec'] = 3 
    BsmtFinType1['LwQ'] = 2 
    BsmtFinType1['Unf'] = 1 
    BsmtFinType1['none'] = 0 

    data.BsmtFinType1 = data.BsmtFinType1.map(BsmtFinType1)

    
    data.BsmtFinType2 = data.BsmtFinType2.map(BsmtFinType1)

    
    CentralAir = {}
    CentralAir['N'] = 0
    CentralAir['Y'] = 1

    data.CentralAir = data.CentralAir.map(CentralAir)

    
    GarageFinish = {}
    GarageFinish['Fin'] = 3 
    GarageFinish['RFn'] = 2 
    GarageFinish['Unf'] = 1 
    GarageFinish['none'] = 0 
    
    data.GarageFinish = data.GarageFinish.map(GarageFinish)
    
    
    Functional = {}
    Functional['Typ'] = 7   
    Functional['Min1'] = 6  
    Functional['Min2'] = 5  
    Functional['Mod'] = 4   
    Functional['Maj1'] = 3  
    Functional['Maj2'] = 2  
    Functional['Sev'] = 1   
    Functional['Sal'] = 0   

    data.Functional = data.Functional.map(Functional)
    
    
    Street = {}
    Street['Grvl'] = 0 
    Street['Pave'] = 1 

    data.Street = data.Street.map(Street)

    
    
    Fence = {}
    Fence['GdPrv'] = 5 
    Fence['MnPrv'] = 4 
    Fence['GdWo'] = 3 
    Fence['MnWw'] = 2 
    Fence['none'] = 1 

    data.Fence = data.Fence.map(Fence)
    
    
    
    Utilities = {}
    Utilities["AllPub"] = 5 
    Utilities["NoSewr"] = 4 
    Utilities["NoSeWa"] = 3 
    Utilities["ELO"] = 1 
    
    data.Utilities = data.Utilities.map(Utilities)
    
    keys = ["ExterQual", "ExterQual", "HeatingQC", "KitchenQual", "FireplaceQu", "GarageCond", "PavedDrive", "LotShape", "BsmtQual", "BsmtCond", "GarageQual", "PoolQC", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "CentralAir", "GarageFinish", "Functional", "Street", "Fence", "Utilities" ]
    print("Mapping done for {}".format(keys))
            
    return data, keys
