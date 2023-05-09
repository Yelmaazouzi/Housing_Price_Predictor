import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import Ridge
from sklearn import linear_model as lm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import make_scorer
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
import math

# =============================================================================
def main():
    # Read the original data files
    trainDF = pd.read_csv("data/train.csv")
    testDF = pd.read_csv("data/test.csv")
    
    #demonstrateHelpers(trainDF)
    
    #mean = trainDF.loc[:,'SalePrice'].mean()
    #std = trainDF.loc[:,'SalePrice'].std()
    
    trainInput, testInput, trainOutput, testIDs, predictors = transformData(trainDF, testDF)
    
    doExperiment(trainInput, trainOutput, predictors)
    doExperiment2(trainInput, trainOutput, predictors)
    doExperiment3(trainInput,trainOutput, predictors)
    doExperiment5(trainInput,trainOutput, predictors)
    #doExperiment6(trainInput, trainOutput, predictors)
    
    doKaggleTest(trainInput, testInput, trainOutput, testIDs, predictors)

    
# ===============================================================================
'''
Does k-fold CV on the Kaggle training set using LinearRegression.
(You might review the discussion in hw06 about the so-called "Kaggle training set"
versus other sets.)
'''
def doExperiment(trainInput, trainOutput, predictors):
    alg = LinearRegression()
    cvMeanScore = model_selection.cross_val_score(alg, trainInput.loc[:, predictors], trainOutput, cv=10, scoring='r2', n_jobs=-1).mean()
    print("Linear Regression, CV Average Score:", cvMeanScore)

def doExperiment2(trainInput, trainOutput, predictors):
    alg = BayesianRidge()
    cvMeanScore = model_selection.cross_val_score(alg, trainInput.loc[:, predictors], trainOutput, cv=10, scoring='r2', n_jobs=-1).mean()
    print("Bayesian Ridge, CV Average Score:", cvMeanScore) 
    
def doExperiment3(trainInput, trainOutput, predictors):
    alg = Ridge(0.2)
    cvMeanScore = model_selection.cross_val_score(alg, trainInput.loc[:, predictors], trainOutput, cv=10, scoring='r2', n_jobs=-1).mean()
    print("Ridge(0.2), CV Average Score:", cvMeanScore) 

def doExperiment4(trainInput, trainOutput, predictors):
    alg = lm.Lasso(alpha = 0.9)
    cvMeanScore = model_selection.cross_val_score(alg, trainInput.loc[:, predictors], trainOutput, cv=10, scoring='r2', n_jobs=-1).mean()
    print("Lasso(0.9), CV Average Score:", cvMeanScore) 

def doExperiment5(trainInput, trainOutput, predictors):
    alg = GradientBoostingRegressor(n_estimators=100,learning_rate = 0.1)
    cvMeanScore = model_selection.cross_val_score(alg, trainInput.loc[:, predictors], trainOutput, cv=10, scoring='r2', n_jobs=-1).mean()
    print("Gradient Boosting Regressor (n=100, rate =0.1), CV Average Score:", cvMeanScore) 
   
def doExperiment6(trainInput, trainOutput, predictors):
    alg = xgb.XGBRegressor(objective = "reg:linear", random_state=42)
    cvMeanScore = model_selection.cross_val_score(alg, trainInput.loc[:, predictors], trainOutput, cv=10, scoring='r2', n_jobs=-1).mean()
    print("XGB Regressor (n=100, rate =0.1), CV Average Score:", cvMeanScore)    
    
# ===============================================================================
'''
Runs the algorithm on the testing set and writes the results to a csv file.
'''
def doKaggleTest(trainInput, testInput, trainOutput, testIDs, predictors):
    alg = LinearRegression()
    #alg = GradientBoostingRegressor(n_estimators=100,learning_rate = 0.1)

    # Train the algorithm using all the training data
    alg.fit(trainInput.loc[:, predictors], trainOutput)

    # Make predictions on the test set.
    predictions = alg.predict(testInput.loc[:, predictors])
    exp = lambda x: math.exp(x)
    predictions = np.array([exp(x) for x in predictions])
    
    # Create a new dataframe with only the columns Kaggle wants from the dataset.
    submission = pd.DataFrame({
        "Id": testIDs,
        "SalePrice": predictions
    })

    # Prepare CSV
    submission.to_csv('data/testResults.csv', index=False)
    # Now, this .csv file can be uploaded to Kaggle

# ============================================================================
# Data cleaning - conversion, normalization

def normalize(df, columns):
    df.loc[:,columns] = (df.loc[:,columns] - df.loc[:,columns].min())/(df.loc[:,columns].max()-df.loc[:,columns].min())
    return df

def standardize(trainDF, testDF, columns):
    trainDF.loc[:,columns] = (trainDF.loc[:,columns] - trainDF.loc[:,columns].mean())/ trainDF.loc[:,columns].std()
    #testDF.loc[:,columns.drop(['SalePrice'])] = (testDF.loc[:,columns.drop(['SalePrice'])] - trainDF.loc[:,columns.drop(['SalePrice'])].mean())/ trainDF.loc[:,columns.drop(['SalePrice'])].std()
    return trainDF,testDF

'''
Pre-processing code will go in this function (and helper functions you call from here).
'''
def transformData(trainDF, testDF):
    
    '''Numeric Data'''
    # corr_data = trainDF.corr()
    # plt.subplots(figsize = (12,9))
    # sns.heatmap(corr_data,vmax=0.9,square=True)
    # plt.show()
    
    
    
    # How many of the varibles are numeric and contain missing values?
    # print("Before:")
    # print("All numeric attributes: ", getNumericAttrs(trainDF))
    # print(getAttrsWithMissingValues(trainDF).intersection(getNumericAttrs(trainDF)))
    # print(getAttrsWithMissingValues(testDF).intersection(getNumericAttrs(testDF)))
    
    '''Filling the missing values in the LotFrontage column'''
    avgLotFrontage = trainDF.loc[:,'LotFrontage'].mean()
    # algo =  KNeighborsClassifier(n_neighbors = 10)
    # trainDF.loc[:,'MSZoneNumber'] = trainDF.loc[:,'MSZoning'].map(
    #     lambda x: 0 if x == "RL" else (1 if x =="RM" else (2 if x =="RH" else (3 if x =="FV" else 4)))
    #     )
    # trainInputForAlgo = trainDF.loc[~(trainDF.loc[:,'LotFrontage'].isna()),['LotArea','MSSubClass','YearBuilt','MSZoneNumber']]
    # trainInputForAlgo = standardize(trainInputForAlgo, ['LotArea','MSSubClass','YearBuilt','MSZoneNumber'])
    # trainOuptutForAlgo = trainDF.loc[~(trainDF.loc[:,'LotFrontage'].isna()),'LotFrontage']
    # algo.fit(trainInputForAlgo,trainOuptutForAlgo)    
    # print(np.mean(model_selection.cross_val_score(algo, trainInputForAlgo, trainOuptutForAlgo, cv=10, scoring='accuracy')))
    trainDF.loc[trainDF.loc[:,'LotFrontage'].isna(),'LotFrontage'] = avgLotFrontage
    testDF.loc[testDF.loc[:,'LotFrontage'].isna(),'LotFrontage'] = avgLotFrontage
    '''end'''
    
    '''Filling the missing values in the MasVVnrArea column'''
    avgMasVnrArea = trainDF.loc[:,'MasVnrArea'].mean()
    trainDF.loc[trainDF.loc[:,'MasVnrArea'].isna(),'MasVnrArea'] = avgMasVnrArea
    testDF.loc[testDF.loc[:,'MasVnrArea'].isna(),'MasVnrArea'] = avgMasVnrArea
    '''end'''
    
    
    trainDF.loc[trainDF.loc[:,'GarageFinish'].isna(),'GarageYrBlt'] = trainDF.loc[trainDF.loc[:,'GarageFinish'].isna(),'YrSold']
    testDF.loc[testDF.loc[:,'GarageFinish'].isna(),'GarageYrBlt'] = testDF.loc[testDF.loc[:,'GarageFinish'].isna(),'YrSold']
    
    
    '''treating the columns that have missing values only in test dataset'''  
    rowWithNoBasement = testDF.loc[testDF.loc[:,'BsmtQual'].isna(),'BsmtQual'].index
    testDF.loc[rowWithNoBasement,['BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath']] = 0
    
    rowWithNoGarage = testDF.loc[testDF.loc[:,'GarageFinish'].isna(),'GarageFinish'].index
    testDF.loc[rowWithNoGarage,['GarageCars','GarageArea']] = 0
    

    # print("After:")
    # print("All numeric attributes: ", getNumericAttrs(trainDF))
    # print(getAttrsWithMissingValues(trainDF).intersection(getNumericAttrs(trainDF)))
    # print(getAttrsWithMissingValues(testDF).intersection(getNumericAttrs(testDF)))    
    
    # predictors = ['1stFlrSF', '2ndFlrSF','LotArea','FullBath','HalfBath','BedroomAbvGr','TotRmsAbvGrd','KitchenAbvGr','Fireplaces','LotFrontage','MSZoning_RL','MSZoning_RM'
    #               ,'MSZoning_RH','MSZoning_FV','Street_Pave','Street_Grvl']
    '''
    You'll want to use far more predictors than just these two columns, of course. But when you add
    more, you'll need to do things like handle missing values and convert non-numeric to numeric.
    Other preprocessing steps would likely be wise too, like standardization, get_dummies, 
    or converting or creating attributes based on your intuition about what's relevant in housing prices.
    '''
    
    
    '''Generating new variables and recoding old variables'''
    
    '''creating age of the house'''
    trainDF.loc[:,'ageOfHouse'] = trainDF.loc[:,'YrSold'] - trainDF.loc[:,'YearBuilt']
    testDF.loc[:,'ageOfHouse'] = testDF.loc[:,'YrSold'] - testDF.loc[:,'YearBuilt']
    trainDF.loc[:,'ageAfRemod'] = trainDF.loc[:,'YrSold'] - trainDF.loc[:,'YearRemodAdd']
    trainDF.loc[trainDF.loc[:,'ageAfRemod']<0,'ageAfRemod'] = 0
    testDF.loc[:,'ageAfRemod'] = testDF.loc[:,'YrSold'] - testDF.loc[:,'YearRemodAdd']
    testDF.loc[testDF.loc[:,'ageAfRemod']<0,'ageAfRemod'] = 0
    
    trainDF.loc[:,'TotFinBsmtSF'] = trainDF.loc[:,'BsmtFinSF1']+trainDF.loc[:,'BsmtFinSF2']
    testDF.loc[:,'TotFinBsmtSF'] = testDF.loc[:,'BsmtFinSF1']+testDF.loc[:,'BsmtFinSF2']
    
    trainDF.loc[:,'BsmtBath'] = trainDF.loc[:,'BsmtFullBath'] + trainDF.loc[:,'BsmtHalfBath'] * 0.5
    testDF.loc[:,'BsmtBath'] = testDF.loc[:,'BsmtFullBath'] + testDF.loc[:,'BsmtHalfBath'] * 0.5
    
    trainDF.loc[:,'Bath'] = trainDF.loc[:,'FullBath'] + trainDF.loc[:,'HalfBath'] * 0.5
    testDF.loc[:,'Bath'] = testDF.loc[:,'FullBath'] + testDF.loc[:,'HalfBath'] * 0.5
    
    
    '''Standardizing the predictors'''
    
    
    
    
    ''' Non Numeric Data'''
    # print("Before:")
    # print("All nonnumeric attributes: ", getNonNumericAttrs(trainDF))
    # print(getAttrsWithMissingValues(trainDF).intersection(getNonNumericAttrs(trainDF)))
    # print(getAttrsWithMissingValues(testDF).intersection(getNonNumericAttrs(testDF)))
    
    trainDF.loc[trainDF.loc[:,'Alley'].isna(),'Alley'] = "No Alley"
    testDF.loc[testDF.loc[:,'Alley'].isna(),'Alley'] = "No Alley"
        
    trainDF.loc[trainDF.loc[:,'BsmtQual'].isna(),['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1', 'BsmtFinType2']] = "No Basement"
    testDF.loc[testDF.loc[:,'BsmtQual'].isna(),['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1', 'BsmtFinType2']] = "No Basement"
    
    trainDF.loc[trainDF.loc[:,'Fireplaces']==0,'FireplaceQu'] = "No Fireplace"
    testDF.loc[testDF.loc[:,'Fireplaces']==0,'FireplaceQu'] = "No Fireplace"
    
    '''Garage related non numeric variables'''
    trainDF.loc[trainDF.loc[:,'GarageType'].isna(),['GarageType','GarageFinish','GarageQual','GarageCond']] = "No Garage"
    testDF.loc[testDF.loc[:,'GarageType'].isna(),['GarageType','GarageFinish','GarageQual','GarageCond']] = "No Garage"
    
    trainDF.loc[trainDF.loc[:,'PoolQC'].isna(),'PoolQC'] = "No Pool"
    testDF.loc[testDF.loc[:,'PoolQC'].isna(),'PoolQC'] = "No Pool"
    
    trainDF.loc[trainDF.loc[:,'Fence'].isna(),'Fence'] = "No Fence"
    testDF.loc[testDF.loc[:,'Fence'].isna(),'Fence'] = "No Fence"
    
    trainDF.loc[trainDF.loc[:,'MiscFeature'].isna(),'MiscFeature'] = "None"
    testDF.loc[testDF.loc[:,'MiscFeature'].isna(),'MiscFeature'] = "None"
    
    trainDF.loc[trainDF.loc[:,'BsmtExposure'].isna(),'BsmtExposure'] = trainDF.loc[:,'BsmtExposure'].mode().loc[0]
    trainDF.loc[trainDF.loc[:,'BsmtFinType2'].isna(),'BsmtFinType2'] = trainDF.loc[:,'BsmtFinType2'].mode().loc[0]
    testDF.loc[testDF.loc[:,'BsmtExposure'].isna(),'BsmtExposure'] = trainDF.loc[:,'BsmtExposure'].mode().loc[0]
    testDF.loc[testDF.loc[:,'BsmtFinType2'].isna(),'BsmtFinType2'] = trainDF.loc[:,'BsmtFinType2'].mode().loc[0]
    trainDF.loc[trainDF.loc[:,'MasVnrType'].isna(),'MasVnrType'] = trainDF.loc[:,'MasVnrType'].mode().loc[0]
    testDF.loc[testDF.loc[:,'MasVnrType'].isna(),'MasVnrType'] = trainDF.loc[:,'MasVnrType'].mode().loc[0]
    trainDF.loc[trainDF.loc[:,'Electrical'].isna(),'Electrical'] = trainDF.loc[:,'Electrical'].mode().loc[0]
    testDF.loc[testDF.loc[:,'Electrical'].isna(),'Electrical'] = trainDF.loc[:,'Electrical'].mode().loc[0]
    
    #trainDF,testDF = standardize(trainDF,testDF,getNumericAttrs(trainDF).drop(['Id','MSSubClass','OverallQual']))
    #trainDF,testDF = standardize(trainDF,testDF,getNumericAttrs(trainDF))
    
    # trainDF = pd.get_dummies(trainDF,columns=getNonNumericAttrs(trainDF))
    trainDF = pd.get_dummies(trainDF,columns=['MSZoning','Utilities','LotConfig','Neighborhood','BldgType','RoofMatl',
                                              'MasVnrType','ExterQual','BsmtQual','BsmtCond','BsmtExposure','Heating',
                                              'HeatingQC','CentralAir','KitchenQual','Functional','SaleType'])
    testDF = pd.get_dummies(testDF,columns=['MSZoning','Utilities','LotConfig','Neighborhood','BldgType','RoofMatl',
                                              'MasVnrType','ExterQual','BsmtQual','BsmtCond','BsmtExposure','Heating',
                                              'HeatingQC','CentralAir','KitchenQual','Functional','SaleType'])
    trainDF = pd.get_dummies(trainDF,columns=['MSSubClass','OverallQual'])
    testDF = pd.get_dummies(testDF,columns=['MSSubClass','OverallQual'])
    
    
    # How many of the varibles are numeric and contain missing values?
    # print("After:")
    # print("All nonnumeric attributes: ", getNonNumericAttrs(trainDF))
    # print(getAttrsWithMissingValues(trainDF).intersection(getNonNumericAttrs(trainDF)))
    # print(getAttrsWithMissingValues(testDF).intersection(getNonNumericAttrs(testDF)))
    
    
    
    '''Standardizing the numeric attributes'''
    #trainDF = standardize(trainDF,['ageOfHouse','ageAfRemod','BsmtUnfSF','TotalBsmtSF','TotFinBsmtSF','1stFlrSF','GarageArea'])
    
    #predictors = getNumericAttrs(trainDF).drop(['Id','SalePrice'])
    predictors = getNumericAttrs(trainDF).drop(['Id','SalePrice','GarageYrBlt','YearBuilt','YearRemodAdd','MoSold','YrSold','OverallCond',
                                                'GarageCars','LowQualFinSF','BsmtFinSF1','BsmtFinSF2','TotalBsmtSF','GrLivArea',
                                                'BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','TotRmsAbvGrd','3SsnPorch'])
    
    
    
    trainInput = trainDF.loc[:, predictors]
    testDF.loc[:,['Utilities_NoSeWa','RoofMatl_ClyTile','RoofMatl_Membran','RoofMatl_Metal','RoofMatl_Roll','Heating_Floor','Heating_OthW']] = 0 
    #testInput = testDF.loc[:, predictors]
    
    #trainDF,testDF = standardize(trainDF,testDF,getNumericAttrs(trainDF))
    
    testInput = testDF.loc[:,:]
    '''
    Any transformations you do on the trainInput will need to be done on the
    testInput the same way. (For example, using the exact same min and max, if
    you're doing normalization.)
    '''
    sns.distplot(trainDF.loc[:,'SalePrice'])
    plt.show()
    trainDF.loc[:,'SalePrice'] = trainDF.loc[:,'SalePrice'].map(
              lambda x: math.log(x)
         )
 
    trainOutput = trainDF.loc[:, 'SalePrice']
    sns.distplot(trainOutput)
    plt.show()
    testIDs = testDF.loc[:, 'Id']
    
   
    
    return trainInput, testInput, trainOutput, testIDs, predictors
    
# ===============================================================================
'''
Demonstrates some provided helper functions that you might find useful.
'''
def demonstrateHelpers(trainDF):
    print("Attributes with missing values:", getAttrsWithMissingValues(trainDF), sep='\n')
    
    numericAttrs = getNumericAttrs(trainDF)
    print("Numeric attributes:", numericAttrs, sep='\n')
    
    nonnumericAttrs = getNonNumericAttrs(trainDF)
    print("Non-numeric attributes:", nonnumericAttrs, sep='\n')

    print("Values, for each non-numeric attribute:", getAttrToValuesDictionary(trainDF.loc[:, nonnumericAttrs]), sep='\n')

# ===============================================================================
'''
Returns a dictionary mapping an attribute to the array of values for that attribute.
'''
def getAttrToValuesDictionary(df):
    attrToValues = {}
    for attr in df.columns.values:
        attrToValues[attr] = df.loc[:, attr].unique()

    return attrToValues

# ===============================================================================
'''
Returns the attributes with missing values.
'''
def getAttrsWithMissingValues(df):
    valueCountSeries = df.count(axis=0)  # 0 to count down the rows
    numCases = df.shape[0]  # Number of examples - number of rows in the data frame
    missingSeries = (numCases - valueCountSeries)  # A Series showing the number of missing values, for each attribute
    attrsWithMissingValues = missingSeries[missingSeries != 0].index
    return attrsWithMissingValues

# =============================================================================

'''
Returns the numeric attributes.
'''
def getNumericAttrs(df):
    return __getNumericHelper(df, True)

'''
Returns the non-numeric attributes.
'''
def getNonNumericAttrs(df):
    return __getNumericHelper(df, False)

def __getNumericHelper(df, findNumeric):
    isNumeric = df.applymap(np.isreal) # np.isreal is a function that takes a value and returns True (the value is real) or False
                                       # applymap applies the given function to the whole data frame
                                       # So this returns a DataFrame of True/False values indicating for each value in the original DataFrame whether it is real (numeric) or not

    isNumeric = isNumeric.all() # all: For each column, returns whether all elements are True
    attrs = isNumeric.loc[isNumeric==findNumeric].index # selects the values in isNumeric that are <findNumeric> (True or False)
    return attrs

# =============================================================================

if __name__ == "__main__":
    main()


