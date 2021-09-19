# This code fits machine learning models
# Tian Jiang


#%% Libraries
import numpy as np 
import pandas as pd
from datetime import date
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
#%matplotlib inline  
from statsmodels.tools.eval_measures import mse, rmse
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.tree import DecisionTreeRegressor
import math
import warnings
warnings.filterwarnings("ignore")
#import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from matplotlib.dates import  DateFormatter
import os

# To measure execution time...
import time

# Function
def MAPE(y2_test, y2_preds):
    valMAPE = np.mean(np.abs((y2_test - y2_preds) / y2_test))*100 
    if(np.isinf(valMAPE) or np.isnan(valMAPE)):
        valMAPE = np.mean(np.abs((y2_test[y2_test>0] - y2_preds[y2_test>0]) / y2_test[y2_test>0]))*100
    if(np.isnan(valMAPE)):
        valMAPE = 100
    return(valMAPE)


pathORIG = os.getcwd()
folderSaved = 'Saved/MLM'
os.mkdir(folderSaved)
pathFolderSaved = folderSaved + '/'


#%% Parameters (
print('==========================================')
print('PARAMETERS')
print('==========================================')
# (OTHERS - DAILY)
LBack = 4 # Look back >=1 
LPred = 91 # Future prediction >=0 ... 0 means X(t) -> Y(t+1) 
# 91 means actaully 92... i.e. information from years 2015, 2016, 2017, 2018
# This number should be the same as number of rows in 'test period'-1

# Dataset and State 
nameState = 'Manitoba'  
#nameState = 'mean_weather_and_NDVI'  
nameDataset = 'Dataset/'+nameState+'.csv'

selColVariable = ['Temp-Max', 'Temp-Min', 'pcp', 'NDVI']
selVAR_X = ['Temp-Max', 'Temp-Min', 'pcp'] # Regressor variables 
selVAR_Y = ['NDVI'] # Response variable 

# Train and test day   
dayTrain_ini = date(2000,4,1) # Train INI 
dayTrain_fin = date(2014,9,30) # Train FINAL 
dayTest_ini = date(2015,4,1) # Test INI 
dayTest_fin = date(2018,9,30) # Test FINAL 



#%% CREATE DATASET
print('==========================================')
print('CREATE DATASET')
print('==========================================')
# Open Dataset
dfComplete = pd.read_csv(pathORIG+'/'+nameDataset) 
countiesTXT = list(np.unique(dfComplete.Municipality)) # List of Counties
dfAux = dfComplete.copy()
dfAux.week = pd.to_datetime(dfAux.week, format='%m/%d/%y')  
# Creating individual tables 
df_VAR = []
for iVar in range(len(selColVariable)):
    df_Aux2 = dfAux.pivot(index='week', columns='Municipality', values=selColVariable[iVar])  
    countiesTXT = list(df_Aux2.columns)
    df_Aux2.columns = df_Aux2.columns + ': ' + selColVariable[iVar]
    df_VAR.append(df_Aux2.copy()) 
# Build Full Datasets 
df_Full_X = pd.DataFrame(index=df_VAR[0].index)
df_Full_Y = pd.DataFrame(index=df_VAR[0].index)
for kCT in range(len(countiesTXT)): # For all columns
    for iVarX in selVAR_X: # For variables in Regressors
        id_X = selColVariable.index(iVarX) # Index
        # Select Column
        df_dataCOL = pd.DataFrame( index=df_VAR[id_X].index, data=df_VAR[id_X][df_VAR[id_X].columns[kCT]] )  
        # Concatenate...  
        df_Full_X = pd.concat([df_Full_X.copy(), df_dataCOL.copy()], axis=1)
    for iVarY in selVAR_Y: # For variables in Response
        id_Y = selColVariable.index(iVarY) # Index
        # Select Column
        df_dataCOL = pd.DataFrame( index=df_VAR[id_Y].index, data=df_VAR[id_Y][df_VAR[id_Y].columns[kCT]] )  
        # Concatenate...  
        df_Full_Y = pd.concat([df_Full_Y.copy(), df_dataCOL.copy()], axis=1)



#%% Only Train 
##### Auxiliary Variables #######
# --------------- Regresor variables ...
df_X = df_Full_X.copy() # All variables
# --------------- Response variables ...
df_Y = df_Full_Y.copy() # All variables
# --- Features ...
NFeatures = df_X.shape[1] # Number of Features (Input)
NFeaturesOutput = df_Y.shape[1] # Number of Features (Output)
# --- Only keep the training set
df_Y = df_Y[df_Y.index>=np.datetime64(dayTrain_ini)]
df_Y = df_Y[df_Y.index<=np.datetime64(dayTrain_fin)]
df_X = df_X[df_X.index>=np.datetime64(dayTrain_ini)]
df_X = df_X[df_X.index<=np.datetime64(dayTrain_fin)]

# Number of rows for Training
NTotal = len(df_Y)  

### Standardized data
# scaler_X = StandardScaler()
# scaler_Y = StandardScaler()
# scaler_X.fit(df_X)
# scaler_Y.fit(df_Y)
# dataNorm_X = scaler_X.transform(df_X)
# dataNorm_Y = scaler_Y.transform(df_Y)
###  scaler_X.inverse_transform(dataNorm_X) #+++ To de-standardrized data
# Unnormalized data
dataNorm_X = df_X.values 
dataNorm_Y = df_Y.values 
# Split into train and test sets 
dataX, dataY = [], [] 
dataX = np.zeros((NTotal-LBack+1, LBack*NFeatures))
for i in range(NTotal-LBack+1):
    dataX[i,:] = dataNorm_X[i:(i+LBack),:].reshape((1,LBack*NFeatures))

dataY = np.zeros((NTotal-LBack-LPred, NFeaturesOutput)) 
for i in range(NTotal-LBack-LPred):
    dataY[i,:] = dataNorm_Y[i+LBack+LPred,:]

train_size = dataY.shape[0]
pred_size =  dataX.shape[0] - train_size
trainX = dataX[0:train_size,:]
trainY = dataY[0:train_size,:]
predX = dataX[train_size:(train_size+pred_size),:]

# Commented out IPython magic to ensure Python compatibility.
#%%
###############################################################################################
# SVR
###############################################################################################
# ---- To create the folder
folderRun = pathFolderSaved +  'SVR'
os.mkdir(folderRun) 

#%% Fit regression model
# "estimator__" for inner parameter
gs_svr = GridSearchCV(MultiOutputRegressor(SVR()),
                   param_grid={"estimator__C": [1e0, 1e1, 1e2, 1e3],
                               "estimator__gamma": np.logspace(-2, 2, 5)})

t0 = time.time()
gs_svr.fit(trainX, trainY)
svr_fit = time.time() - t0
print("SVR complexity and bandwidth selected and model fitted in %.3f s"
#       % svr_fit)

selectedModel = gs_svr.best_estimator_



# %% 
print('==========================================')
print('MAKE PREDICTIONS')
print('==========================================')
# make predictions 
trainPredict = selectedModel.predict(trainX) 
predPredict = selectedModel.predict(predX) 
# invert standardrized predictions 
#scaler_X.inverse_transform(dataNorm_X) #+++ To de-standardrized data
# trainPredict = scaler_Y.inverse_transform(trainPredict) #+++ To de-standardrized data
# trainY = scaler_Y.inverse_transform(trainY) 
# predPredict = scaler_Y.inverse_transform(predPredict) 
# For non-standardrized predictions 
#trainPredict = trainPredict
#trainY = trainY
#predPredict = predPredict
# Mean squared error  
trainScore = math.sqrt(mean_squared_error(trainY, trainPredict))
print('Train Score: %.2f RMSE' % (trainScore)) 

#%% Datasets in Pandas (Save predictions)
# --- CSV files 
df_trainPredict = pd.DataFrame(trainPredict, columns=df_Y.columns, index=df_Full_Y.index[LBack+LPred:LBack+LPred+train_size])
df_predPredict = pd.DataFrame(predPredict, columns=df_Y.columns, index=df_Full_Y.index[LBack+LPred+train_size:LBack+LPred+train_size+pred_size]) 
df_trainPredict.to_csv(folderRun+'/'+nameState+'_trainPredict.csv')
df_predPredict.to_csv(folderRun+'/'+nameState+'_predPredict.csv') 
# --- PLOT 
#df_Y.plot(figsize=(40,6), legend=True)
#df_trainPredict.plot(figsize=(40,6), legend=True)  
#df_predPredict.plot(figsize=(40,6), legend=True)

#%% To compute Errors and measures 
# Select Portion of Training
df_trainREAL = df_Full_Y[df_Full_Y.index>=df_trainPredict.index[0]]
df_trainREAL = df_trainREAL[df_trainREAL.index<=df_trainPredict.index[-1]]
# Select Portion of Prediction
df_predREAL = df_Full_Y[df_Full_Y.index>=df_predPredict.index[0]]
df_predREAL = df_predREAL[df_predREAL.index<=df_predPredict.index[-1]]
# Empty Matrix Error
measureTXT = ['Train_MAE', 'Train_MSE', 'Train_RMSE', 'Train_MAPE', 'Test_MAE', 'Test_MSE', 'Test_RMSE', 'Test_MAPE']
df_Measures = pd.DataFrame(index=measureTXT)  
df_Measures.index.name = 'Measures'

# To compute all Errors
# os.mkdir(folderRun+'/'+'GraphsMunicipality')
for txtCT in list(df_Full_Y.columns): # For all columns
    #txtCT in countiesTXT:  
    #print('***@@@*** Measures: '+txtCT+'   @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')  
    auxCOL = []
    # TRAIN
    auxCOL.append(  mean_absolute_error(df_trainREAL[txtCT].values, df_trainPredict[txtCT].values)  )
    auxCOL.append(  mse(df_trainREAL[txtCT].values, df_trainPredict[txtCT].values)  )
    auxCOL.append(  rmse(df_trainREAL[txtCT].values, df_trainPredict[txtCT].values)  )
    auxCOL.append(  MAPE(df_trainREAL[txtCT].values, df_trainPredict[txtCT].values)  )
    # TEST
    auxCOL.append(  mean_absolute_error(df_predREAL[txtCT].values, df_predPredict[txtCT].values)  )
    auxCOL.append(  mse(df_predREAL[txtCT].values, df_predPredict[txtCT].values)  )
    auxCOL.append(  rmse(df_predREAL[txtCT].values, df_predPredict[txtCT].values)  )
    auxCOL.append(  MAPE(df_predREAL[txtCT].values, df_predPredict[txtCT].values)  )
    # Building 
    df_Mcol = pd.DataFrame(index=measureTXT, data={txtCT.split(':')[0]:auxCOL})
    df_Mcol.index.name = 'Measures' 
    # Concatenate...
    df_Measures = pd.concat([df_Measures.copy(), df_Mcol.copy()], axis=1)

    # Plot time series
    df_CTPlot = pd.DataFrame(index=df_Full_Y.index, data={'Real':list(df_Full_Y[txtCT].values)})
    vecTrainPred = [np.nan]*sum(df_Full_Y.index<df_trainPredict[txtCT].index[0]) + list(df_trainPredict[txtCT].values) + [np.nan]*sum(df_Full_Y.index>df_trainPredict[txtCT].index[-1])
    vecPredPred = [np.nan]*sum(df_Full_Y.index<df_predPredict[txtCT].index[0]) + list(df_predPredict[txtCT].values) + [np.nan]*sum(df_Full_Y.index>df_predPredict[txtCT].index[-1])
    # Concatenate...  
    df_CTPlot = pd.concat([df_CTPlot.copy(), pd.DataFrame(index=df_Full_Y.index, data={'Train':vecTrainPred})], axis=1)
    df_CTPlot = pd.concat([df_CTPlot.copy(), pd.DataFrame(index=df_Full_Y.index, data={'Test':vecPredPred})], axis=1)
    # Plot
    #df_CTPlot.plot(title=txtCT, color=['gray', 'blue', 'red']).legend('best')
    #df_CTPlot.plot(title=txtCT.split(':')[0], ylabel=txtCT.split(': ')[1], color=['gray', 'blue', 'red'])
    df_CTPlot.plot(title=txtCT.split(':')[0], linewidth=0.5, marker='.', ylabel=txtCT.split(': ')[1], color=['gray', 'blue', 'red'])
    plt.savefig(folderRun+'/'+'GraphsMunicipality/' + txtCT.split(':')[0] + '.pdf', bbox_inches='tight')
    plt.close()

# --- CSV files 
df_Measures.to_csv(folderRun+'/'+nameState+'_Measures_Error.csv')  

#%% Computing further statistics on measures...
# Compute Overall
df_Total_Mean = pd.DataFrame(index=df_Measures.index, data=df_Measures.mean(axis=1), columns=['Total_Mean_Error_Dataset']) 
# Save Data
folderResult = pathFolderSaved + 'Overall_Results_SVR'
os.mkdir(folderResult)
df_Total_Mean.to_csv(folderResult + '/Total_Mean.csv')  
df_Total_Mean

df_Total_Mean

# Commented out IPython magic to ensure Python compatibility.
#%%
###############################################################################################
# LASSO
###############################################################################################
# ---- To create the folder
folderRun = pathFolderSaved +  'LASSO'
os.mkdir(folderRun) 

#%% Fit regression model
# "estimator__" for inner parameter
gs_lasso = GridSearchCV(MultiOutputRegressor(Lasso()),
                        param_grid={"estimator__alpha": np.logspace(-4, -0.5, 10)})

t0 = time.time()
gs_lasso.fit(trainX, trainY)
lasso_fit = time.time() - t0
print("Lasso penalty coefficient selected and model fitted in %.3f s"
#       % lasso_fit)

selectedModel = gs_lasso.best_estimator_



# %% 
print('==========================================')
print('MAKE PREDICTIONS')
print('==========================================')
# make predictions 
trainPredict = selectedModel.predict(trainX) 
predPredict = selectedModel.predict(predX) 
# invert standardrized predictions 
#scaler_X.inverse_transform(dataNorm_X) #+++ To de-standardrized data
# trainPredict = scaler_Y.inverse_transform(trainPredict) #+++ To de-standardrized data
# trainY = scaler_Y.inverse_transform(trainY) 
# predPredict = scaler_Y.inverse_transform(predPredict) 
# For non-standardrized predictions 
#trainPredict = trainPredict
#trainY = trainY
#predPredict = predPredict
# Mean squared error  
trainScore = math.sqrt(mean_squared_error(trainY, trainPredict))
print('Train Score: %.2f RMSE' % (trainScore)) 

#%% Datasets in Pandas (Save predictions)
# --- CSV files 
df_trainPredict = pd.DataFrame(trainPredict, columns=df_Y.columns, index=df_Full_Y.index[LBack+LPred:LBack+LPred+train_size])
df_predPredict = pd.DataFrame(predPredict, columns=df_Y.columns, index=df_Full_Y.index[LBack+LPred+train_size:LBack+LPred+train_size+pred_size]) 
df_trainPredict.to_csv(folderRun+'/'+nameState+'_trainPredict.csv')
df_predPredict.to_csv(folderRun+'/'+nameState+'_predPredict.csv') 
# --- PLOT 
#df_Y.plot(figsize=(40,6), legend=True)
#df_trainPredict.plot(figsize=(40,6), legend=True)  
#df_predPredict.plot(figsize=(40,6), legend=True)

#%% To compute Errors and measures 
# Select Portion of Training
df_trainREAL = df_Full_Y[df_Full_Y.index>=df_trainPredict.index[0]]
df_trainREAL = df_trainREAL[df_trainREAL.index<=df_trainPredict.index[-1]]
# Select Portion of Prediction
df_predREAL = df_Full_Y[df_Full_Y.index>=df_predPredict.index[0]]
df_predREAL = df_predREAL[df_predREAL.index<=df_predPredict.index[-1]]
# Empty Matrix Error
measureTXT = ['Train_MAE', 'Train_MSE', 'Train_RMSE', 'Train_MAPE', 'Test_MAE', 'Test_MSE', 'Test_RMSE', 'Test_MAPE']
df_Measures = pd.DataFrame(index=measureTXT)  
df_Measures.index.name = 'Measures'

# To compute all Errors
os.mkdir(folderRun+'/'+'GraphsMunicipality')
for txtCT in list(df_Full_Y.columns): # For all columns
    #txtCT in countiesTXT:  
    #print('***@@@*** Measures: '+txtCT+'   @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')  
    auxCOL = []
    # TRAIN
    auxCOL.append(  mean_absolute_error(df_trainREAL[txtCT].values, df_trainPredict[txtCT].values)  )
    auxCOL.append(  mse(df_trainREAL[txtCT].values, df_trainPredict[txtCT].values)  )
    auxCOL.append(  rmse(df_trainREAL[txtCT].values, df_trainPredict[txtCT].values)  )
    auxCOL.append(  MAPE(df_trainREAL[txtCT].values, df_trainPredict[txtCT].values)  )
    # TEST
    auxCOL.append(  mean_absolute_error(df_predREAL[txtCT].values, df_predPredict[txtCT].values)  )
    auxCOL.append(  mse(df_predREAL[txtCT].values, df_predPredict[txtCT].values)  )
    auxCOL.append(  rmse(df_predREAL[txtCT].values, df_predPredict[txtCT].values)  )
    auxCOL.append(  MAPE(df_predREAL[txtCT].values, df_predPredict[txtCT].values)  )
    # Building 
    df_Mcol = pd.DataFrame(index=measureTXT, data={txtCT.split(':')[0]:auxCOL})
    df_Mcol.index.name = 'Measures' 
    # Concatenate...
    df_Measures = pd.concat([df_Measures.copy(), df_Mcol.copy()], axis=1)

    # Plot time series
    df_CTPlot = pd.DataFrame(index=df_Full_Y.index, data={'Real':list(df_Full_Y[txtCT].values)})
    vecTrainPred = [np.nan]*sum(df_Full_Y.index<df_trainPredict[txtCT].index[0]) + list(df_trainPredict[txtCT].values) + [np.nan]*sum(df_Full_Y.index>df_trainPredict[txtCT].index[-1])
    vecPredPred = [np.nan]*sum(df_Full_Y.index<df_predPredict[txtCT].index[0]) + list(df_predPredict[txtCT].values) + [np.nan]*sum(df_Full_Y.index>df_predPredict[txtCT].index[-1])
    # Concatenate...  
    df_CTPlot = pd.concat([df_CTPlot.copy(), pd.DataFrame(index=df_Full_Y.index, data={'Train':vecTrainPred})], axis=1)
    df_CTPlot = pd.concat([df_CTPlot.copy(), pd.DataFrame(index=df_Full_Y.index, data={'Test':vecPredPred})], axis=1)
    # Plot
    #df_CTPlot.plot(title=txtCT, color=['gray', 'blue', 'red']).legend('best')
    #df_CTPlot.plot(title=txtCT.split(':')[0], ylabel=txtCT.split(': ')[1], color=['gray', 'blue', 'red'])
    df_CTPlot.plot(title=txtCT.split(':')[0], linewidth=0.5, marker='.', ylabel=txtCT.split(': ')[1], color=['gray', 'blue', 'red'])
    plt.savefig(folderRun+'/'+'GraphsMunicipality/' + txtCT.split(':')[0] + '.pdf', bbox_inches='tight')
    plt.close()

# --- CSV files 
df_Measures.to_csv(folderRun+'/'+nameState+'_Measures_Error.csv')  

#%% Computing further statistics on measures...
# Compute Overall
df_Total_Mean = pd.DataFrame(index=df_Measures.index, data=df_Measures.mean(axis=1), columns=['Total_Mean_Error_Dataset']) 
# Save Data
folderResult = pathFolderSaved + 'Overall_Results_LASSO'
os.mkdir(folderResult)
df_Total_Mean.to_csv(folderResult + '/Total_Mean.csv')  
df_Total_Mean

# Commented out IPython magic to ensure Python compatibility.
#%%
###############################################################################################
# Decision Tree
###############################################################################################
# ---- To create the folder
folderRun = pathFolderSaved +  'DT'
# os.mkdir(folderRun) 

#%% Fit regression model
# "estimator__" for inner parameter
gs_dt = GridSearchCV(DecisionTreeRegressor(),
                     param_grid={"max_depth": list(range(10))})

t0 = time.time()
gs_dt.fit(trainX, trainY)
dt_fit = time.time() - t0
print("Random forest number of trees and max depth selected and model fitted in %.3f s"
#       % dt_fit)

selectedModel = gs_dt.best_estimator_


# %% 
print('==========================================')
print('MAKE PREDICTIONS')
print('==========================================')
# make predictions 
trainPredict = selectedModel.predict(trainX) 
predPredict = selectedModel.predict(predX) 
# invert standardrized predictions 
#scaler_X.inverse_transform(dataNorm_X) #+++ To de-standardrized data
# trainPredict = scaler_Y.inverse_transform(trainPredict) #+++ To de-standardrized data
# trainY = scaler_Y.inverse_transform(trainY) 
# predPredict = scaler_Y.inverse_transform(predPredict) 
# For non-standardrized predictions 
#trainPredict = trainPredict
#trainY = trainY
#predPredict = predPredict
# Mean squared error  
trainScore = math.sqrt(mean_squared_error(trainY, trainPredict))
print('Train Score: %.2f RMSE' % (trainScore)) 

# Datasets in Pandas (Save predictions)
# --- CSV files 
df_trainPredict = pd.DataFrame(trainPredict, columns=df_Y.columns, index=df_Full_Y.index[LBack+LPred:LBack+LPred+train_size])
df_predPredict = pd.DataFrame(predPredict, columns=df_Y.columns, index=df_Full_Y.index[LBack+LPred+train_size:LBack+LPred+train_size+pred_size]) 
df_trainPredict.to_csv(folderRun+'/'+nameState+'_trainPredict.csv')
df_predPredict.to_csv(folderRun+'/'+nameState+'_predPredict.csv') 
# --- PLOT 
#df_Y.plot(figsize=(40,6), legend=True)
#df_trainPredict.plot(figsize=(40,6), legend=True)  
#df_predPredict.plot(figsize=(40,6), legend=True)

# To compute Errors and measures 
# Select Portion of Training
df_trainREAL = df_Full_Y[df_Full_Y.index>=df_trainPredict.index[0]]
df_trainREAL = df_trainREAL[df_trainREAL.index<=df_trainPredict.index[-1]]
# Select Portion of Prediction
df_predREAL = df_Full_Y[df_Full_Y.index>=df_predPredict.index[0]]
df_predREAL = df_predREAL[df_predREAL.index<=df_predPredict.index[-1]]
# Empty Matrix Error
measureTXT = ['Train_MAE', 'Train_MSE', 'Train_RMSE', 'Train_MAPE', 'Test_MAE', 'Test_MSE', 'Test_RMSE', 'Test_MAPE']
df_Measures = pd.DataFrame(index=measureTXT)  
df_Measures.index.name = 'Measures'

# To compute all Errors
# os.mkdir(folderRun+'/'+'GraphsMunicipality')
for txtCT in list(df_Full_Y.columns): # For all columns
    #txtCT in countiesTXT:  
    #print('***@@@*** Measures: '+txtCT+'   @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')  
    auxCOL = []
    # TRAIN
    auxCOL.append(  mean_absolute_error(df_trainREAL[txtCT].values, df_trainPredict[txtCT].values)  )
    auxCOL.append(  mse(df_trainREAL[txtCT].values, df_trainPredict[txtCT].values)  )
    auxCOL.append(  rmse(df_trainREAL[txtCT].values, df_trainPredict[txtCT].values)  )
    auxCOL.append(  MAPE(df_trainREAL[txtCT].values, df_trainPredict[txtCT].values)  )
    # TEST
    auxCOL.append(  mean_absolute_error(df_predREAL[txtCT].values, df_predPredict[txtCT].values)  )
    auxCOL.append(  mse(df_predREAL[txtCT].values, df_predPredict[txtCT].values)  )
    auxCOL.append(  rmse(df_predREAL[txtCT].values, df_predPredict[txtCT].values)  )
    auxCOL.append(  MAPE(df_predREAL[txtCT].values, df_predPredict[txtCT].values)  )
    # Building 
    df_Mcol = pd.DataFrame(index=measureTXT, data={txtCT.split(':')[0]:auxCOL})
    df_Mcol.index.name = 'Measures' 
    # Concatenate...
    df_Measures = pd.concat([df_Measures.copy(), df_Mcol.copy()], axis=1)

    # Plot time series
    df_CTPlot = pd.DataFrame(index=df_Full_Y.index, data={'Real':list(df_Full_Y[txtCT].values)})
    vecTrainPred = [np.nan]*sum(df_Full_Y.index<df_trainPredict[txtCT].index[0]) + list(df_trainPredict[txtCT].values) + [np.nan]*sum(df_Full_Y.index>df_trainPredict[txtCT].index[-1])
    vecPredPred = [np.nan]*sum(df_Full_Y.index<df_predPredict[txtCT].index[0]) + list(df_predPredict[txtCT].values) + [np.nan]*sum(df_Full_Y.index>df_predPredict[txtCT].index[-1])
    # Concatenate...  
    df_CTPlot = pd.concat([df_CTPlot.copy(), pd.DataFrame(index=df_Full_Y.index, data={'Train':vecTrainPred})], axis=1)
    df_CTPlot = pd.concat([df_CTPlot.copy(), pd.DataFrame(index=df_Full_Y.index, data={'Test':vecPredPred})], axis=1)
    # Plot
    #df_CTPlot.plot(title=txtCT, color=['gray', 'blue', 'red']).legend('best')
    #df_CTPlot.plot(title=txtCT.split(':')[0], ylabel=txtCT.split(': ')[1], color=['gray', 'blue', 'red'])
    df_CTPlot.plot(title=txtCT.split(':')[0], linewidth=0.5, marker='.', ylabel=txtCT.split(': ')[1], color=['gray', 'blue', 'red'])
    plt.savefig(folderRun+'/'+'GraphsMunicipality/' + txtCT.split(':')[0] + '.pdf', bbox_inches='tight')
    plt.close()

# --- CSV files 
df_Measures.to_csv(folderRun+'/'+nameState+'_Measures_Error.csv')  

# Computing further statistics on measures...
# Compute Overall
df_Total_Mean = pd.DataFrame(index=df_Measures.index, data=df_Measures.mean(axis=1), columns=['Total_Mean_Error_Dataset']) 
# Save Data
folderResult = pathFolderSaved + 'Overall_Results_DT'
# os.mkdir(folderResult)
df_Total_Mean.to_csv(folderResult + '/Total_Mean.csv')  
df_Total_Mean
