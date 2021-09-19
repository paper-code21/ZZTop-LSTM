# This code for the baseline LSTM model
# Ignacio Segovia-Dominguez
# Tian


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
import math
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Activation
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
import warnings
warnings.filterwarnings("ignore")
#import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from matplotlib.dates import  DateFormatter
import os

# Function
def MAPE(y2_test, y2_preds):
    valMAPE = np.mean(np.abs((y2_test - y2_preds) / y2_test))*100 
    if(np.isinf(valMAPE) or np.isnan(valMAPE)):
        valMAPE = np.mean(np.abs((y2_test[y2_test>0] - y2_preds[y2_test>0]) / y2_test[y2_test>0]))*100
    if(np.isnan(valMAPE)):
        valMAPE = 100
    return(valMAPE)

# To measure execution time...
import time
start_time = time.time()


pathORIG = os.getcwd()
folderSaved = 'Saved/Baseline'
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
dayTrain_ini = date(2000,4,2) # Train INI 
dayTrain_fin = date(2014,9,7) # Train FINAL 
dayTest_ini = date(2015,4,5) # Test INI 
dayTest_fin = date(2018,9,2) # Test FINAL 

# About Runs
Number_Runs = 10

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

#%%
###################
# RUNS
###################
ALL_MEASURES = []

for ID_RUN in range(Number_Runs):
    # ---- To create the folder
    folderRun = pathFolderSaved +  'RUN_' + str(ID_RUN)
    os.mkdir(folderRun) 


    #%% Only Train 
    print('==========================================')
    print('RUN: '+str(ID_RUN)+'  >>  FORMAT OF TIME SERIES - TRAINING AND TEST SETS')
    print('==========================================')
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
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()
    scaler_X.fit(df_X)
    scaler_Y.fit(df_Y)
    dataNorm_X = scaler_X.transform(df_X)
    dataNorm_Y = scaler_Y.transform(df_Y)
    ###  scaler_X.inverse_transform(dataNorm_X) #+++ To de-standardrized data
    # Unnormalized data
    #dataNorm_X = df_X.values 
    #dataNorm_Y = df_Y.values 
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

    # %% Prepare for LSTM
    # reshape input to be [samples, time steps, features]
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    predX = np.reshape(predX, (predX.shape[0], 1, predX.shape[1]))

    #%% Create and fit the LSTM network
    print('==========================================')
    print('RUN: '+str(ID_RUN)+'  >>  COMPUTE LSTM')
    print('==========================================')
    model = Sequential()
    batch_size = 8 
    #pDrop = 0.1
    pDrop = 0.2
    valPatience = 800
    #valPatience = 1000
    #valPatience = 3000

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=valPatience) 
    mc = ModelCheckpoint(folderRun+'/'+'best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True) 

    #model.add(LSTM(512, activation='linear', return_sequences=True, input_shape=(1, LBack*NFeatures), dropout=pDrop))
    #model.add(LSTM(256, activation='linear', return_sequences=True, dropout=pDrop)) 
    #model.add(LSTM(256, activation='linear', dropout=pDrop))

    model.add(LSTM(256, activation='linear', return_sequences=True, input_shape=(1, LBack*NFeatures), dropout=pDrop))
    model.add(LSTM(128, activation='linear', return_sequences=True, dropout=pDrop)) 
    model.add(LSTM(128, activation='linear', dropout=pDrop))

    #model.add(Dense(1)) # A solely output...
    model.add(Dense(trainY.shape[1])) # Multiple Outputs...

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error']) 

    ### OPTION 1 FOR VALIDATION
    #history = model.fit(trainX, trainY, epochs=8000, batch_size=batch_size, verbose=0, callbacks=[es,mc], validation_data=(trainX, trainY))  
    ### OPTION 2 FOR VALIDATION
    history = model.fit(trainX, trainY, epochs=8000, batch_size=batch_size, verbose=0, callbacks=[es,mc], validation_split=0.2)  

    #%% plot history
    print('==========================================')
    print('RUN: '+str(ID_RUN)+'  >>  PLOT HISTORY OF TRAINING')
    print('==========================================')
    plt.plot(history.history['loss'], label='train') 
    plt.plot(history.history['val_loss'], label='validation')
    plt.title(nameState)
    plt.ylabel('Error')
    plt.xlabel('Epochs')
    plt.legend()
    #plt.show()
    plt.savefig(folderRun+'/'+'History.pdf', bbox_inches='tight')
    plt.close()

    #%% To test if we saved the best model
    print('==========================================')
    print('RUN: '+str(ID_RUN)+'  >>  SAVE THE BEST MODEL (TO AVOID RETRAINING IN THE FUTURE)')
    print('==========================================')
    saved_model = load_model(folderRun+'/'+'best_model.h5')
    # trainPred = saved_model.predict(trainX)
    # plt.plot(trainPredict)
    # plt.plot(trainPred)
    # plt.show()

    # %% Predictions
    print('==========================================')
    print('RUN: '+str(ID_RUN)+'  >>  MAKE PREDICTIONS')
    print('==========================================')
    # make predictions 
    trainPredict = saved_model.predict(trainX) 
    predPredict = saved_model.predict(predX) 
    # invert standardrized predictions 
    #scaler_X.inverse_transform(dataNorm_X) #+++ To de-standardrized data
    trainPredict = scaler_Y.inverse_transform(trainPredict) #+++ To de-standardrized data
    trainY = scaler_Y.inverse_transform(trainY) 
    predPredict = scaler_Y.inverse_transform(predPredict) 
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

    df_Measures_ALL = pd.DataFrame(index=measureTXT)  
    df_Measures_ALL.index.name = 'Measures'

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

        # os.remove("demofile.txt")

    #--- Save the computed error measures
    ALL_MEASURES.append(df_Measures)

    # --- CSV files 
    df_Measures.to_csv(folderRun+'/'+nameState+'_Measures_Error.csv')

#%% Computing further statistics on measures...
# Compute Overall
df_Overall_Mean = pd.DataFrame(index=ALL_MEASURES[0].index, data=np.dstack(ALL_MEASURES).mean(axis=2), columns=ALL_MEASURES[0].columns)  
df_Overall_STD = pd.DataFrame(index=ALL_MEASURES[0].index, data=np.dstack(ALL_MEASURES).std(axis=2), columns=ALL_MEASURES[0].columns)  
df_Total_Mean = pd.DataFrame(index=df_Overall_Mean.index, data=df_Overall_Mean.mean(axis=1), columns=['Total_Mean_Error_Dataset']) 
df_Total_STD = pd.DataFrame(index=df_Overall_STD.index, data=df_Overall_STD.mean(axis=1), columns=['Total_STD_Error_Dataset']) 

# Save Data
folderResult = pathFolderSaved + 'Overall_Results'
os.mkdir(folderResult)
df_Overall_Mean.to_csv(folderResult + '/Overall_Mean.csv')  
df_Overall_STD.to_csv(folderResult + '/Overall_STD.csv')  
df_Total_Mean.to_csv(folderResult + '/Total_Mean.csv')  
df_Total_STD.to_csv(folderResult + '/Total_STD.csv')

#%% Timing
print("\n TIME: "+str((time.time() - start_time))+" Seg ---  "+str((time.time() - start_time)/60)+" Min ---  "+str((time.time() - start_time)/(60*60))+" Hr ")
