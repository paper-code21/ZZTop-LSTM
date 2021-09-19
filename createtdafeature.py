# To create a vector based on sliding-window information from a 'whole' zigzag persistence
# Ignacio Segovia-Dominguez
# Tian Jiang


#%% Libraries
import numpy as np 
import pandas as pd
import time   
from sklearn.neighbors import DistanceMetric
from sklearn.metrics.pairwise import cosine_similarity
import os

start_time = time.time() # To measure time

#%% Parameters
# Maximum Dimension of Holes (It means.. 0 and 1)
maxDimHoles = 2 
## To choose the dimension of the topological summary; e.g. 0-dimensional, 1-dimensional, 2-dimensional...
#dimHomology = 0 
# Windows size, it should be  >= 1
sizeWindow = [4, 3, 5, 23]
# Total of graphs in the dynamic network
NGraphs = 437

# size of sliding windows


#%% Open data
nom_ZigzagBarcode = 'DYN_NET/IMGS/matBarcodeTOTAL.txt'
df_ZigzagBarcode = pd.read_csv(nom_ZigzagBarcode, sep=' ', header=None) # List of Counties
df_ZigzagBarcode.columns=['Dimension', 'Birth', 'Death']

#%% Topological feature folder
folderTOPO = 'TOPO_FEAT'
os.mkdir(folderTOPO) 

namesTopoF=["BirthMean", "BirthSTD", "DeathMean", "DeathSTD","LifeMean","LifeSTD","NumBar","NumPrev","NumNext",
            "DistQ0", "DistQ1", "DistQ2", "DistQ3", "DistQ4", "CosSimQ0", "CosSimQ1", "CosSimQ2", "CosSimQ3", "CosSimQ4"]


for NWin in sizeWindow:
    for dimHomology in range(0,maxDimHoles+1):
        if dimHomology == maxDimHoles:
            #%% Select Zigzag Barcodes
            df_ZZBC = df_ZigzagBarcode.copy()
            selZZBC = df_ZZBC.to_numpy()
            # take two columns of Birth and Death
            selZZBC = selZZBC[:,1:3]

            ##### proportion of persistant pattern len = 1
            #%% To Extract Zigzag persistence from each window
            all_Windows = []
            all_BCW = [] # To save all barcodes
            for iw in range(NGraphs-NWin+1):
                selRows = (selZZBC[:,1]>=iw) & (selZZBC[:,0]<=(iw+NWin-1)) # Barcodes in the range
                BCW = selZZBC[selRows,:] # To save window's barcode
                # Values' correction (shift to [0,1])
                BCW = BCW - iw 
                BCW[BCW[:,0]<0, 0] = 0 # Assign zeros in lower limit
                BCW[BCW[:,1]>(NWin-1), 1] = NWin-1 # Assign zeros in upper limit
                # To save Barcodes of each window
                all_BCW.append(BCW)
                # To save the windows
                all_Windows.append([iw, iw+NWin-1])
                # To print some results
                print('Window: *** '+str(iw)+', '+str(iw+NWin-1)+' ***')
                print(BCW)

            #%% --- To Build Vectors (OPTION 1)
            # --- COMMENT OR UNCOMMENT NEXT CODE...
            mat_TopoF = []
            for kw in range(len(all_BCW)):
                if(all_BCW[kw].shape[0]==0): # Empty barcode
                    vecRow = [0] * (9+5) + [1]*5
                else: # Barcode with Bars 
                    vecRow = [] 
                    # Computing on Births
                    vecRow.append( np.mean(all_BCW[kw][:,0]) ) # Mean 
                    vecRow.append( np.std(all_BCW[kw][:,0]) ) # STD
                    # Computing on Deaths
                    vecRow.append( np.mean(all_BCW[kw][:,1]) ) # Mean 
                    vecRow.append( np.std(all_BCW[kw][:,1]) ) # STD
                    # Computing on length of Bars
                    auxRange = np.ptp(all_BCW[kw], axis=1)
                    vecRow.append( np.mean(auxRange) ) # Mean
                    vecRow.append( np.std(auxRange) ) # STD 
                
                    # Others
                    vecRow.append( all_BCW[kw].shape[0] ) # Number of bars 
                    vecRow.append( sum(all_BCW[kw][:,0] == 0) ) # Number of bars from previous
                    vecRow.append( sum(all_BCW[kw][:,1] == NWin-1) ) # Number of bars to next

                    distOBJ = DistanceMetric.get_metric('euclidean') 
                    vecDis = distOBJ.pairwise(all_BCW[kw])[np.triu_indices(all_BCW[kw].shape[0], k=1)]
                    vecDisCos = cosine_similarity(all_BCW[kw])[np.triu_indices(all_BCW[kw].shape[0], k=1)]
                    
                    if (vecDis.shape[0]==0): # There are not distances
                        vecRow = vecRow + [0,0,0,0,0, 1,1,1,1,1]
                    else:
                        vecRow = vecRow + np.quantile(vecDis, [0.0, 0.25, 0.5, 0.75, 1.0]).tolist()
                        vecRow = vecRow + np.quantile(vecDisCos, [0.0, 0.25, 0.5, 0.75, 1.0]).tolist()

                # To save converted topological features of each window
                mat_TopoF.append( vecRow )
        

            #%% To save topological features in a file 
            namesTopoFdim = [x + "_ALL" for x in namesTopoF ]
            df_TopoF = pd.DataFrame(data=mat_TopoF, columns = namesTopoFdim)
            df_TopoF.to_csv(folderTOPO + '/TopoFeatures_dimALL_win' + str(NWin) + '.csv', index=False)
        else:
            #%% Select Zigzag Barcodes
            df_ZZBC = df_ZigzagBarcode[df_ZigzagBarcode['Dimension']==dimHomology].copy()
            selZZBC = df_ZZBC.to_numpy()
            # take two columns of Birth and Death
            selZZBC = selZZBC[:,1:3]

            ##### proportion of persistant pattern len = 1
            #%% To Extract Zigzag persistence from each window
            all_Windows = []
            all_BCW = [] # To save all barcodes
            for iw in range(NGraphs-NWin+1):
                selRows = (selZZBC[:,1]>=iw) & (selZZBC[:,0]<=(iw+NWin-1)) # Barcodes in the range
                BCW = selZZBC[selRows,:] # To save window's barcode
                # Values' correction (shift to [0,1])
                BCW = BCW - iw 
                BCW[BCW[:,0]<0, 0] = 0 # Assign zeros in lower limit
                BCW[BCW[:,1]>(NWin-1), 1] = NWin-1 # Assign zeros in upper limit
                # To save Barcodes of each window
                all_BCW.append(BCW)
                # To save the windows
                all_Windows.append([iw, iw+NWin-1])
                # To print some results
                print('Window: *** '+str(iw)+', '+str(iw+NWin-1)+' ***')
                print(BCW)

            #%% --- To Build Vectors (OPTION 1)
            # --- COMMENT OR UNCOMMENT NEXT CODE...
            mat_TopoF = []
            for kw in range(len(all_BCW)):
                if (all_BCW[kw].shape[0]==0): # Empty barcode
                    vecRow = [0] * (9+5) + [1]*5
                else: # Barcode with Bars 
                    vecRow = [] 
                    # Computing on Births
                    vecRow.append( np.mean(all_BCW[kw][:,0]) ) # Mean 
                    vecRow.append( np.std(all_BCW[kw][:,0]) ) # STD
                    # Computing on Deaths
                    vecRow.append( np.mean(all_BCW[kw][:,1]) ) # Mean 
                    vecRow.append( np.std(all_BCW[kw][:,1]) ) # STD
                    # Computing on length of Bars
                    auxRange = np.ptp(all_BCW[kw], axis=1)
                    vecRow.append( np.mean(auxRange) ) # Mean
                    vecRow.append( np.std(auxRange) ) # STD                
                
                    # Others
                    vecRow.append( all_BCW[kw].shape[0] ) # Number of bars | Number of 2D points  
                    vecRow.append( sum(all_BCW[kw][:,0] == 0) ) # Number of bars from previous
                    vecRow.append( sum(all_BCW[kw][:,1] == NWin-1) ) # Number of bars to next
                    # vecRow.append( sum(all_BCW[kw][:,0] == 0) / all_BCW[kw].shape[0]  ) # Percentage of bars from previous
                    # vecRow.append( sum(all_BCW[kw][:,1] == NWin-1) / all_BCW[kw].shape[0]  ) # Percentage of bars to next

                    distOBJ = DistanceMetric.get_metric('euclidean') 
                    vecDis = distOBJ.pairwise(all_BCW[kw])[np.triu_indices(all_BCW[kw].shape[0], k=1)]
                    vecDisCos = cosine_similarity(all_BCW[kw])[np.triu_indices(all_BCW[kw].shape[0], k=1)]
                    
                    if (vecDis.shape[0]==0): # There are not distances
                        vecRow = vecRow + [0,0,0,0,0, 1,1,1,1,1]
                    else:
                        vecRow = vecRow + np.quantile(vecDis, [0.0, 0.25, 0.5, 0.75, 1.0]).tolist()
                        vecRow = vecRow + np.quantile(vecDisCos, [0.0, 0.25, 0.5, 0.75, 1.0]).tolist()                      

                # To save converted topological features of each window
                mat_TopoF.append( vecRow )

            #%% To save topological features in a file 
            namesTopoFdim = [x + "_" + str(dimHomology) for x in namesTopoF ]
            df_TopoF = pd.DataFrame(data=mat_TopoF, columns = namesTopoFdim)
            df_TopoF.to_csv(folderTOPO + '/TopoFeatures_dim' + str(dimHomology) + '_win' + str(NWin) + '.csv', index=False)