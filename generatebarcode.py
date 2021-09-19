# ZIGZAG on DYNAMIC NETWORKS
# -------------------------------------------------------
# This code computes the ZIGZAG persistence diagram on 
# dynamic networks (DN). 
# -------------------------------------------------------
# Ignacio Segovia-Dominguez
# Tian Jiang
# Meichen Huang

### Use dinamic network generate zigzag
##

#%% Libraries
import numpy as np 
import pandas as pd
import time   
import os
import difflib
from sklearn.neighbors import DistanceMetric
from math import radians
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt 
import networkx as nx

import zigzagtools as zzt
from scipy.spatial.distance import squareform
import dionysus as d


start_time = time.time() # To measure time

pathORIG = os.getcwd()
folderResult = pathORIG + '/DYN_NET' 
os.mkdir(folderResult)

#%% Parameters
nom_Weather_NDVI = 'DATA/weather_NDVI_annualyield_2000-2018/mean_weather_and_NDVI.csv'
nom_Locations = 'DATA/Locations/MB_centroid_lat_long.csv'
selColVariables = ['Temp-Max', 'Temp-Min', 'pcp']  

QuanThr_KM = 2/3 # Quantile Threshold for Kilometers
QuanThr_Feature = 2/3 # Quantile Threshold for Distance between features

#%% Open Data
# NDVI
df_WTH_NDVI_Orig = pd.read_csv(nom_Weather_NDVI) # List of Counties
df_WTH_NDVI = df_WTH_NDVI_Orig.copy() 
df_WTH_NDVI.week = pd.to_datetime(df_WTH_NDVI.week, format='%m/%d/%y')
# Locations
df_Location_Orig = pd.read_csv(nom_Locations) # List of Locations
df_Location = df_Location_Orig.copy()

#%% Get DataFrames, each feature
# Split in dataframes
df_ALL = []
for iVar, colVar in enumerate(selColVariables):
    df_ALL.append(df_WTH_NDVI.pivot(index='week', columns='Municipality', values=colVar).copy())

# Split by TimeStamp
# [timestamp, municipality, features] : (437, 37, 3)
# e,g,   dataFeatures[0,:,:]
dataFeatures = np.dstack(df_ALL) 

#%% Geographical Information
# List of Municipalities
vecMunicipality = list(df_ALL[0].columns)
vecCSDNAME = list(df_Location['CSDNAME'])
vecCSDNAME_upper = [aux.upper() for aux in vecCSDNAME] # To Upper Case
# To verify all found elements match
vecREGIONS = [] # To save all regions/municipalities
for txtMUNI in vecMunicipality:
    print('*****   ', txtMUNI, '   *****')
    rowRegion = []
    rowRegion.append(txtMUNI) # The Municipality
    for txtMUNI_split in txtMUNI.split('-'):
        if(txtMUNI_split=='OAKVIEW'): 
            txtFound = difflib.get_close_matches('BLANSHARD', vecCSDNAME_upper)
            print(txtMUNI_split, txtFound)
            rowRegion.append(txtFound[0]) # Adding element
            txtFound = difflib.get_close_matches('SASKATCHEWAN', vecCSDNAME_upper)
            print(txtMUNI_split, txtFound)
            rowRegion.append(txtFound[0]) # Adding element
            txtFound = difflib.get_close_matches('RAPID CITY', vecCSDNAME_upper)
            print(txtMUNI_split, txtFound)
            rowRegion.append(txtFound[0]) # Adding element
        elif(txtMUNI_split=='PRAIRIE LAKES'): 
            txtFound = difflib.get_close_matches('STRATHCONA', vecCSDNAME_upper)
            print(txtMUNI_split, txtFound)
            rowRegion.append(txtFound[0]) # Adding element
            txtFound = difflib.get_close_matches('RIVERSIDE', vecCSDNAME_upper)
            print(txtMUNI_split, txtFound)
            rowRegion.append(txtFound[0]) # Adding element
        elif(txtMUNI_split=='PRAIRIE VIEW'): 
            txtFound = difflib.get_close_matches('BIRTLE', vecCSDNAME_upper)
            print(txtMUNI_split, txtFound)
            rowRegion.append(txtFound[0]) # Adding element
            txtFound = difflib.get_close_matches('MINIOTA', vecCSDNAME_upper)
            print(txtMUNI_split, txtFound)
            rowRegion.append(txtFound[0]) # Adding element
        elif(txtMUNI_split=='SWAN VALLEY WEST'): 
            txtFound = difflib.get_close_matches('SWAN RIVER', vecCSDNAME_upper)
            print(txtMUNI_split, txtFound)
            rowRegion.append(txtFound[0]) # Adding element
            txtFound = difflib.get_close_matches('BENITO', vecCSDNAME_upper)
            print(txtMUNI_split, txtFound)
            rowRegion.append(txtFound[0]) # Adding element
        elif(txtMUNI_split=='WESTLAKE'): 
            txtFound = difflib.get_close_matches('LAKEVIEW', vecCSDNAME_upper)
            print(txtMUNI_split, txtFound)
            rowRegion.append(txtFound[0]) # Adding element
            txtFound = difflib.get_close_matches('WESTBOURNE', vecCSDNAME_upper)
            print(txtMUNI_split, txtFound)
            rowRegion.append(txtFound[0]) # Adding element
        elif(txtMUNI_split=='YELLOWHEAD'): 
            txtFound = difflib.get_close_matches('SHOAL LAKE', vecCSDNAME_upper)
            print(txtMUNI_split, txtFound)
            rowRegion.append(txtFound[0]) # Adding element
            txtFound = difflib.get_close_matches('STRATHCLAIR', vecCSDNAME_upper)
            print(txtMUNI_split, txtFound)
            rowRegion.append(txtFound[0]) # Adding element
        else:
            txtFound = difflib.get_close_matches(txtMUNI_split, vecCSDNAME_upper)
            print(txtMUNI_split, txtFound)
            rowRegion.append(txtFound[0]) # Adding element
    
    # Adding Row
    vecREGIONS.append(rowRegion)

#difflib.get_close_matches(vecMunicipality[0], vecCSDNAME, len(vecCSDNAME), 0)
#difflib.get_close_matches(vecMunicipality[0], vecCSDNAME_upper)

#%% Obtaining Longitude and Latitude
dataLocation = []
for arrRegion in vecREGIONS:
    vecID = []
    for region in arrRegion[1:]:
        vecID.append(vecCSDNAME_upper.index(region))
    #print('********   ', arrRegion[0], '   *********')
    #print(df_Location.iloc[vecID])
    # To save Data Location
    auxAverage = df_Location.iloc[vecID].mean(axis=0)
    dataLocation.append( [arrRegion[0], auxAverage.Longitude, auxAverage.Latitude] )

# Just to put it in a dataframe
df_Average_Location = pd.DataFrame(data=dataLocation, columns=['Region', 'Longitude', 'Latitude'])

#%% Build the Network using 'dataLocation'
# Geographical location
LonLat = df_Average_Location[['Longitude', 'Latitude']].to_numpy()  

#--- Pairwise distance in Kilometers
# Taken from here: # https://stackoverflow.com/questions/19412462/getting-distance-between-two-points-based-on-latitude-longitude  
distOBJ = DistanceMetric.get_metric('haversine')
X = LonLat[:,[1,0]] # Reverse, first latitude, then longitude
R = 6373.0 # Factor...
matDisKM = R*distOBJ.pairwise(X) # Pairwise distance in Kilometers
# Compute threshold based on quantile
km_threshold = np.quantile(np.reshape(matDisKM,(1,-1)), QuanThr_KM)
#(matDisKM<km_threshold).sum(axis=0) # How many for each element

#%% Build the Network using 'dataFeatures' 
dataWeek = []
for idWeek in range(dataFeatures.shape[0]): 
    print('Computing graph of week: '+str(idWeek)+' - Total weeks: '+str(dataFeatures.shape[0]))
    dataWeek.append([idWeek, str(df_ALL[0].index.date[idWeek])])  # To save week 

    #--- Pairwise distance between features
    # Standardized data
    scaler_X = StandardScaler()
    scaler_X.fit(dataFeatures[idWeek,:,:])
    dataNorm_X = scaler_X.transform(dataFeatures[idWeek,:,:])
    # Distance between features 
    distOBJ_feat = DistanceMetric.get_metric('euclidean')
    matDis_Feat = distOBJ_feat.pairwise(dataNorm_X) # Pairwise distance
    # Compute threshold based on quantile
    feat_threshold = np.quantile(np.reshape(matDis_Feat,(1,-1)), QuanThr_Feature)
    #(matDis_Feat<feat_threshold).sum(axis=0) # How many for each element

    # Extract graph connection and Weights
    surv_connect = np.triu( (matDisKM<=km_threshold)&(matDis_Feat<=feat_threshold), k=1 )
    edge_connect = np.transpose(np.nonzero(surv_connect))
    maxFeatDis = np.max(matDis_Feat[surv_connect]) # Max distance (to normalize)
    # Build Dataframe Graph

    dataGraph = []
    for edge in edge_connect:
        dataGraph.append([edge[0]+1, edge[1]+1, matDis_Feat[edge[0], edge[1]]/maxFeatDis])
        # Note we add +1 because we want to start graph indices with 1 (not with 0)
        

    df_Graph = pd.DataFrame(data=dataGraph, columns=['Start','End','Weight'])
    # Save EdgeList
    df_Graph.to_csv(folderResult + '/File' + str(idWeek)+'.csv', index=False, header=False)

    # --- Draw the Graph
    NVertices = LonLat.shape[0]
    plt.figure(num=None, figsize=(5, 5), dpi=80, facecolor='w', edgecolor='k')
    g = nx.Graph()
    MGraph = np.array(dataGraph)
    g.add_nodes_from(list(range(1,NVertices+1))) # Add vertices...
    if(MGraph.ndim==1 and len(MGraph)>0):
        g.add_edge(MGraph[0], MGraph[1], weight=MGraph[2])
    elif(MGraph.ndim==2):
        for k in range(0,MGraph.shape[0]):
            g.add_edge(MGraph[k,0], MGraph[k,1], weight=MGraph[k,2])
    plt.title(str(idWeek)+': '+str(df_ALL[0].index.date[idWeek]))
    pos = nx.circular_layout(g)
    #pos = nx.spring_layout(GraphsNetX[i])
    #pos = nx.spectral_layout(GraphsNetX[i]) 
    nx.draw(g, pos, node_size=15, width=1.5, edge_color='orchid') 
    #nx.draw_circular(GraphsNetX[i], node_size=15, edge_color='r') 
    labels = nx.get_edge_attributes(g, 'weight')
    for lab in labels:
        labels[lab] = round(labels[lab],2)
    nx.draw_networkx_edge_labels(g, pos, edge_labels=labels, font_size=1)

    plt.savefig(folderResult + '/Graph_'+str(idWeek)+'.pdf', bbox_inches='tight')
    plt.close()

# Relationship between idWeek and Date-Week
df_DataWeek = pd.DataFrame(data=dataWeek, columns=['idWeek','Date'])
df_DataWeek.to_csv(folderResult + '/DATE_WEEK.csv', index=False)



#%% Timing
print("\n TIME: "+str((time.time() - start_time))+" Seg ---  "+str((time.time() - start_time)/60)+" Min ---  "+str((time.time() - start_time)/(60*60))+" Hr ")

# %%

#%% #############################

path = os.getcwd()

#%% Parameters 
#nameFolderNet = 'Example8/E6N_All/File'  # Example 1
nameFolderNet = folderResult + '/File'  # Example 2
#nameFolderNet = 'Example8/E6N_2Del/File'  # Example 3
NVertices = 37 # Number of vertices
scaleParameter = 1.0 # Scale Parameter (Maximum) ### Keep it on 1.0 ###
maxDimHoles = 2 # Maximum Dimension of Holes (It means.. 0 and 1)
sizeWindow = 437 # Number of Graphs


folderIMG = 'DYN_NET/IMGS'
os.mkdir(folderIMG)


#%% To measure time
start_time = time.time() 
#for indWin in range(0,TotalNets-sizeWindow+1):
#    print('*************  '+str(indWin)+'  *************')

#%% Open all sets (point-cloud/Graphs)
print("Loading data...") # Beginning
Graphs = []
for i in range(0,sizeWindow):
    #edgesList = np.loadtxt(nameFolderNet+str(i+1)+".txt") # Load data
    edgesList = np.loadtxt(nameFolderNet+str(i)+".csv", delimiter=',') # Load data
    Graphs.append(edgesList)
print("  --- End Loading...") # Ending

#%% Plot Graph
GraphsNetX = []
# #plt.figure(num=None, figsize=(16, 1.5), dpi=80, facecolor='w', edgecolor='k')
for i in range(0,sizeWindow):
    print("Graph" + str(i)) # Ending
    g = nx.Graph()
    g.add_nodes_from(list(range(1,NVertices+1))) # Add vertices...
    if(Graphs[i].ndim==1 and len(Graphs[i])>0):
        g.add_edge(Graphs[i][0], Graphs[i][1], weight=Graphs[i][2])
    elif(Graphs[i].ndim==2):
        for k in range(0,Graphs[i].shape[0]):
            g.add_edge(Graphs[i][k,0], Graphs[i][k,1], weight=Graphs[i][k,2])
    GraphsNetX.append(g)
    plt.subplot(1, sizeWindow, i+1)
    plt.title(str(i))
    pos = nx.circular_layout(GraphsNetX[i])
    nx.draw(GraphsNetX[i], pos, node_size=15, edge_color='r') 
    #nx.draw_circular(GraphsNetX[i], node_size=15, edge_color='r') 
    labels = nx.get_edge_attributes(GraphsNetX[i], 'weight')
    for lab in labels:
       labels[lab] = round(labels[lab],2)
    nx.draw_networkx_edge_labels(GraphsNetX[i], pos, edge_labels=labels, font_size=5)


plt.savefig(folderIMG + '/Graphs.pdf', bbox_inches='tight')
plt.show()

# %% Building unions and computing distance matrices
print("Building unions and computing distance matrices..." + str(iwin))  # Beginning
GUnions = []
MDisGUnions = []
for i in range(0, sizeWindow - 1):
    # --- To concatenate graphs
    unionAux = []
    # if(Graphs[i].ndim==1 and len(Graphs[i])==0): # Empty graph
    #     unionAux = Graphs[i+1]
    # elif(Graphs[i+1].ndim==1 and len(Graphs[i+1])==0): # Empty graph
    #     unionAux = Graphs[i]
    # else:
    #     unionAux = np.concatenate((Graphs[i],Graphs[i+1]),axis=0)
    # --- To build the distance matrix
    MDisAux = np.zeros((2 * NVertices, 2 * NVertices))
    A = nx.adjacency_matrix(GraphsNetX[i]).todense()
    B = nx.adjacency_matrix(GraphsNetX[i + 1]).todense()

    # ----- Version Original (2)
    C = (A + B)/2
    A[A==0] = 1.1
    A[range(NVertices), range(NVertices)] = 0
    B[B==0] = 1.1
    B[range(NVertices), range(NVertices)] = 0
    MDisAux[0:NVertices, 0:NVertices] = A
    C[C==0] = 1.1
    C[range(NVertices), range(NVertices)] = 0
    MDisAux[NVertices:(2 * NVertices), NVertices:(2 * NVertices)] = B
    MDisAux[0:NVertices, NVertices:(2 * NVertices)] = C
    MDisAux[NVertices:(2 * NVertices), 0:NVertices] = C.transpose()

    # Distance in condensed form
    pDisAux = squareform(MDisAux)

    # --- To save unions and distances
    GUnions.append(unionAux)  # To save union
    MDisGUnions.append(pDisAux)  # To save distance matrix
print("  --- End unions...")  # Ending

# %% To perform Ripser computations
print("Computing Vietoris-Rips complexes..." + str(iwin))  # Beginning

GVRips = []
for i in range(0, sizeWindow - 1):
    print(i)
    ripsAux = d.fill_rips(MDisGUnions[i], maxDimHoles, scaleParameter)
    GVRips.append(ripsAux)
print("  --- End Vietoris-Rips computation")  # Ending


# %% Shifting filtrations...
print("Shifting filtrations...")  # Beginning
GVRips_shift = []
GVRips_shift.append(GVRips[0])  # Shift 0... original rips01
for i in range(1, sizeWindow - 1):
    shiftAux = zzt.shift_filtration(GVRips[i], NVertices * i)
    GVRips_shift.append(shiftAux)
print("  --- End shifting...")  # Ending

# %% To Combine complexes
print("Combining complexes..." + str(iwin))  # Beginning
completeGVRips = zzt.complex_union(GVRips[0], GVRips_shift[1])
for i in range(2, sizeWindow - 1):
    completeGVRips = zzt.complex_union(completeGVRips, GVRips_shift[i])
print("  --- End combining")  # Ending

# %% To compute the time intervals of simplices
print("Determining time intervals..." + str(iwin))  # Beginning
time_intervals = zzt.build_zigzag_times(completeGVRips, NVertices, sizeWindow)
print("  --- End time")  # Beginning

# %% To compute Zigzag persistence
print("Computing Zigzag homology..." + str(iwin))  # Beginning
G_zz, G_dgms, G_cells = d.zigzag_homology_persistence(completeGVRips, time_intervals)
print("  --- End Zigzag")  # Beginning

# %% To show persistence intervals
print("Persistence intervals:")
print("++++++++++++++++++++++")
print(G_dgms)
for i, dgm in enumerate(G_dgms):
    print(i)
    for p in dgm:
        print(p)
print("++++++++++++++++++++++")
for i, dgm in enumerate(G_dgms):
    print("Dimension:", i)
    if (i < 2):
        for p in dgm:
            print(p)

for i, dgm in enumerate(G_dgms):
    print("Dimension:", i)
    if (i < 2):
        d.plot.plot_bars(G_dgms[i], show=True)

# %% Personalized plot
for i, dgm in enumerate(G_dgms):
    print("Dimension:", i)
    if (i < 2):
        BCfull = np.zeros((len(dgm), 3)) 
        matBarcode = np.zeros((len(dgm), 2))
        k = 0
        for p in dgm:
            BCfull[k,0] = i
            BCfull[k,1] = p.birth/2
            BCfull[k,2] = p.death/2
            # print("( "+str(p.birth)+"  "+str(p.death)+" )")
            matBarcode[k, 0] = p.birth
            matBarcode[k, 1] = p.death
            k = k + 1
        matBarcode = matBarcode/2  ## final PD ##
        #print(matBarcode)
        for j in range(0, matBarcode.shape[0]):
            plt.plot(matBarcode[j], [j, j], 'b')
            #plt.close()
        #Human readable data
        if(i==0):
            BCALL = BCfull 
        else:
            BCALL = np.concatenate((BCALL, BCfull),axis=0)
        np.savetxt(folderIMG + '/matBarcodeTOTAL_win'+ str(iwin) + '.txt', BCALL) # [n-dimension  t-birth  t-death]
        # my_xticks = [0,1,2,3,4,5,6,7,8,9,10,11]
        # plt.xticks(x, my_xticks)
        plt.xticks(np.arange(sizeWindow))
        plt.grid(axis='x', linestyle='-')
        plt.savefig(folderIMG + '/BoxPlot' + str(i) + '_win'+ str(iwin) + '.pdf', bbox_inches='tight')
        plt.close()
        np.savetxt(folderIMG + '/matBarcode' + str(i) + '_win'+ str(iwin) + '.txt', matBarcode) # for each 'i-dimension' have [t-birth  t-death]

# %%
# Timing
print("RUNNING TIME: "+str((time.time() - start_time))+" Seg ---  "+str((time.time() - start_time)/60)+" Min ---  "+str((time.time() - start_time)/(60*60))+" Hr ")
