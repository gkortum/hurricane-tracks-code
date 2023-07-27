#!/usr/bin/env python
# coding: utf-8

# In[17]:


# General and data processing libraries
import xarray as xr
import numpy as np
from datetime import date
import cftime as cftime
import math
import random
import pandas as pd
import warnings

import pickle


#Plotting and visualization libraries
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.cbook as cbook
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import cartopy.crs as ccrs
import cartopy as cartopy
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

from joblib import Parallel, delayed

#import statistical and machine learning libraries
import statsmodels.api as sm

import sklearn
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import scipy.interpolate as interp

# number of clusters
num_clusters = 4

# number of ensemble members from dataset
num_ensembles = 10

# minimum wind speed threshold for all model ensemble storms 
min_wind = 17

# minimum wind speed threshold for historical dataset storms 
min_wind_hist = 17

# peak wind speed threshold for storms used to define clusters
min_wind2=17

# After clusters are defined, analyze only hurricanes (Peak wind speed > 33)
peak_wind=33

year_start= 1970
year_end = 2021

from datetime import datetime
from datetime import timedelta


import predictions2


import SLP_boots
# hibestenses,lowbestenses,ensemblesSLP = SLP_boots.load_data()

import regression4

# df5_4x = pd.read_csv('data/df5_4x.csv')
# df5_4x=df5_4x.drop("Unnamed: 0",axis=1)
# df5Model_4x = df5_4x[df5_4x['ensemble']<num_ensembles]
# modelLat_4x, modelLon_4x, X2_4x,X1_4x= regression.get_models(df5Model_4x)

# hibestenses_4x,lowbestenses_4x,hibestdfs_4x,lowbestdfs_4x,hibestX2s_4x,lowbestX2s_4x=predictions.get_hi_low_bestdfs(df5Model_4x, X2_4x)
# modelLength2_4x, Lengths_4x, lengths2_4x, genlons_4x, genlats_4x, genmonths_4x= predictions.get_length_model(df5Model_4x)
# windmean2_4x=predictions.get_meansW(df5Model_4x)
# modelLength2_4x.summary()
# modelLength_4x = modelLength2_4x



# predictedL, predicted4xL, predicted, predicted4x= predictions2.load_track_predictions()


# def savepreds_all_2(ens,year,b,hibestdf,hibestX2,lowbestdf,lowbestX2,hibestens,lowbestens):
def savepreds1(df5Model, X2,ensemblesSLP, genEns, genYear,X1, modelLon,modelLat,modelLength,windmean2, fourxSLP): 
    count = 0
    print(genYear)
    df52=df5Model[(df5Model['ensemble']==genEns) & (df5Model['year']==genYear)].reset_index().drop('index', axis = 1)
    X22=X2[(df5Model['ensemble']==genEns) & (df5Model['year']==genYear)].reset_index().drop('index', axis = 1)
    if df52.shape[0]!=0:
#         print("gen ENS = "+str(genEns))
#         print("gen year = "+str(genYear))
        for slpEns in range(0,10,1):
            for slpYear in range(1970,2022,1):

    #                     print("SLP ENS = "+str(slpEns))
    #                     print("SLP year = "+str(slpYear))
    #                     print("Count = "+str(count))

                to_append = predictions2.predictPoly2(X1,ensemblesSLP,X22,modelLon,modelLat,modelLength,df52,fourxSLP,bootstrapensembles=[], constSLP=True, bootSLP=False,setens=slpEns,setyear =slpYear,lengthGiven=False, windmean=windmean2,niterations=df52.shape[0],constGen=False)


                to_append['genEnsemble']= np.ones(to_append.shape[0])*genEns
                to_append['genYear']= np.ones(to_append.shape[0])*genYear
                to_append['slpEnsemble']= np.ones(to_append.shape[0])*slpEns
                to_append['slpYear']= np.ones(to_append.shape[0])*slpYear
                if count ==0:
                    predictedDF=to_append
                else: 
                    predictedDF = predictedDF.append(to_append)

                count +=1
        if fourxSLP ==True: 
            fourx = '_4x'
        else: 
            fourx = ''
        predictedAll = predictedDF.reset_index().drop('index',axis =1)
        predictedAll.to_csv('/tigress/gkortum/thesisSummer/predictions_experiments2/predicted_All_genEns_'+str(genEns)+'_genYear'+str(genYear)+str(fourx)+'.csv')

        return predictedAll





def save_predicted_boots(predictedL, hibestenses,lowbestenses,filepath,filepathlow):
    predictedBoots=[]
    predictedBootslow=[]




    for i in range(100):
        print(i)

        count = 0
        for year in range(1970,2022,1):
            if year ==1970: 
                predictedBoots1=predictedL[predictedL['year']==year][predictedL['ensemble']==hibestenses[i][count]]
                predictedBootslow1=predictedL[predictedL['year']==year][predictedL['ensemble']==lowbestenses[i][count]]

            else: 
                predictedBoots1=predictedBoots1.append(predictedL[predictedL['year']==year][predictedL['ensemble']==hibestenses[i][count]])
                predictedBootslow1=predictedBootslow1.append(predictedL[predictedL['year']==year][predictedL['ensemble']==lowbestenses[i][count]])

            count +=1

        predictedBoots.append(predictedBoots1.reset_index().drop('index',axis =1))
        predictedBootslow.append(predictedBootslow1.reset_index().drop('index',axis =1))





    with open(filepath, 'wb') as f:
        pickle.dump(predictedBoots,f)


    with open(filepathlow, 'wb') as f:
        pickle.dump(predictedBootslow,f)
def load_predAll():
    count = 0
    for i in range(10):
        print(i)
        for year in range(1970,2022,1):
            print(count)
            if count ==0: 
                predAll = pd.read_csv('/tigress/gkortum/thesisSummer/predictions_experiments2/predicted_All_genEns_'+str(i)+'_genYear'+str(year)+'.csv').iloc[:,1:]
            else: 
                try: 
                    predAll = predAll.append(pd.read_csv('/tigress/gkortum/thesisSummer/predictions_experiments2/predicted_All_genEns_'+str(i)+'_genYear'+str(year)+'.csv').iloc[:,1:])
                except FileNotFoundError: 
                    continue
            count +=1
    return predAll
def load_predAll_4x():
    count = 0
    for i in range(10):
        for year in range(1970,2022,1):
            print(count)
            if count ==0: 
                predAll = pd.read_csv('/tigress/gkortum/thesisSummer/predictions_experiments2/predicted_All_genEns_'+str(i)+'_genYear'+str(year)+'_4x.csv').iloc[:,1:]
            else: 
                try: 
                    predAll = predAll.append(pd.read_csv('/tigress/gkortum/thesisSummer/predictions_experiments2/predicted_All_genEns_'+str(i)+'_genYear'+str(year)+'_4x.csv').iloc[:,1:])
                except FileNotFoundError: 
                    try: 
                        predAll = predAll.append(pd.read_csv('/tigress/gkortum/thesisSummer/predictions_experiments2/predicted_All_genEns_'+str(i)+'_genYear'+str(year)+'4x.csv').iloc[:,1:])
                    except FileNotFoundError:
                        continue
            count +=1
    return predAll


def save_prediction_experiments(b, predAll, hibestenses, lowbestenses):
    hiLowEnsdfSLP = pd.DataFrame(np.array([range(1970,2022,1),hibestenses[b],lowbestenses[b]]).T,columns=['slpYear','hibestens','lowbestens'])
    predAll2SLP = pd.merge(predAll,hiLowEnsdfSLP,how="left",on='slpYear')
    
    hiLowEnsdfGen = pd.DataFrame(np.array([range(1970,2022,1),hibestenses[b],lowbestenses[b]]).T,columns=['genYear','hibestens','lowbestens'])
    predAll2Gen = pd.merge(predAll,hiLowEnsdfGen,how="left",on='genYear')    
    
    
    
    
    constSLPlow=predAll2Gen[predAll2Gen['genEnsemble']==predAll2Gen['lowbestens']]
    constSLPhi=predAll2Gen[predAll2Gen['genEnsemble']==predAll2Gen['hibestens']]

    
    constSLPlowdf = constSLPlow[['currentlat','currentlon','genYear','slpEnsemble','slpYear']]
    constSLPlowdf.columns=['lat','lon','year','setens','setyr']
    constSLPhidf = constSLPhi[['currentlat','currentlon','genYear','slpEnsemble','slpYear']]
    constSLPhidf.columns=['lat','lon','year','setens','setyr']

    constGenlow=predAll2SLP[predAll2SLP['slpEnsemble']==predAll2SLP['lowbestens']]
    constGenhi=predAll2SLP[predAll2SLP['slpEnsemble']==predAll2SLP['hibestens']]

    constGenlowdf = constGenlow[['currentlat','currentlon','slpYear','genEnsemble','genYear']]
    constGenlowdf.columns=['lat','lon','year','setens','setyr']
    
    
    constGenhidf = constGenhi[['currentlat','currentlon','slpYear','genEnsemble','genYear']]
    constGenhidf.columns=['lat','lon','year','setens','setyr']
    return constSLPhidf, constSLPlowdf, constGenhidf, constGenlowdf
def save_prediction_experiments_2(predAll, hibestenses, lowbestenses,fourx = False):
    if fourx ==False: 
        SLPconsthiList=[]
        SLPconstlowList=[]
        genconsthiList=[]
        genconstlowList=[]
        for b in range(100):
            print(b)
            constSLPhidf, constSLPlowdf, constGenhidf, constGenlowdf= save_prediction_experiments(b, predAll, hibestenses, lowbestenses)
            SLPconsthiList.append(constSLPhidf)
            SLPconstlowList.append(constSLPlowdf)
            genconsthiList.append(constGenhidf)
            genconstlowList.append(constGenlowdf)
        with open('predictions_experiments2/SLPconsthiList', 'wb') as f:
            pickle.dump(SLPconsthiList,f)  
        with open('predictions_experiments2/SLPconstlowList', 'wb') as f:
            pickle.dump(SLPconstlowList,f)  
        with open('predictions_experiments2/genconsthiList', 'wb') as f:
            pickle.dump(genconsthiList,f)  
        with open('predictions_experiments2/genconstlowList', 'wb') as f:
            pickle.dump(genconstlowList,f)  
    
    else:
        predAll_4x = predAll
        SLPconsthiList4x=[]
        SLPconstlowList4x=[]
        genconsthiList4x=[]
        genconstlowList4x=[]
        for b in range(100):
            print(b)
            constSLPhidf, constSLPlowdf, constGenhidf, constGenlowdf= save_prediction_experiments(b, predAll_4x, hibestenses, lowbestenses)
            SLPconsthiList4x.append(constSLPhidf)
            SLPconstlowList4x.append(constSLPlowdf)
            genconsthiList4x.append(constGenhidf)
            genconstlowList4x.append(constGenlowdf)    

        with open('predictions_experiments2/SLPconsthiList_4x', 'wb') as f:
            pickle.dump(SLPconsthiList4x,f)  
        with open('predictions_experiments2/SLPconstlowList_4x', 'wb') as f:
            pickle.dump(SLPconstlowList4x,f)  
        with open('predictions_experiments2/genconsthiList_4x', 'wb') as f:
            pickle.dump(genconsthiList4x,f)  
        with open('predictions_experiments2/genconstlowList_4x', 'wb') as f:
            pickle.dump(genconstlowList4x,f)  
