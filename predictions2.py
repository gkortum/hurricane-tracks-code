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



#import statistical and machine learning libraries
import statsmodels.api as sm

import sklearn
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import scipy.interpolate as interp




from datetime import datetime
from datetime import timedelta

# In[52]:

def get_hi_low_bestdfs(df5,X2):
    hibestdfs=[]
    hibestX2s=[]
    lowbestdfs=[]
    lowbestX2s=[]
    hibestenses=[]
    lowbestenses=[]
    for b in range(0,100,1):
   
    #     if b==1: 
    #         thestring=''
    #     else: 
        thestring= '_'+str(b)

        with open('data/bootstrap ensembles 3/lowbestens'+thestring, 'rb') as f:
            lowbestens = pickle.load(f)

        with open('data/bootstrap ensembles 3/hibestens'+thestring, 'rb') as f:
            hibestens= pickle.load(f)
        hibestenses.append(hibestens)
        lowbestenses.append(lowbestens)
        hibestdfs.append(getdf(hibestens,df5,X2)[0])
        lowbestdfs.append(getdf(lowbestens,df5,X2)[0])
        hibestX2s.append(getdf(hibestens,df5,X2)[1])
        lowbestX2s.append(getdf(lowbestens,df5,X2)[1])
    return hibestenses,lowbestenses,hibestdfs,lowbestdfs,hibestX2s,lowbestX2s


def date_to_nth_day(dfDate):
        date = pd.to_datetime(dfDate, format=format)
        daysyr=[]
        for i in range(date.shape[0]):
            theDate=date.iloc[i]
            new_year_day = pd.Timestamp(year=theDate.year, month=1, day=1)
            daysyr.append((theDate - new_year_day).days + 1)
        return daysyr


    
def get_length_model(df5):
    
    count =0
    for i in df5.groupby(['ensemble2','year2','zSTORM12']).first()['XCOUNT2'].values: 
        if i !=1:
#             print(count)
            break
        count +=1
    lengthsFull = df5.groupby(['ensemble2','year2','zSTORM12']).count().reset_index()
    
    
    lengths2= lengthsFull['lon2'].values
    
    maxwind = df5.groupby(['ensemble2','year2','zSTORM12']).max()['wind2'].values

    genlons = df5[df5['XCOUNT2']==1].groupby(['ensemble2','year2','zSTORM12']).mean()['lon2'].values
    genlats = df5[df5['XCOUNT2']==1].groupby(['ensemble2','year2','zSTORM12']).mean()['lat2'].values


    genmonths = df5[df5['XCOUNT2']==1].groupby(['ensemble2','year2','zSTORM12']).mean()['month2'].values
    gendays = df5[df5['XCOUNT2']==1].groupby(['ensemble2','year2','zSTORM12']).mean()['day2'].values
    genyears = df5[df5['XCOUNT2']==1].groupby(['ensemble2','year2','zSTORM12']).mean()['year'].values
    dfDate = pd.DataFrame({'year': genyears,'month': genmonths,'day': gendays})
    lb = preprocessing.LabelBinarizer()
    lb.fit_transform(range(1,13,1))

    daysofyr=date_to_nth_day(dfDate)
    Xgen=np.array([genlons,genlats, genlons*genlats, genlons*genlons, genlats*genlats]).T
#     Xgen=np.concatenate((np.array([genlons,genlats, genlons*genlats, genlons*genlons, genlats*genlats]), lb.transform(genmonths).T)).T




    Xgen2 = sm.add_constant(Xgen)


    modelLength2 = sm.OLS(lengths2,Xgen2).fit(cov_type='HC0')

    preds2=modelLength2.predict(Xgen2)


    Lengths = pd.DataFrame(np.asarray([lengths2,preds2]).T)
    Lengths.columns=['true length', 'predlength']
    Lengths['meanlength']=np.mean(lengths2)*np.ones(len(lengths2))
    Lengths['prederr']=np.square(Lengths['true length'].values-Lengths['predlength'].values)
    Lengths['meanerr']=np.square(Lengths['true length'].values-Lengths['meanlength'].values)
    Lengths['genlat']=genlats
    Lengths['genlon']=genlons
    Lengths['pred2']=preds2
    Lengths['prederr2']=np.square(Lengths['true length'].values-Lengths['pred2'].values)
    return modelLength2, Lengths, lengths2, genlons, genlats, genmonths
def get_meansW(df5):


    prevzstorm=-100
    prevyr = 1970
    allwind = []

    years=[]
    ens=[]
    for i in range(df5.shape[0]):
        if ((df5['zSTORM1'].values[i]!=prevzstorm)or (df5['year'].values[i]!=prevyr)): 
#             if not first time 
            if prevzstorm !=-100:
           
                winds = np.asarray(winds)
                try: 
                    winds_interp = interp.interp1d(np.arange(winds.size),winds)
                    winds_compress = winds_interp(np.linspace(0,winds.size-1,50))
                    allwind.append(winds_compress)


                except ValueError:
                    winds_compress = np.ones(50)* winds[0]
                    allwind.append(winds_compress)
                   
                years.append(year)
                ens.append(ensemble)
            

            year = df5['year2'].values[i]
            ensemble =df5['ensemble2'].values[i]
            prevzstorm = df5['zSTORM12'].values[i]
            prevyr =year
            winds = []
          

        winds.append(df5['wind2'].values[i])



    df = pd.DataFrame(np.asarray([years, ens,allwind]).T)
    df.columns = ['years', 'ens','allwind' ]
    data_proper = df['allwind'].apply(pd.Series)
    df_new = pd.concat([df.drop('allwind',axis=1), data_proper], axis=1)
    windmean = df_new.mean()

    return windmean.values[2:]


    



def get4xds(theTime, e):
   
    year = theTime.year
    if year ==2022: 
        year = 2021
    path1 = 'data/SLP_NA_only/'
    path2 = '/tiger/scratch/gpfs/gkortum/SLP_data/'
    if year ==2021: 
        exp = 'amipHadISSTlong_chancorr_tigercpu_intelmpi_18_1080PE_extend2021'
    elif year ==2020: 
        exp = 'amipHadISSTlong_chancorr_tigercpu_intelmpi_18_1080PE_extend2020'
    else: 
        exp = 'amipHadISSTlong_chancorr_tigercpu_intelmpi_18_1080PE'
    number = e+1
    if number ==11: 
        number =10
    if number ==10: 
        try: 
            SLPds= xr.open_dataset(path1+'MODEL_OUT_AM2.5C360_'+exp+'_en'+str(number)+'_POSTP_'+str(int(year))+'0101.atmos_4xdailyNA.nc').slp.loc[theTime]
        except: 
            SLPds= xr.open_dataset(path2+'MODEL_OUT_AM2.5C360_'+exp+'_en'+str(number)+'_POSTP_'+str(int(year))+'0101.atmos_4xdailyNA.nc').slp.loc[theTime]

    else: 
        try: 
            SLPds= xr.open_dataset(path1+'MODEL_OUT_AM2.5C360_'+exp+'_en0'+str(number)+'_POSTP_'+str(int(year))+'0101.atmos_4xdailyNA.nc').slp.loc[theTime]
        except: 
            SLPds= xr.open_dataset(path2+'MODEL_OUT_AM2.5C360_'+exp+'_en0'+str(number)+'_POSTP_'+str(int(year))+'0101.atmos_4xdailyNA.nc').slp.loc[theTime]
    ygrad = np.true_divide(SLPds.differentiate('grid_yt'), 111000)
    ygrad = ygrad.rolling(grid_yt= 20, 
                          min_periods = 2,
                          center = True).mean().dropna("grid_yt").rolling(grid_xt= 40,
                             min_periods = 2,center = True).mean().dropna("grid_xt")

    xgrad = np.true_divide(SLPds.differentiate('grid_xt'),np.cos(np.radians(np.asarray(SLPds['grid_yt']))).reshape([360,1])*111000)
    xgrad = xgrad.rolling(grid_yt= 20, 
                          min_periods = 2,
                          center = True).mean().dropna("grid_yt").rolling(grid_xt= 40,
                             min_periods = 2,center = True).mean().dropna("grid_xt")
    DS=xr.Dataset({'SLP': SLPds, 'SLPxgrad':xgrad, 'SLPygrad': ygrad})
    return DS



# In[36]:




def getdf(hibestens,df5,X2):
    for i in range(len(hibestens)): 
        year = 1970+i
        ensemble = int(hibestens[i])
        if i ==0:
            hibestdf = df5[(df5['year']==year)&(df5['ensemble']==ensemble)]
            hibestX2 = X2[(df5['year']==year)&(df5['ensemble']==ensemble)]
        else: 
            hibestdf = hibestdf.append(df5[(df5['year']==year)&(df5['ensemble']==ensemble)])
            hibestX2 = hibestX2.append(X2[(df5['year']==year)&(df5['ensemble']==ensemble)])

    hibestdf=hibestdf.reset_index().drop('index',axis =1)   
    return hibestdf,hibestX2





# In[50]:


# with open('ensemblesSLP', 'rb') as f:
#     ensemblesSLP = pickle.load(f)
    
    
# with open('/tigress/gkortum/ThesisCurrent/lowbestens', 'rb') as f:
#     lowbestens = pickle.load(f)

# with open('/tigress/gkortum/ThesisCurrent/hibestens', 'rb') as f:
#     hibestens= pickle.load(f)
# df5 = pd.read_csv('df54xallens_lysisconstraint.csv')    
    
    


# with open('X2_4xSLP', 'rb') as f:
#     X2 = pickle.load(f)
# with open('X1_4xSLP', 'rb') as f:
#     X1 = pickle.load(f)
# with open('modelLat', 'rb') as f:
#     modelLat= pickle.load(f)
# with open('modelLon', 'rb') as f:
#     modelLon = pickle.load(f)    
    
# with open('modelLength', 'rb') as f:
#     modelLength= pickle.load(f)
# with open('windmean2', 'rb') as f:
#     windmean2= pickle.load(f)
    
# hibestdf,hibestX2= getdf(hibestens)
# lowbestdf,lowbestX2= getdf(lowbestens)


# # recursive prediction- normal

# In[2]:


#predict tracks based off of model and x data using the first two points of each storm only

def genpreds(X1,ensemblesSLP,fourxSLP,i,X_pred,currlon,currlat,modelLat,modelLon,datasetsSLP,df5,startTime=0,lengthGiven=True,constSLP=False,winds_compress=[], setyear=0,constGen=False,y=0,setens = 0, ensemble =0, bootstrapensembles =[]):

    lon2 = X_pred[-1][2]
    lat2= X_pred[-1][0]

    a = np.asarray(X_pred[-1]).reshape(1,-1)

    alat = np.asarray([a[0][2],a[0][4],a[0][5],a[0][6]]).reshape(1,-1)
    alon = np.asarray([a[0][2],a[0][4],a[0][5],a[0][7]]).reshape(1,-1)

    lat = modelLat.predict(alat)[0]
    lon = modelLon.predict(alon)[0]
    currlon = currlon + lon/(math.cos(math.radians(currlat)))
    currlat = currlat + lat

    sinlat = math.sin(math.radians(currlat))
    coslat = math.cos(math.radians(currlat))
    if lengthGiven: 
        wind= X1['wind'].values[i]
    else: 
        wind=winds_compress[i]
    windbeta = (wind*coslat- X1['windbeta'].mean())/X1['windbeta'].std()
    wind2beta = (wind*wind*coslat- X1['wind2beta'].mean())/X1['wind2beta'].std()


    if constSLP:

        theYear=setyear
        delta = timedelta(hours = 6*i)
        theMonth =(startTime+delta).month
        theDay =(startTime+delta).day
        theHour =(startTime+delta).hour
        theEnsemble = setens



    elif lengthGiven:
        if constGen:
            theYear = y
            n = int(y - 1970)
            theEnsemble = int(bootstrapensembles[n])
            
        else:
            theYear = df5['year'].values[i]
            theEnsemble = int(df5['ensemble'].values[i])
        theMonth = df5['month'].values[i]
        theHour = df5['hour'].values[i]
        theDay = df5['day'].values[i]
        
    else: 
        delta = timedelta(hours = 6*i)
        if constGen:
            theYear = y
            n = int(y - 1970)
            theEnsemble = int(bootstrapensembles[n])
                 
            
        else:
            theYear = (startTime+delta).year
            theEnsemble = int(ensemble)

        theMonth =(startTime+delta).month
        theDay =(startTime+delta).day
        theHour =(startTime+delta).hour
      

        
        
    if not fourxSLP:   
        theDay = 15
        theHour=12

    
    
    theTime = cftime.DatetimeJulian(int(theYear),int(theMonth),int(theDay),int(theHour))
    m = int(theYear - 1970)
    if (currlon>360):
        theLon = currlon-360
    else: theLon = currlon
        
    if not fourxSLP:
        SLPgradx = datasetsSLP[m]['SLPxgrad'].sel(grid_yt=currlat, grid_xt=theLon, time = theTime, method='nearest').values/sinlat
        SLPgrady = datasetsSLP[m]['SLPygrad'].sel(grid_yt=currlat, grid_xt=theLon, time = theTime, method='nearest').values/sinlat
    else: 
        datasetsSLP = get4xds(theTime,theEnsemble)

        SLPgradx = datasetsSLP['SLPxgrad'].sel(grid_yt=currlat, grid_xt=theLon,method='nearest').values
        SLPgrady = datasetsSLP['SLPygrad'].sel(grid_yt=currlat, grid_xt=theLon, method='nearest').values
  
    SLPx = ((SLPgradx- X1['SLPgradxdivide'].mean())/X1['SLPgradxdivide'].std())
    SLPy = ((SLPgrady- X1['SLPgradydivide'].mean())/X1['SLPgradydivide'].std())


    toAppend = np.asarray([1,lat, coslat, lon, windbeta,wind2beta, SLPx, SLPy])

    return toAppend,currlat,currlon






def predictLoop(X1,ensemblesSLP,X2,modelLon,modelLat,df5,fourxSLP=False, bootstrapensembles=[], constSLP=False, bootSLP=False,setens=0,setyear =0,lengthGiven=True,modelLength = 0,windmean=[],niterations=0,constGen=False):
    #list of all predicted points
    X_pred = []
    currlats = []
    currlons = []
    curryears=[]
    curryears2=[]
    currzStorm1s = []
    currEns = []
#     if lengthGiven ==False: 
#         winds_interp = scipy.interpolate.interp1d(np.arange(windmean.size),windmean)
    if constGen: 
        theRange=range(1970,2022,1)
        
    if not constGen: theRange=range(1)
    for y in theRange: 
        print(y)
        stormNum = 0
        print(niterations)
        for i in range(niterations):
            
            print(i)
#             print(i)
            # new storm genesis scenario
            if df5['XCOUNT'].values[i]==df5['XCOUNT'][0]:
                firstPoint = list(X2.iloc[i,:].values)
                X_pred.append(firstPoint)
                currlat=df5['lat'].values[i]
                currlon = df5['lon'].values[i]
                currlats.append(currlat)
                currlons.append(currlon)
                curryears.append(y)
                currEns.append(df5['ensemble'].values[i])
                
                startTime = cftime.DatetimeJulian(int(df5['year'].values[i]),int(df5['month'].values[i]),int(df5['day'].values[i]),int(df5['hour'].values[i]))
                if len(curryears2)>0: 
                    if startTime.year != curryears2[-1]:
                        stormNum= 0
                curryears2.append(startTime.year)
                ensemble = int(df5['ensemble'].values[i])
                stormNum+=1
                currzStorm1s.append(stormNum)
                if constSLP:
                    if not fourxSLP:
                    
                        datasetsSLP = ensemblesSLP[setens]
                    elif fourxSLP: datasetsSLP=[]
                else:
                    if not bootSLP:
                        if not fourxSLP:
                            datasetsSLP = ensemblesSLP[ensemble]
                        else: 
                            datasetsSLP=[]
                    else: 
                        if not constGen:
                            n = int(startTime.year - 1970)
                        elif constGen:
                            n = int(y - 1970)
                        m = int(bootstrapensembles[n])
                        if not fourxSLP:
                            datasetsSLP = ensemblesSLP[m]
                        else: 
                            datasetsSLP=[]
                if lengthGiven ==False: 
                    Xgen=np.array([1,currlon,currlat, currlon*currlat, currlon*currlon, currlat*currlat]).T
                    thelength = np.maximum(4,modelLength.predict(Xgen))
                    winds_interp = interp.interp1d(np.arange(windmean.size),windmean)
                    winds_compress = winds_interp(np.linspace(0,windmean.size-1,int(thelength)))

             

                    for w in range(1,int(thelength),1):
                        toAppend,currlat,currlon=genpreds(X1,ensemblesSLP,fourxSLP,w,X_pred,currlon,currlat,modelLat,modelLon,datasetsSLP,df5,startTime,lengthGiven,constSLP,winds_compress,setyear,constGen, y,setens, ensemble, bootstrapensembles)
                        currlats.append(currlat)
                        currlons.append(currlon)
                        curryears.append(y)
                        curryears2.append(startTime.year)
                        X_pred.append(list(toAppend))
                        currzStorm1s.append(stormNum)
                        currEns.append(df5['ensemble'].values[i])


            else: 
                if lengthGiven ==False: 
                    continue
                else:

                    toAppend,currlat,currlon=genpreds(X1,ensemblesSLP,fourxSLP,i,X_pred,currlon,currlat,modelLat,modelLon,datasetsSLP,df5,startTime,lengthGiven,constSLP,[],setyear,constGen,y,setens,ensemble, bootstrapensembles)
                    currlats.append(currlat)
                    currlons.append(currlon)
                    curryears.append(y)
                    curryears2.append(df5['year'].values[i])
                    currEns.append(df5['ensemble'].values[i])
                    currzStorm1s.append(stormNum)
                    X_pred.append(list(toAppend))
    return X_pred, currlats,currlons,curryears,curryears2,currzStorm1s,currEns


def predictPoly2(X1,ensemblesSLP, X2,modelLon,modelLat,modelLength,df5,fourxSLP = False, bootstrapensembles=[], constSLP=False, bootSLP=False,setens=0,setyear =0,lengthGiven=True,windmean=[],niterations=0,constGen=False):
        
   

        X_pred, currlats,currlons,curryears,curryears2,currzStorm1s,currEns=predictLoop(X1,ensemblesSLP,X2,modelLon,modelLat,df5,fourxSLP, bootstrapensembles, constSLP, bootSLP,setens,setyear,lengthGiven,modelLength,windmean,niterations,constGen)
            
            #case of the first 'predicted' point in a storm - it is taken as given
  
        
        #return a dataframe of predicted tracks generated based off the model
        predicted = pd.DataFrame(X_pred)
        predicted.columns =X2.columns
        if constGen:
            predicted['year']=curryears
#             predicted['ensemble']=np.ones(len(curryears))*df5['ensemble'].values[0]
        else:
            predicted['year']= curryears2
#             predicted['ensemble']=df5['ensemble'].values[0:predicted.shape[0]]
        predicted['currentlat']=currlats
        predicted['currentlon']=currlons
        predicted['zSTORM1']=currzStorm1s
        predicted['ensemble']=currEns
    
        return predicted





# plot_tracks(predicted, df5)


# In[57]:


# ens=1
# year = 1972
# df52=df5[(df5['ensemble']==ens) & (df5['year']==year)].reset_index().drop('index', axis = 1)
# X22=X2[(df5['ensemble']==ens) & (df5['year']==year)].reset_index().drop('index', axis = 1)

# predictedgenconst=predictPoly2(X22,modelLon,modelLat,modelLength,df52,fourxSLP=False, bootstrapensembles=hibestens, constSLP=False, bootSLP=True,setens=0,setyear =0,lengthGiven=False, windmean=windmean2,niterations=df52.shape[0],constGen=True)
    


# In[ ]:


# niterations=X2.shape[0]




# predmeans = modelLon.predict([1, X2['coslat'].mean(),
#                               X2['londiffolder'].mean(), 
                               
#                               X2['windbeta'].mean(), 
#                               X2['wind2beta'].mean(),
#                               X2['SLPgrady'].mean()])



# predmeans = modelLat.predict([1, 
#                               X2['latdiffolder'].mean(), 
#                                X2['coslat'].mean(),
#                               X2['windbeta'].mean(), 
#                               X2['wind2beta'].mean(),
#                               X2['SLPgradx'].mean()])

# predmeansSTD= modelLon.predict([1, X2['coslat'].mean()+X2['coslat'].std(), 
#                             X2['londiffolder'].mean(),
#                               X2['windbeta'].mean(), 
#                               X2['wind2beta'].mean(),
#                               X2['SLPgrady'].mean()])


# print(predmeansSTD-predmeans)


def get_means2(df):
    prevzstorm=-100
    prevyr = 1970
    allLat = []
    allLon=[]

    years=[]
    ens=[]
    for i in range(df.shape[0]):
        if ((df['zSTORM1'].values[i]!=prevzstorm)or (df['year'].values[i]!=prevyr)): 
#             if not first time 
            if prevzstorm !=-100:
           
                predlats = np.asarray(predlats)
                try: 


                    lats = np.asarray(lats)
                    lats_interp = interp.interp1d(np.arange(lats.size),lats)
                    lats_compress = lats_interp(np.linspace(0,lats.size-1,30))
                    allLat.append(lats_compress)

                    lons = np.asarray(lons)
                    lons_interp = interp.interp1d(np.arange(lons.size),lons)
                    lons_compress = lons_interp(np.linspace(0,lons.size-1,30))
                    allLon.append(lons_compress)
                except ValueError:

                    lats_compress =  np.ones(30)* lats[0]
                    allLat.append(lats_compress)
                    lons_compress =  np.ones(30)* lons[0]
                    allLon.append(lons_compress)
                years.append(year)
                ens.append(ensemble)
            

            year = df['year'].values[i]
            ensemble = df['ensemble'].values[i]
            prevzstorm = df['zSTORM1'].values[i]
            prevyr = df['year'].values[i]
            predlats = []
            predlons=[]
            lats=[]
            lons=[]

        try:
            lats.append(df['lat'].values[i])
            lons.append(df['lon'].values[i])
        except KeyError: 
            lats.append(df['currentlat'].values[i])
            lons.append(df['currentlon'].values[i])  


    df2 = pd.DataFrame(np.asarray([years, ens,allLon]).T)
    df2.columns = ['years', 'ens','allLon' ]
    data_proper = df2['allLon'].apply(pd.Series)
    df_new = pd.concat([df2.drop('allLon',axis=1), data_proper], axis=1)
    lonmean = df_new.groupby(['ens','years']).mean()
    df_lon = df_new

    df2 = pd.DataFrame(np.asarray([years, ens,allLat]).T)
    df2.columns = ['years', 'ens','allLat' ]
    data_proper = df2['allLat'].apply(pd.Series)
    df_new = pd.concat([df2.drop('allLat',axis=1), data_proper], axis=1)
    latmean = df_new.groupby(['ens','years']).mean()
    df_lat = df_new
    return lonmean, latmean,  df_lat, df_lon
    

def get_means(predicted,df5):
    predicted2=predicted[['currentlat','currentlon','year','ensemble', 'zSTORM1']]
#     df5=df5[df5['ensemble'].isin(predicted['ensemble'].unique())].reset_index()
#     predicted2[['XCOUNT','lat','lon','zSTORM1']]= df5[['XCOUNT','lat','lon','zSTORM1']]
    lonmean, latmean,  df_lat, df_lon=get_means2(df5)
    lonPredmean, latPredmean,  df_predlat, df_predlon=get_means2(predicted2)
    return lonmean, latmean, lonPredmean, latPredmean, df_predlat, df_predlon






def plot_tracks3(predicted, df5, theYears, theEnsemble):
    #first ensemble member only 
        lonmean, latmean, lonPredmean, latPredmean, df_predlat, df_predlon  = get_means(predicted,df5)
        fig, a = plt.subplots(ncols=2,nrows=len(theYears),figsize=(18,6*len(theYears)), subplot_kw={'projection': ccrs.PlateCarree(central_longitude=0)}, dpi = 300)
        number = theEnsemble
        count =0
        for year in theYears:
            try: latmean1 = latmean.loc[int(number)].loc[year].values
            except KeyError: 
                continue
            row = count
            count+=1
            

            dfdropyr = df5[df5['year']==year]
            dfdropyr = dfdropyr[dfdropyr['ensemble']==number]

            
            lats = dfdropyr['lat']
            lons = dfdropyr['lon']
            
               
            latmean1 = latmean.loc[int(number)].loc[year].values
     
            lonmean1=lonmean.loc[number].loc[year].values
           #only keep given year and ensemble member data
            b = predicted
            dfpreddropyr= b[b['year']==year]
            dfpreddropyr = dfpreddropyr[dfpreddropyr['ensemble']==number]
            latsp = dfpreddropyr['currentlat']
            lonsp = dfpreddropyr['currentlon']
                  
            latmean2 = latPredmean.loc[int(number)].loc[year].values
     
            lonmean2=lonPredmean.loc[number].loc[year].values



#             differences =[]
#             for m in range(len(latmean2)):
#                 locpred = (latmean2[m],lonmean2[m]) 
#                 locreal = (latmean1[m],lonmean1[m]) 
#                 differences.append(geopy.distance.distance(locpred, locreal).km)
            
#             a[0,1].plot(range(30), differences)
#             a[0,1].set_xlim([0,30])
#             a[0,1].set_ylim([0,1000])
#             a[0,1].set_title('differences')
     


            for first in range(0,len(theYears),1):
                for second in range(0,2,1):
                    a[first,second].set_extent([-110,10,0,70], crs=ccrs.PlateCarree())
  
                    a[first,second].add_feature(cartopy.feature.LAND, facecolor= 'ivory',edgecolor='steelblue')
                    a[first,second].add_feature(cartopy.feature.LAKES, facecolor='aliceblue',edgecolor='lightblue')
                    a[first,second].add_feature(cartopy.feature.RIVERS, edgecolor ='aliceblue')

                    if second ==0:   
                        a[first,second].set_yticks([0,15,30,45,60], crs = ccrs.PlateCarree())
                        lat_formatter = LatitudeFormatter()
                        a[first,second].yaxis.set_major_formatter(lat_formatter)

                    
                    lon_formatter = LongitudeFormatter()
                    a[first,second].xaxis.set_major_formatter(lon_formatter)
#                     if first == len(theYears)-1: 
                    a[first,second].set_xticks([-90, -60,-30,0], crs = ccrs.PlateCarree())

#             a[row,0].set_title('year = ' + str(year)+' dataset = '+str(number))




#             a[0,0].scatter(lons, lats, s= 2)
         
#             a[0,0].scatter(lonmean1, latmean1, c = 'red', s= 4)
#             c = ax.scatter(lonmean1, latmean1, c = 'red')
    #         plt.colorbar(a,axins1, ticks = (10,50))


       
            a[row,0].scatter(lons, lats, s= 6, c = 'midnightblue', label = 'Actual')
         
            a[row,0].scatter(lonsp,latsp, s = 6, c = 'tomato', label = 'Predicted')
            a[row,0].legend(markerscale=4,loc ='upper left')
            a[row,0].set_title(str(year)+' Ensemble '+str(number+1))

            a[row,1].scatter(lonmean1, latmean1, c = 'midnightblue', s= 6, label ='Actual mean')
            a[row,1].scatter(lonmean2, latmean2, c = 'tomato',s= 6, label ='Predicted mean')
            a[row,1].set_title(str(year)+' Ensemble '+str(number+1))

            a[row,1].legend(markerscale=4,loc ='upper left')
        plt.subplots_adjust(hspace = 0.11, wspace=0.07)
        fig.savefig('/tigress/gkortum/thesisSummer/figures/sampletracks2.jpg')

        
def save_track_predictions(X1, ensemblesSLP, X2, modelLon, modelLat, modelLength2, df5, windmean2, SLP4x): 
    if SLP4x: 
        predicted4x = predictPoly2(X1, ensemblesSLP,X2,modelLon,modelLat,modelLength2,df5,fourxSLP = True, bootstrapensembles=[], constSLP=False, bootSLP=False,setens=0,setyear =0,lengthGiven=False,windmean=windmean2,niterations=X2.shape[0],constGen=False)
        predicted4x.to_csv('data/predicted4xnoL3.csv')
        predicted4xL = predictPoly2(X1, ensemblesSLP,X2,modelLon,modelLat,modelLength2,df5,fourxSLP = True, bootstrapensembles=[], constSLP=False, bootSLP=False,setens=0,setyear =0,lengthGiven=True,windmean=windmean2,niterations=X2.shape[0],constGen=False)
        predicted4xL.to_csv('data/predicted4x3.csv')
    else:     
        predictedL = predictPoly2(X1, ensemblesSLP,X2,modelLon,modelLat,modelLength2,df5,fourxSLP = False, bootstrapensembles=[], constSLP=False, bootSLP=False,setens=0,setyear =0,lengthGiven=True,windmean=windmean2,niterations=X2.shape[0],constGen=False) 
        predictedL.to_csv('data/predicted2.csv')

        predicted = predictPoly2(X1, ensemblesSLP,X2,modelLon,modelLat,modelLength2,df5,fourxSLP = False, bootstrapensembles=[], constSLP=False, bootSLP=False,setens=0,setyear =0,lengthGiven=False,windmean=windmean2,niterations=X2.shape[0],constGen=False)
        predicted.to_csv('data/predictednoL2.csv')

def load_track_predictions():
    predictedL = pd.read_csv('data/predicted2.csv') 
    predictedL = predictedL.drop("Unnamed: 0",axis=1)


    predicted4xL = pd.read_csv('data/predicted4x3.csv') 
    predicted4xL = predicted4xL.drop("Unnamed: 0",axis=1)


    predicted = pd.read_csv('data/predictednoL2.csv')
    predicted = predicted.drop("Unnamed: 0",axis=1)





    predicted4x = pd.read_csv('data/predicted4xnoL3.csv')
    predicted4x = predicted4x.drop("Unnamed: 0",axis=1)
    return predictedL, predicted4xL, predicted, predicted4x
#     return predictedL, predicted
        

