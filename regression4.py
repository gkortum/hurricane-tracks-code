import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy as cartopy
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import cftime as cftime
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.gridspec as gridspec
import statsmodels.api as sm
import math
#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'
import matplotlib.dates as mdates
import matplotlib.cbook as cbook
import random
from joblib import Parallel, delayed
import multiprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib
import matplotlib.patches as mpatches
from matplotlib import animation
import pandas as pd
import warnings
import pickle
from datetime import date
# import seaborn as sns
warnings.filterwarnings('ignore')
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.ticker as mticker
import importlib


import statsmodels.api as sm

import sklearn
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import preprocessing 

import scipy.interpolate as interp

warnings.filterwarnings('ignore')
from datetime import timedelta
from joblib import Parallel, delayed
import multiprocessing
num_cores = multiprocessing.cpu_count()



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


def nearest_mean(x_in, y_in, z_in, x_out, y_out, n_nearest=6, decay=4):
    """Given z(x, y), find z(x', y') when x' and y' are not on the x, y grid

    Args:
        x_in (1d or 2d array)
        y_in (1d or 2d array)
        z_in (2d array)
        x_out (1d array)
        y_out (1d array)
        n_nearest (int): number of nearest input grid points to average over; the smaller the faster
        decay (int): exponent over 1/dist^2; the larger the more local

    Returns:
        z_out (1d array)
    """

#     assert len(x_out) == len(y_out)

    small = 1e-9

    xx, yy = np.meshgrid(x_in, y_in)

    weight = 1/((xx - x_out)**2 + (yy - y_out)**2 + small)
#     weights = [1/((xx - x_out[i])**2 + (yy - y_out[i])**2 + small)
#                for i in range(len(x_out))] # weights proportional to 1/dist^2

    idx_large = np.argsort(weight.flatten())[-n_nearest:] # indices of the largest weights

    weights_large = weight.flatten()[idx_large]

    z_out = np.average(z_in.flatten()[idx_large], weights=weights_large**decay)


    return z_out


def addSLPloop(i,df,n): 
#         for i in range(20):
    
    theYear = df['year'].values[i]
    theMonth = df['month'].values[i]
    theDay = df['day'].values[i]
    theHour=df['hour'].values[i]

    theTime = cftime.DatetimeJulian(theYear,theMonth,theDay,theHour)


#     del datasetsSLP
    datasetsSLP = get4xds(theTime,n)
    if (df['lon'].values[i]>360):
        theLon = df['lon'].values[i]-360
    else: theLon = df['lon'].values[i]

#     SLPgradxv = nearest_mean(datasetsSLP.SLPxgrad.grid_xt, 
#                              datasetsSLP.SLPxgrad.grid_yt, 
#                              datasetsSLP.SLPxgrad.values, 
#                              theLon, 
#                              df['lat'].values[i])

#     SLPgradyv = nearest_mean(datasetsSLP.SLPygrad.grid_xt, 
#                              datasetsSLP.SLPygrad.grid_yt, 
#                              datasetsSLP.SLPygrad.values, 
#                              theLon, 
#                              df['lat'].values[i])

    SLPgradxv = datasetsSLP['SLPxgrad'].sel(grid_yt=df['lat'].values[i], grid_xt=theLon,method='nearest').values
    SLPgradyv = datasetsSLP['SLPygrad'].sel(grid_yt=df['lat'].values[i], grid_xt=theLon, method='nearest').values
#     print(str(n)+ ' and '+str(i))
    return [SLPgradyv,SLPgradxv,i]
#             SLPgradx.append(SLPgradxv)

def addSLP4x(merged,n):
    df = merged
    print(df.shape[0])
    df = df.dropna().reset_index()
#     SLPgrads=[]

   
    datasetsSLP = 0
#     print(df.shape[0])

    
    SLPgrads=Parallel(n_jobs=80, verbose = 60)(delayed(addSLPloop)(i,df,n) for i in range(df.shape[0]))

    dftosort = pd.DataFrame(np.array(SLPgrads))
    dftosort.columns = ['SLPgrady','SLPgradx','count']
    dfsorted = dftosort.sort_values(by = 'count').reset_index().drop('index', axis = 1)
    SLPgradx = dfsorted['SLPgradx'].values
    SLPgrady=dfsorted['SLPgrady'].values
    df['SLPgradx']=np.asarray(SLPgradx)
    df['SLPgrady']=np.asarray(SLPgrady)
  

    return df


def addSLPloopmonthly(i,df,n,ensemblesSLP): 
    datasetsSLP = ensemblesSLP[n]
#         for i in range(20):
#     print(df['year'].values)
    theYear = df['year'].values[i]
    if theYear ==2022: 
        theYear =2021
    theMonth = df['month'].values[i]
    theDay = 15
    theHour=12

    theTime = cftime.DatetimeJulian(theYear,theMonth,theDay,theHour)
    m = int(theYear - 1970)
    if (df['lon'].values[i]>360):
        theLon = df['lon'].values[i]-360
    else: theLon = df['lon'].values[i]



    SLPgradxv = datasetsSLP[m]['SLPxgrad'].sel(grid_yt=df['lat'].values[i], grid_xt=theLon, time = theTime, method='nearest').values
    SLPgradyv = datasetsSLP[m]['SLPygrad'].sel(grid_yt=df['lat'].values[i], grid_xt=theLon, time = theTime, method='nearest').values



#     (str(n)+ ' and '+str(i))
    return [SLPgradyv,SLPgradxv,i]
#             SLPgradx.append(SLPgradxv)

def addSLP(merged,n,ensemblesSLP):

    
    df = merged


    df = df.dropna().reset_index()


    SLPgrads = []
    for i in range(df.shape[0]):
        SLPgrads.append(addSLPloopmonthly(i,df,n,ensemblesSLP))
            
#     SLPgrads=Parallel(n_jobs=4,require='sharedmem')(delayed(addSLPloopmonthly)(i,df,n) for i in range(df.shape[0]))

    dftosort = pd.DataFrame(np.array(SLPgrads))
    dftosort.columns = ['SLPgrady','SLPgradx','count']
    dfsorted = dftosort.sort_values(by = 'count').reset_index().drop('index', axis = 1)
    SLPgradx = dfsorted['SLPgradx'].values
    SLPgrady=dfsorted['SLPgrady'].values
    df['SLPgradx']=np.asarray(SLPgradx)
    df['SLPgrady']=np.asarray(SLPgrady)

    return df





min_wind = 17
total_storm_length_thresh = 7
min_wind2=33
class structureData: 
    def __init__(self, dsAll, dsHist,tempdata, ensemblesSLP, allEns=True, slp4x = False):
        #full dataset 
#         self.dsAll = dsAll.where(((dsAll.LON<420)&(dsAll.LON>240)),drop =True)
#         self.dsAllExcl = self.dsAll.where(((self.dsAll.WARM!=-1)&(self.dsAll.WIND > min_wind)), drop = True)

        self.tempdata= tempdata
        self.allEns = allEns
        self.slp4x =slp4x
        
        self.ensemblesSLP=ensemblesSLP
        self.dsAll2 = dsAll.where(((dsAll.LON<420)&(dsAll.LON>240)),drop = True)
        
#         self.dsAll2 = self.dsAll1.where(self.dsAll1.LAT<40)
        self.dsAll2excl= self.dsAll2.where((self.dsAll2.WARM!=-1)&(self.dsAll2.WIND > min_wind))
        self.dsAll2excl33 = self.dsAll2.where((self.dsAll2.WARM!=-1)&(self.dsAll2.WIND>min_wind2))
      
        self.dsHist= dsHist
        self.dsHistexcl= dsHist.where((self.dsHist.WIND > min_wind))
        self.dsHistexcl33 = dsHist.where((self.dsHist.WARM!=-1)&(self.dsHist.WIND > min_wind2))
        self.dsAll= xr.merge([self.dsAll2, self.dsHist])
        self.dsAllexcl33= xr.merge([self.dsAll2excl33, self.dsHistexcl33])
        self.dsAllExcl= xr.merge([self.dsAll2excl, self.dsHistexcl])
#         self.dsAll=self.dsAll2
#         self.dsAllexcl33=self.dsAll2excl33
#         self.dsAllExcl=self.dsAll2excl
#         self.number = number
         

   # method to obtain a dataframe from the Xarray datasets     
    def returnDF(self):
        if self.allEns: ensRange = self.dsAll.dims['YENS']
        else: ensRange = 1
        #iterate through the n ensemble members 
        for n in range(11):

            print(n)
            theList = self.restructure(number=n)

            theArray = np.asarray(theList)

            newArray = []

            for i in theArray:
               
                newArray.append(np.hstack(i))
        
            #incorporate annual mean temperature data
            tempdata=self.tempdata
            tempdata = tempdata[tempdata.iloc[:, 0]>=1970]
            tempdata['year']=tempdata.iloc[:,0]
            tempdata['meanTemp']=tempdata.iloc[:,1]
            tempdata=tempdata.iloc[:,3:5]
            
            #construct new dataframe 
            newDF = pd.DataFrame(np.asarray(newArray).T)
            newDF.columns = ['lon', 'lat', 'year', 'month','day', 'zSTORM1', 'hour','XCOUNT','wind','SLP']
            #add annual mean temperature data to the dataframe 
            merged = newDF
            merged = pd.merge(newDF, tempdata, how='left', on='year')
            merged['meanTemp']=np.ones(merged.shape[0])
            #add a column indicating the ensemble member number
            merged['ensemble']= n*np.ones(newDF.shape[0])
#             print(merged['year'].values)
            if self.slp4x: 
                merged=addSLP4x(merged,n)
            else: 
                merged=addSLP(merged,n,self.ensemblesSLP)


            if n ==0: 
                theDF = merged
            else: theDF = theDF.append(merged)

        return theDF

    def front_back_indices4(self,winds,windsAllSpeed):
 
        
        a = winds
#         a[np.isnan(windsAllSpeedGapFill)]=np.nan
        addFront = False
        addEnd = False 

        if (not np.isnan(a[0]))& (not np.isnan(a[-1])):
            addFront = True
            a2=np.insert(a,0,np.nan)
            addEnd =True 
            a2=np.append(a2,np.nan)

        elif (not np.isnan(a[-1])):
            addEnd =True 
            a2=np.append(a,np.nan)

        elif (not np.isnan(a[0])):
            addFront = True
            a2=np.insert(a,0,np.nan) 

        else: a2 = a.copy()
        b=np.argwhere(np.isnan(a2))
        b=b.T[0]
        c=(np.diff(b)>3)
        d = np.argwhere(c).T[0]
        if (len(d)==0):
            return -100,-100
        else:
            first = d[0]
            last = d[-1]
            firstindex = b[first]+int(not addFront)
            lastindex = b[last+1]-1 - int(addFront)
            return firstindex, lastindex
                
    # helper method to restructure Xarray dataset into lists  
    def restructure(self, number):
            ds = self.dsAll.sel(YENS=number+1).drop('YENS')
            ds2 = self.dsAllExcl.sel(YENS=number+1).drop('YENS')
            dsExcl33= self.dsAllexcl33.sel(YENS=number+1).drop('YENS')
#             ds = self.dsAll
            lonsList,  latsList, yearsList,monthsList,daysList,zSTORM1List,XCOUNTList,hoursList, windsList,SLPList = ([] for i in range(10)) 

            #each year
            for i in ds2['YRT'].values.astype(int):

                #each storm
                for j in ds2['ZSTORM1'].values.astype(int):
                    windsAllSpeeds = ds['WIND'].sel(YRT=i).sel(ZSTORM1=j).values
                    winds = ds2['WIND'].sel(YRT=i).sel(ZSTORM1=j).values
                    winds2 = dsExcl33['WIND'].sel(YRT=i).sel(ZSTORM1=j).values 
#                     lats= ds['LAT'].sel(YRT=i).sel(ZSTORM1=j).values
                    #if winds is not all nan
                    if np.sum(np.isnan(winds2))!=len(winds2):
                        firstindex,lastindex= self.front_back_indices4(winds,windsAllSpeeds)
                        if firstindex!=-100:
                            thelength = len(ds['LAT'].sel(YRT=i).sel(ZSTORM1=j).values[firstindex:lastindex+1])
                            if thelength>total_storm_length_thresh:
                                latsList.append(ds['LAT'].sel(YRT=i).sel(ZSTORM1=j).values[firstindex:lastindex+1])
                                lonsList.append(ds['LON'].sel(YRT=i).sel(ZSTORM1=j).values[firstindex:lastindex+1])
                                XCOUNTList.append(np.asarray(range(1,thelength+1,1)))

                                yearsList.append(ds['YEAR'].sel(YRT=i).sel(ZSTORM1=j).values[firstindex:lastindex+1])
                                hoursList.append(ds['HOUR'].sel(YRT=i).sel(ZSTORM1=j).values[firstindex:lastindex+1])
                                daysList.append(ds['DAY'].sel(YRT=i).sel(ZSTORM1=j).values[firstindex:lastindex+1])
                                monthsList.append(ds['MONTH'].sel(YRT=i).sel(ZSTORM1=j).values[firstindex:lastindex+1])
                                if j==21: print(thelength)
                                zSTORM1List.append(j*np.ones(thelength))
                                windsList.append(ds['WIND'].sel(YRT=i).sel(ZSTORM1=j).values[firstindex:lastindex+1])
                                SLPList.append(ds['SLP'].sel(YRT=i).sel(ZSTORM1=j).values[firstindex:lastindex+1])
        
            return lonsList,latsList, yearsList, monthsList,daysList, zSTORM1List, hoursList, XCOUNTList, windsList,SLPList
    
    #return dataframe with day of year included 
    def AddDay(self):
        merged = self.returnDF()[['lon', 'lat', 'year', 'month','day', 'zSTORM1', 'hour','XCOUNT','wind','SLP','meanTemp','ensemble','SLPgradx','SLPgrady']]
        dfDate2 = pd.DataFrame({'year': merged['year'].values, 'month': merged['month'].values,'day': merged['day']})
#         merged['dayofyr']=self.date_to_nth_day(dfDate2)
   
        return merged
    
    
    #helper method to get the day of year from a date
    @staticmethod
    def date_to_nth_day(dfDate):
            date = pd.to_datetime(dfDate, format=format)
            daysyr=[]
            for i in range(date.shape[0]):
                try: 
                    theDate=date.iloc[i]
                    new_year_day = pd.Timestamp(year=theDate.year, month=1, day=1)
                    daysyr.append((theDate - new_year_day).days + 1)
                except TypeError: 
                    daysyr.append(np.nan)
            return daysyr
        
        
        
        
    #restructure the data for the autoregressive model   
    def for_regression(self):
        merged= self.AddDay()
        rows = []
        lats = []
        lons=[]
        SLPx = []
        SLPy = []

        # build a dataframe where each row contains the variables 
        #from one timestep before and two timesteps before
        
        for i in range(merged.shape[0]):
            #account for first two points in any storm
            if merged['XCOUNT'].values[i]==1:
                twobefore = merged.iloc[i,:].values

            elif merged['XCOUNT'].values[i]==2:
                onebefore = merged.iloc[i,:].values
            
            else: 
                if twobefore.shape[0] !=1:
                    #build feature space
                    rows.append(np.concatenate((twobefore,onebefore),axis = None))
                    
                    #build y variables
                    lats.append(merged['lat'].values[i])
                    lons.append(merged['lon'].values[i])


                    SLPx.append(merged['SLPgradx'].values[i])
                    SLPy.append(merged['SLPgrady'].values[i])
      
                # case of a nan row
                if np.isnan(merged.iloc[i,:].values[0]):
                    twobefore = np.asarray([0])
                    onebefore = np.asarray([0])

                    
                # to be added in next iteration
                else: 
                    twobefore = onebefore
                    onebefore = merged.iloc[i,:].values
        #convert to dataframe 
        lats = np.asarray(lats)       
        lons = np.asarray(lons)   
        SLPx=np.asarray(SLPx)
        SLPy=np.asarray(SLPy)
  
        df5=pd.DataFrame(np.asarray(rows))
        df5.columns = ['lon2', 'lat2', 'year2', 'month2','day2', 'zSTORM12', 'hour2','XCOUNT2','wind2','SLP2','meanTemp2','ensemble2','SLPgradx2','SLPgrady2','lon', 'lat', 'year', 'month','day', 'zSTORM1', 'hour','XCOUNT','wind','SLP','meanTemp','ensemble','SLPgradx','SLPgrady']
        #add in y variables 
        df5['ylat']=lats
        df5['ylon']=lons

        df5=df5.dropna()
        return df5[df5['year']!=2022]
        
        



def get_models(df5): 
    #variables to be kept for the regression
    X1= df5[['lat2','lat', 'lon2','lon','SLPgradx','SLPgrady','XCOUNT', 'wind','ylat','ylon']]


    X1['latdiffolder'] = X1['lat']-X1['lat2']
    X1['latdiffnewer'] = (X1['ylat']-X1['lat'])
    X1['coslat']= np.cos(np.radians(X1['lat']))
    X1['londiffolder'] = (X1['lon']-X1['lon2'])*np.cos(np.radians(X1['lat2']))
    X1['londiffnewer'] = ((X1['ylon']-X1['lon'])*np.cos(np.radians(X1['lat'])))
    X1['windbeta']= X1['wind']* X1['coslat']
    X1['wind2beta']= X1['wind']*X1['wind']*X1['coslat']
    X1['wind3beta']= X1['wind']*X1['wind']*X1['wind']*X1['coslat']
    X1['SLPgradxdivide']= X1['SLPgradx']/np.sin(np.radians(X1['lat']))
    X1['SLPgradydivide']=X1['SLPgrady']/np.sin(np.radians(X1['lat']))

    X3 = X1[['windbeta','wind2beta','SLPgradxdivide','SLPgradydivide']]
    scaler = preprocessing.StandardScaler().fit(X3)
    X = scaler.transform(X3)
    X = pd.DataFrame(X, columns= X3.columns)
    X2 = X1[['latdiffolder']]
    X2['coslat']= X1['coslat']
    X2['londiffolder']= X1['londiffolder']
    X2['windbeta']= X['windbeta']
    X2['wind2beta']= X['wind2beta']
    # X2['wind3beta']= X1['wind3beta']
    X2['SLPgradx']= X['SLPgradxdivide']
    X2['SLPgrady']= X['SLPgradydivide']
    # X2['SLPgradx']= X['SLPgradx']/np.sin(np.radians(X1['lat']))
    # X2['SLPgrady']=X['SLPgrady']/np.sin(np.radians(X1['lat']))

    # X2= X1[['lattdiffolder','coslat','londiffolder','windbeta','SLPgradx','SLPgrady']]
    # X2 = X[['latdiffolder','coslat','londiffolder','windbeta','SLPgradx','SLPgrady']]
    X2 = sm.add_constant(X2)
    # # #run 4 regressions for each of lat, lon, SLP, and wind 
    # # #include the annual mean temp variable 

    modelLat = sm.OLS(X1['latdiffnewer'].values,X2[['coslat','windbeta','wind2beta','SLPgradx']]).fit(cov_type='HC0')
    modelLon = sm.OLS(X1['londiffnewer'].values,X2[['coslat','windbeta','wind2beta','SLPgrady']]).fit(cov_type='HC0')
    #subtract mean and divide by standard deviation
    return modelLat,modelLon, X2, X1


def transform_coefs(modelLat_df, windbeta_mean, windbeta_std, wind2beta_mean, wind2beta_std, SLPgradx_mean, SLPgradx_std, y = True): 
    meters_per_degree = 111000
    seconds_per_6hours = 6*60*60
    radius_earth= 6371000
    omega = 2*np.pi/86400
    twoomega_overa= omega*2/radius_earth
    twoomega = 2*omega
    
    if y:
        latdiffolder = 'latdiffolder'
        SLPgradx = 'SLPgradx'
    else: 
        latdiffolder = 'londiffolder'
        SLPgradx = 'SLPgrady'
    modelLat_df2= modelLat_df.copy()
    modelLat_df2['coef']= modelLat_df['coef']*(meters_per_degree/seconds_per_6hours)
    modelLat_df2['std err']= modelLat_df['std err']*(meters_per_degree/seconds_per_6hours)
#     print(modelLat_df)
#     print(modelLat_df2)
    modelLat_df3= modelLat_df2.copy()
    modelLat_df3['coef']['coslat']= modelLat_df2['coef']['coslat']/twoomega_overa
    modelLat_df3['std err']['coslat']= modelLat_df2['std err']['coslat']/twoomega_overa





    modelLat_df3['coef']['windbeta']= modelLat_df2['coef']['windbeta']/(windbeta_std*twoomega_overa)
    modelLat_df3['std err']['windbeta']= modelLat_df2['std err']['windbeta']/(windbeta_std*twoomega_overa)
#     print(modelLat_df3)
    modelLat_df3['coef']['wind2beta']= modelLat_df2['coef']['wind2beta']/(wind2beta_std*twoomega_overa)
    modelLat_df3['std err']['wind2beta']= modelLat_df2['std err']['wind2beta']/(wind2beta_std*twoomega_overa)


    modelLat_df3['coef'][SLPgradx]= (modelLat_df2['coef'][SLPgradx]/(SLPgradx_std))*(twoomega/100)
    modelLat_df3['std err'][SLPgradx]= (modelLat_df2['std err'][SLPgradx]/(SLPgradx_std))*(twoomega/100)



    
    
    return modelLat_df3

