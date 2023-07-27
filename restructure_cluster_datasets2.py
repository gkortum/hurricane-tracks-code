
#import libraries
import numpy as np
import xarray as xr
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import scipy.interpolate as interp

from format import *

#resample historical data to 6-hourly frequency and structure in Xarray dataset



wind_conversion = 0.514444
long_length = 140
longitude_number = 360
time_increment = 6
storm_max_number = 35


# make points per storm variable 140 long
def sixhourly(thevar,c):
    d = np.where(~np.isnan(c),thevar,c)
    e = d[~np.isnan(d)]
    thelength = len(e)
 
    Nanfillers = np.nan*np.ones(long_length -thelength)
  
    f = np.append(e,Nanfillers)
    return f

# make list of storms 35 long
def reshape35(stormList):
    thearray=np.array(stormList)
    rows = thearray.shape[0]
    toappend= np.nan*np.ones([storm_max_number-rows,long_length ])
    return np.concatenate([thearray,toappend], axis = 0)


# restructure into a lot of lists
def dsReal_to_df(dsHist):
    prevyear = 1850
    yearArraysLat = []
    yearArraysLon = []
    yearArraysWind = []
    yearArraysSLP = []
    yearArraysYear = []
    yearArraysMonth = []
    yearArraysDay = []
    yearArraysHour = []
    for i in dsHist['storm'].values:
#         print(i)
        times = dsHist['time'].isel(storm=i).values
        theYear = pd.DatetimeIndex(times).year.values
        theMonths = pd.DatetimeIndex(times).month.values
        theDays = pd.DatetimeIndex(times).day.values
        theHours = pd.DatetimeIndex(times).hour.values
        theWinds = dsHist['usa_wind'].isel(storm=i).values*wind_conversion
        theSLP = dsHist['usa_pres'].isel(storm=i).values
        theLats = dsHist['usa_lat'].isel(storm=i).values
        lons = dsHist['usa_lon'].isel(storm=i).values
        theLons = lons +longitude_number
#         theLons = np.where(lons<0, lons2, lons)

        b= np.nan*np.ones(len(theHours))
        c=np.where(np.isin(theHours,[0,time_increment,time_increment*2,time_increment*3]),theHours,b)

        #resample to 6-hourly 


        if theYear[0] !=prevyear: 
            prevyear = theYear[0]
            if prevyear !=1851: 
                yearArraysLat.append(reshape35(stormsLat))
                yearArraysLon.append(reshape35(stormsLon))
                yearArraysWind.append(reshape35(stormsWind))
                yearArraysSLP.append(reshape35(stormsSLP))
                yearArraysYear.append(reshape35(stormsYear))
                yearArraysMonth.append(reshape35(stormsMonth))
                yearArraysDay.append(reshape35(stormsDay))
                yearArraysHour.append(reshape35(stormsHour))
            stormsLat = []
            stormsLon = []
            stormsWind = []
            stormsSLP = []
            stormsYear = []
            stormsMonth = []
            stormsDay = []
            stormsHour = []
                
        stormsLat.append(sixhourly(theLats,c))
        stormsLon.append(sixhourly(theLons,c))
        stormsWind.append(sixhourly(theWinds,c))
        stormsSLP.append(sixhourly(theSLP,c))
        stormsYear.append(sixhourly(theYear,c))
        stormsMonth.append(sixhourly(theMonths,c))
        stormsDay.append(sixhourly(theDays,c))
        stormsHour.append(sixhourly(theHours,c))

    yearArraysLat.append(reshape35(stormsLat))
    yearArraysLon.append(reshape35(stormsLon))
    yearArraysWind.append(reshape35(stormsWind))
    yearArraysSLP.append(reshape35(stormsSLP))
    yearArraysYear.append(reshape35(stormsYear))
    yearArraysMonth.append(reshape35(stormsMonth))
    yearArraysDay.append(reshape35(stormsDay))
    yearArraysHour.append(reshape35(stormsHour))
    return np.array(yearArraysLat),np.array(yearArraysLon),np.array(yearArraysWind),np.array(yearArraysSLP),np.array(yearArraysYear),np.array(yearArraysMonth),np.array(yearArraysDay),np.array(yearArraysHour)


# convert to xarray
def constructXR(dsHist, num_ensembles,year_start,year_end):
    dsHist=dsHist.where(((dsHist.lon<60)&(dsHist.lon >-120)),drop = True)

#     dsHist = dsHist.where(dsHist.lat<40,drop = True)
    yearArraysLat,yearArraysLon,yearArraysWind,yearArraysSLP,yearArraysYear,yearArraysMonth,yearArraysDay,yearArraysHour =dsReal_to_df(dsHist)
    
    ds = xr.Dataset(
        {
          "LON": (['YRT','ZSTORM1','YENS','XCOUNT'],np.expand_dims(yearArraysLon,axis =2)),
            "LAT": (['YRT','ZSTORM1','YENS','XCOUNT'], np.expand_dims(yearArraysLat,axis =2)),
            "WIND": (['YRT','ZSTORM1','YENS','XCOUNT'], np.expand_dims(yearArraysWind,axis =2)),
            "SLP": (['YRT','ZSTORM1','YENS','XCOUNT'], np.expand_dims(yearArraysSLP,axis =2)),
            "YEAR": (['YRT','ZSTORM1','YENS','XCOUNT'], np.expand_dims(yearArraysYear,axis =2)),
            "MONTH": (['YRT','ZSTORM1','YENS','XCOUNT'], np.expand_dims(yearArraysMonth,axis =2)),
            "DAY": (['YRT','ZSTORM1','YENS','XCOUNT'], np.expand_dims(yearArraysDay,axis =2)),
            "HOUR": (['YRT','ZSTORM1','YENS','XCOUNT'], np.expand_dims(yearArraysHour,axis =2)),
            "WARM": (['YRT','ZSTORM1','YENS','XCOUNT'], np.expand_dims(np.nan*np.ones(yearArraysWind.shape),axis =2)),
        },
        coords={
            'YRT': (range(1851,2022,1)),
            'XCOUNT':(range(1,long_length+1,1)),
            'ZSTORM1':(range(1,storm_max_number+1,1)),
            'YENS': ([num_ensembles+1]),
        }
    )
    return ds.sel(YRT=slice(year_start,year_end,1)).sel(XCOUNT =slice(1,120,1))



#merge the model and historical data and restructure it into lists for it to be clustered
time_after_gen_thresh = 3
total_storm_length_thresh = 7

class Restructure_Data: 
    def __init__(self, dsAll, dsHist, number, num_clusters, min_wind, min_wind_hist, min_wind2):
  #full dataset 
        
        #restrict longitude
        self.dsAll2 = dsAll.where(((dsAll.LON<420)&(dsAll.LON>240)),drop = True)

#         self.dsAll2 = self.dsAll1.where(self.dsAll1.LAT<40)
        #restrict wind speed and warm core
        self.dsAll2excl= self.dsAll2.where((self.dsAll2.WARM!=-1)&(self.dsAll2.WIND > min_wind))
        self.dsHist= dsHist
        self.dsHistexcl= dsHist.where((self.dsHist.WARM!=-1)&(self.dsHist.WIND > min_wind_hist))
        self.dsAll2excl33 = self.dsAll2.where((self.dsAll2.WARM!=-1)&(self.dsAll2.WIND>min_wind2))
        self.dsHistexcl33 = dsHist.where((self.dsHist.WARM!=-1)&(self.dsHist.WIND > min_wind2))
        #merge datasets
        self.dsAll= xr.merge([self.dsAll2, self.dsHist])
        self.dsAllexcl= xr.merge([self.dsAll2excl, self.dsHistexcl])
        self.dsAllexcl33= xr.merge([self.dsAll2excl33, self.dsHistexcl33])
        #number of ensembles plus historical dataset
        self.number = number
        

        #cluster numbers 
        self.num_clusters= num_clusters



    
    #find index of first and last points in a storm
    def front_back_indices4(self, winds, windsAllSpeed):
#         windsAllSpeedGapFill = []
#         startWithNan = False
#         startWithNotNan = False
        
#         firstNan = False
#         firstNotNan = False
#         secondNan = False
#         count = 0
#         for i in windsAllSpeed:
#             if count ==0: 
#                 if np.isnan(i): 
#                     startWithNan = True
#                     firstNan = True
#                 else: 
#                     startWithNotNan = True
#                     firstNotNan=True
                    
#                 windsAllSpeedGapFill.append(i)
#             else: 
#                 if np.isnan(i): 
#                     firstNan = True
#                     if firstNotNan ==True: 
#                         secondNan = True
#                     windsAllSpeedGapFill.append(i)
#                 else: 
#                     firstNotNan =True
#                     if startWithNan:
#                         if secondNan:
#                             windsAllSpeedGapFill.append(np.nan)
                            
#                         else: windsAllSpeedGapFill.append(i)
#                     else:
#                         if firstNan:
#                             windsAllSpeedGapFill.append(np.nan)
#                         else: 
#                             windsAllSpeedGapFill.append(i)
                    

#             count +=1   
        
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
        c=(np.diff(b)>time_after_gen_thresh)
        d = np.argwhere(c).T[0]
        if (len(d)==0):
            return -100,-100
        else:
            first = d[0]
            last = d[-1]
            firstindex = b[first]+int(not addFront)
            lastindex = b[last+1]-1 - int(addFront)
            return firstindex, lastindex
                
    #method to restructure dataset into lists  
    def restructure(self, number2):

            ds = self.dsAll.isel(YENS=int(number2))
            dsExcl = self.dsAllexcl.isel(YENS=int(number2))
            dsExcl33 = self.dsAllexcl33.isel(YENS=int(number2))
            lonsList= []
            windsList=[]
            latsList=[]
            yearsList=[]
            monthsList=[]
            daysList=[]
            zSTORM1List=[]
                #each year
            for i in ds['YRT'].values.astype(int):
                    #each storm
                 
         
                    for j in ds['ZSTORM1'].values.astype(int):
                        windsAllSpeeds = ds['WIND'].sel(YRT=i).sel(ZSTORM1=j).values
                        winds = dsExcl['WIND'].sel(YRT=i).sel(ZSTORM1=j).values
                        #if winds is not all nan
                        winds2 = dsExcl33['WIND'].sel(YRT=i).sel(ZSTORM1=j).values   
                        if np.sum(np.isnan(winds))!=len(winds):
#                             if number2 !=self.number -1:
                            firstindex,lastindex= self.front_back_indices4(winds, windsAllSpeeds)
#                             elif number2 ==self.number -1: 
#                                 firstindex = 0
#                                 lastindex = len(ds['LAT'].sel(YRT=i).sel(ZSTORM1=j).values)
                            if firstindex!=-100:
                                 
                                thelength = len(ds['LAT'].sel(YRT=i).sel(ZSTORM1=j).values[firstindex:lastindex+1])

                                if thelength >total_storm_length_thresh:

                                    toadd = np.nan*np.ones(120-thelength)
                                    latsList.append(np.append(ds['LAT'].sel(YRT=i).sel(ZSTORM1=j).values[firstindex:lastindex+1],toadd))
                                    lonsList.append(np.append(ds['LON'].sel(YRT=i).sel(ZSTORM1=j).values[firstindex:lastindex+1],toadd))

                                    windsList.append(np.append(ds['WIND'].sel(YRT=i).sel(ZSTORM1=j).values[firstindex:lastindex+1],toadd))
                                    yearsList.append(np.append(ds['YEAR'].sel(YRT=i).sel(ZSTORM1=j).values[firstindex:lastindex+1],toadd))

                                    daysList.append(np.append(ds['DAY'].sel(YRT=i).sel(ZSTORM1=j).values[firstindex:lastindex+1],toadd))
                                    monthsList.append(np.append(ds['MONTH'].sel(YRT=i).sel(ZSTORM1=j).values[firstindex:lastindex+1],toadd))

                                    zSTORM1List.append(j)
      
            return lonsList, windsList, latsList, yearsList, monthsList,daysList, zSTORM1List  
    
    
    
    
    
    
# define clusters and compute the silhouette scores from the historical and model data merged     
class defCluster(Restructure_Data): 
    def __init__(self, Restructure_Data):
        #full dataset
        self.Restructure_Data = Restructure_Data
        
        
        self.dsAll = Restructure_Data.dsAll
        self.number = Restructure_Data.number
        self.dsHist= Restructure_Data.dsHist
        self.dsAllexcl = Restructure_Data.dsAllexcl


        #cluster numbers 
        self.num_clusters= Restructure_Data.num_clusters
        


        
    def restructure2(self,number):
  
        return self.Restructure_Data.restructure(number)
    
    #clustering method returns dataframe with clusters
    def cluster(self):
        dfs = []
        dfsplusLabel = []
        for i in range(self.number): 
#         
            lonsList, windsList, latsList,yearsList, monthsList,daysList, zSTORM1List = self.restructure2(number = i)
    
            # construct dataframe with relevant information for clustering
            df1 = self.dfIndicesFromLists(windsList,latsList,lonsList)
            df2 = df1.copy()
            dfs.append(df1)
            df2['dataset'] = np.ones(df2.shape[0])*i
            df2['Zstorm']= np.asarray(zSTORM1List)

            windsList2=[]
            for k in windsList: 
                thewind = np.nanmax(np.array(k))
                windsList2.append(thewind)
            df2['maxwind']=np.asarray(windsList2)
            yearsList2=[]
            for k in yearsList:
                theyear = int(np.nanmean(np.asarray(k)))
                yearsList2.append(theyear)
            df2['Year']= np.asarray(yearsList2)
            dfsplusLabel.append(df2)
        df = pd.concat(dfs)
        dfsplusLabel2 = pd.concat(dfsplusLabel)

        
        num_clusters=self.num_clusters
        clusterer = KMeans(n_clusters=num_clusters, random_state = 10)
        cluster_labels = clusterer.fit_predict(df)

        df['cluster']=cluster_labels
        df['cluster']=self.changeClusterNames(df)
        df['Zstorm']= dfsplusLabel2['Zstorm']
        df['dataset']=dfsplusLabel2['dataset']
        df['Year']=dfsplusLabel2['Year']
        df['maxwind']=dfsplusLabel2['maxwind']


        return df
    
    
    def silhouette_score(self):
        #getting silhouette scores for different cases
        dfs = []
        dfsplusLabel = []
        for i in range(self.number): 
#             ds = self.dsWarms[i]
            lonsList, windsList, latsList,yearsList, monthsList,daysList, zSTORM1List = self.restructure2(number = i)
            df1 = self.dfIndicesFromLists(windsList,latsList,lonsList)
            df2 = df1.copy()
            dfs.append(df1)
            df2['dataset'] = np.ones(df2.shape[0])*i
            df2['Zstorm']= np.asarray(zSTORM1List)
            yearsList2=[]
            for k in yearsList:
                theyear = np.nanmean(np.asarray(k))
                yearsList2.append(theyear)
            df2['Year']= np.asarray(yearsList2)
            dfsplusLabel.append(df2)
        df = pd.concat(dfs)
        dfsplusLabel2 = pd.concat(dfsplusLabel)
        

        scores = []
        for i in range(2,11):


            clusterer = KMeans(n_clusters=i, random_state = 10)

            cluster_labels = clusterer.fit_predict(df)
            silhouette_avg = silhouette_score(df, cluster_labels)
            scores.append(silhouette_avg)
   
        return scores
    


    
    #helper method to relabel so labels represent last longitude ranking
 
    def changeClusterNames(self,df):
        dfmean=df.groupby('cluster').mean()
#         dfmean['meanlon']= np.mean(np.asarray([dfmean['firstlon'],dfmean['maxlon'],dfmean['lastlon']]).T, axis = 1)
#         dfmean = dfmean.sort_values(by='meanlon', axis=0)
        dfmean = dfmean.sort_values(by='lon10', axis=0)
        dfmean['cluster2']=np.asarray(range(self.num_clusters))
        dfmean= dfmean.sort_index()
        cluster2=[]
        for i in df['cluster'].values:
            cluster2.append(dfmean['cluster2'].values[i])
        return np.asarray(cluster2)      
 
    
    #helper method to get the indices for clustering into dataframe
    def dfIndicesFromLists(self,windsList,latsList,lonsList):
        maxindices, lastindices, firstindices= self.getIndices(windsList)
        count = 1
        columnNames = []
        for i in range(20):
            if i >9:
                pre = "lon"
                if i ==10: 
                    count = 1
            else: 
                pre ="lat"
            columnNames.append(pre+str(count))
            count +=1

#         columnNames.append('maxlat')
#         columnNames.append('maxlon')
#         firstlat=self.getvals(firstindices,latsList)
#         firstlon=self.getvals(firstindices,lonsList)

#         lastlat=self.getvals(lastindices,latsList)
#         lastlon=self.getvals(lastindices,lonsList)

        maxlat=self.getvals(maxindices,latsList)
        maxlon=self.getvals(maxindices,lonsList)
        tenLats = self.get10vals(firstindices,lastindices,latsList)
        tenLons = self.get10vals(firstindices,lastindices,lonsList)
 
        concatenated = np.concatenate((tenLats,tenLons), axis =1)
        df = pd.DataFrame(concatenated, 
                   columns =columnNames)
        return df
    
    #helper method to get the values in the indices
    @staticmethod
    def get10vals(indicesStart,indicesEnd,thelist):
        thearray= np.asarray(thelist)
        count = 0
        values=[]

        for i in thearray: 
#             if count ==336:
#                 print(fullstorm)
#                 print(fullstorm_interp)
#                 print(fullstorm_compress)
            fullstorm = i[indicesStart[count]:indicesEnd[count]+1]
            fullstorm = np.array(fullstorm)
            fullstorm_interp = interp.interp1d(np.arange(fullstorm.size),fullstorm)
            fullstorm_compress = fullstorm_interp(np.linspace(0,fullstorm.size-1,10))
            
            values.append(fullstorm_compress)
            count +=1
        return np.asarray(values)

               
    # helper method to get the indices used from lists 
    @staticmethod
    def getIndices(windsList):
        try: 
            maxindices = np.argmax(np.nan_to_num(np.asarray(windsList)),axis = 1)
            ncols= np.shape(np.asarray(windsList))[1]
            #find last indices
            #lastindices = ncols-1-np.sum(np.isnan(np.asarray(windsList)), axis =1)
            lastindices = []
            for i in np.asarray(windsList):
                index=ncols-1
                for k in np.flip(i):
                    if not np.isnan(k):
                        break
                    index-=1
                lastindices.append(index)
            #find first indices
            firstindices = []
            for i in np.asarray(windsList):
                index=0
                for k in i:
                    if not np.isnan(k):
                        break
                    index+=1
                firstindices.append(index)
            #firstindices=np.ones(np.shape(np.asarray(windsList))[0]).astype(int)
            return maxindices, lastindices, firstindices
        except: print(windsList)

    
    #helper method to get the values in the indices
    @staticmethod
    def getvals(indices,thelist):
        thearray= np.asarray(thelist)
        count = 0
        values=[]
        for i in thearray: 
            values.append(i[indices[count]])
            count +=1
        return np.asarray(values) 
    
    # method to visualize the clusters
    def viewClusters(self, number, allens =False):
        df=self.cluster()
        if not allens: 
            df=df[df['dataset']==number]
        
        #threes
        ax=mapsetup(minlat=-10,maxlat=90,minlon=-120,maxlon=30, big =True)
        asarray=np.asarray(df)
        for i in range(df.shape[0]):
            row = asarray[i]
            lats = [row[0],row[2], row[4]]
            lons = [row[1],row[3], row[5]]
            ax.scatter(lons,lats,c=colorList3[row[6].astype(int)], s=20)

        #means    
        dfmean=df.groupby(['cluster']).mean()
        dfstd=df.groupby(['cluster']).std()
        ax=mapsetup(minlat=-5,maxlat=75,minlon=-100,maxlon=45, big = True)
        for i in range(self.num_clusters):
            lats = [np.asarray(dfmean.iloc[i,:])[0], np.asarray(dfmean.iloc[i,:])[2],np.asarray(dfmean.iloc[i,:])[4]]
            lons = [np.asarray(dfmean.iloc[i,:])[1], np.asarray(dfmean.iloc[i,:])[3],np.asarray(dfmean.iloc[i,:])[5]]
            laterr= [np.asarray(dfstd.iloc[i,:])[0], np.asarray(dfstd.iloc[i,:])[2],np.asarray(dfstd.iloc[i,:])[4]]
            lonerr = [np.asarray(dfstd.iloc[i,:])[1], np.asarray(dfstd.iloc[i,:])[3],np.asarray(dfstd.iloc[i,:])[5]]
            cluster = i
            ax.plot(lons, lats,'o-',c=colorList3[cluster],transform=ccrs.PlateCarree(), lw=5,ms=17, label=str(cluster))
            ax.errorbar(lons, lats, xerr=lonerr, c=colorList3[cluster], yerr=laterr,transform=ccrs.PlateCarree(),lw=2)
            ax.legend()
            
    
    
    
    




def plot_silhouette_scores(scores): 
    # plot silhouette scores
    plt.figure(dpi = 300, figsize=(8,6))
    
    plt.plot(range(2,11),scores,'-H', c = 'seagreen', linewidth = 3)
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette score')
    plt.savefig('/tigress/gkortum/thesisSummer/figures/silhouettescore.jpg')

    
 








