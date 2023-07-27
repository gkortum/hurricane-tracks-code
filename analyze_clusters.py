
#import libraries
import numpy as np
import xarray as xr
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


from format import *
from restructure_cluster_datasets import *
#resample historical data to 6-hourly frequency and structure in Xarray dataset






class Analyze_Clusters(defCluster):
    def __init__(self, defCluster, number, num_clusters, hurricanes_only):

        #full dataset with clusters defined
        self.defCluster=defCluster
        self.number = number
 
        #cluster numbers 
        self.num_clusters=num_clusters
   
        self.hurricanes_only = hurricanes_only
        df =self.defCluster.cluster()
        self.df = df[df['dataset']==number]
    
    def list2(self):
        
        number = self.number

        return self.defCluster.restructure2(number)
    # dataframe of all latitudes of tracks
    def dfLAT(self): 
     
        dfLAT = pd.DataFrame(np.asarray(self.list2()[2]))
        dfLAT['Year']= self.df['Year'].values
        dfLAT['cluster']= self.df['cluster'].values
        if self.hurricanes_only:
            dfLAT=dfLAT[self.df['maxwind']>=33].reset_index().drop('index',axis = 1)
        return dfLAT
    # dataframe of all longitudes of tracks
    def dfLON(self):
        dfLON = pd.DataFrame(np.asarray(self.list2()[0]))
        dfLON['Year']= self.df['Year'].values
        dfLON['cluster']= self.df['cluster'].values 
        if self.hurricanes_only:
            dfLON=dfLON[self.df['maxwind']>=33].reset_index().drop('index',axis = 1)
        return dfLON
        
   
    # plot the clusters over the years
    def clusterTimePlot(self):
        dfLAT = self.dfLAT()
        dfLON = self.dfLON()
        for year in [1970,2021]:
            
            ax,fig=mapsetup2(minlat=-10,maxlat=90,minlon=-120,maxlon=15)
            dfLAT2 = dfLAT.drop(dfLAT[dfLAT.Year != year].index)
            dfLON2 = dfLON.drop(dfLON[dfLON.Year != year].index)
            for i in range(dfLAT2.shape[0]):
                lats = np.asarray(dfLAT2.iloc[i,:])[:-2]
                lons = np.asarray(dfLON2.iloc[i,:])[:-2]
                cluster = np.asarray(dfLAT2.iloc[i,:])[-1].astype(int)
                a = ax.scatter(lons, lats, c=colorList3[cluster], s= 6)

            plt.title(str(year))
            if year ==1971: fig.savefig('plot1971.jpg')
            else: fig.savefig('plot2019.jpg')
            
    #add in cluster dummy variables
    def yearCluster01(self):
        yearCluster = self.yearCluster()
        
        list1=[]
        list2=[]
        list3=[]
        list4=[]
        lists = [list1,list2,list3,list4]
        for i in range(len(yearCluster['cluster'])):
            for j in range(self.num_clusters):

                if yearCluster['cluster'].values[i]==j: 
                    lists[j].append(1)
                else: lists[j].append(0)
        yearCluster['cluster0']=list1
        yearCluster['cluster1']=list2
        yearCluster['cluster2']=list3
        if self.num_clusters == 4: 
            yearCluster['cluster3']=list4
        
        #add for above and below 1995
        ones=np.ones(yearCluster['Year'].values.shape[0])
        zeros=np.zeros(yearCluster['Year'].values.shape[0])
        above1995=np.where(yearCluster['Year'].values>1995, ones,zeros)
        yearCluster['above1995']=above1995
        return yearCluster 
    
   #dataframe with dates 
    def dfDate(self):
        monthsList =self.list2()[4]
        daysList = self.list2()[5]
 
        #build in the day of year metrics
        monthsList2=[]
        for i in monthsList:
            for k in i: 
                if not np.isnan(k):
                    monthsList2.append(int(k))
                    break
        daysList2=[]
        for i in daysList:
            for k in i: 
                if not np.isnan(k):
                    daysList2.append(int(k))
                    break
 
        #dataframe of the dates
        count = 0
        for i in self.dfLAT()['Year'].values:
            if not np.isnan(i): count +=1

        dfDate = pd.DataFrame({'year': self.dfLAT()['Year'].values.astype(int), 'month': monthsList2,'day': daysList2})
        dfDate = dfDate.iloc[0:count,:]
        #obtain day of year metrics
        dfDate['dayofyr']=self.date_to_nth_day(dfDate)
        return dfDate
        
        
    #organize the year cluster dataframe
    def yearCluster(self):
        
        #build up yearCluster dataframe
        
        yearCluster= self.dfLAT()[['Year','cluster']]
        yearCluster['ones']= np.ones(yearCluster.shape[0])
#         dfDate = self.dfDate()
#         yearCluster['dayofYear']= dfDate['dayofyr']
        return yearCluster
    
    
    @staticmethod
    def date_to_nth_day(dfDate):
            try:
                date = pd.to_datetime(dfDate, format=format)
            except ValueError: print(dfDate)
            daysyr=[]
            for i in range(date.shape[0]):
                theDate=date.iloc[i]
                new_year_day = pd.Timestamp(year=theDate.year, month=1, day=1)
                daysyr.append((theDate - new_year_day).days + 1)
            return daysyr
        
    def dfDateMoreInfo(self):
        dfDate = self.dfDate()
        count = 0
        for i in self.dfLAT()['Year'].values:
            if not np.isnan(i): count +=1
        dfDate['cluster']= self.dfLON()['cluster'][0:count]
        dates=[]
        datesDiff=[]
        for i in range(dfDate.shape[0]):
            thisDate=date(int(dfDate['year'].values[i]),int(dfDate['month'].values[i]),int(dfDate['day'].values[i]))

            dates.append(thisDate)
            delta = thisDate-dates[i-1]
            if i ==0:
                datesDiff.append(0)
            else: datesDiff.append(delta.days)
        dfDate['date']= np.asarray(dates)   
        dfDate['datesDiff']=np.asarray(datesDiff)
        return dfDate
    
    def barplot(self, groupby, proportion=True):
        yearCluster = self.yearCluster01()
        
        yearClusterSum = yearCluster.groupby(groupby).sum()

        b=np.zeros(len(yearClusterSum.index.values))
        clusterlist = ['cluster0', 'cluster1','cluster2','cluster3']
        plt.figure()
        for i in range(self.num_clusters):
            
            if not proportion:
                plt.bar(x=yearClusterSum.index.values,height=yearClusterSum[clusterlist[i]].values,bottom=b, label='cluster'+str(i), color = colorList3[i])
                plt.ylabel('count of storms')
                b = np.add(b,yearClusterSum[clusterlist[i]].values)

            if proportion:
                plt.bar(x=yearClusterSum.index.values,height=np.divide(yearClusterSum[clusterlist[i]].values, yearClusterSum['ones'].values),bottom=b, label='cluster'+str(i), color = colorList3[i])
                plt.ylim(0,1)
                plt.ylabel('proportion of storms')
                b = np.add(b,np.divide(yearClusterSum[clusterlist[i]].values, yearClusterSum['ones'].values))
    
        plt.legend()
        plt.show()
        return yearCluster, yearClusterSum






 
    def clusteredBar(self, num_ensembles, full, hurricane_only = False):
        yearCluster = self.yearCluster01()
        
        full2 = full[full['dataset']==self.number]
        if hurricane_only: 
            yearCluster= yearCluster[full2['maxwind']>=33]
        yearClusterSum = yearCluster.groupby('above1995').sum()

        clusterlist = ['cluster0', 'cluster1','cluster2','cluster3']

        clusterChanges = []
        

        for i in range(self.num_clusters):

            height=np.true_divide(yearClusterSum[clusterlist[i]].values, yearClusterSum['ones'].values)
#             clusterChanges.append(np.true_divide(height[1]-height[0],height[0]+height[1]))
            clusterChanges.append(height[1]-height[0])
        return clusterChanges


    
def clusters_list(num_ensembles, clusters1, num_clusters, peak_wind):    
    
    theClusters=[]
    dfLATS = []
    dfLONS = []
    year01s=[]
    if peak_wind ==33: hurricanes_only =True
    else: hurricanes_only = False
    for i in range(num_ensembles+1):
        c= Analyze_Clusters(defCluster = clusters1, number = i, num_clusters= num_clusters, hurricanes_only = hurricanes_only)
        theClusters.append(c)
        dfLATS.append(c.dfLAT())
        year01s.append(c.yearCluster01())
        dfLONS.append(c.dfLON())
        
    return theClusters, dfLATS, dfLONS


def dfAll(num_ensembles, dfLATS, dfLONS, full,peak_wind):
    
    latList =np.asarray([])
    lonList =np.asarray([])
    years=np.asarray([])
    clusters=np.asarray([])
    ens = np.asarray([])
    maxwind = np.asarray([])
    for n in range(num_ensembles+1):
        for i in range(dfLATS[n].shape[0]):
            latList =np.append(latList, dfLATS[n].iloc[i,0:-2].values)

            lonList =np.append(lonList,dfLONS[n].iloc[i,0:-2].values)
            years =np.append(years, dfLATS[n]['Year'].values[i]*np.ones(120))
            maxwind =np.append(maxwind, full[full['dataset']==n]['maxwind'].values[i]*np.ones(120))
            clusters = np.append(clusters, dfLATS[n]['cluster'].values[i]*np.ones(120))
            ens = np.append(ens, n*np.ones(120))
    dfall= pd.DataFrame(np.array([latList, lonList, years, clusters,ens,maxwind]).T, columns=['lat', 'lon', 'Year','cluster','ensemble','maxwind'])
    dfall=dfall.dropna().reset_index().drop('index',axis = 1)
    dfall= dfall[dfall['maxwind']>=peak_wind].reset_index().drop('index',axis = 1)

    return dfall



#get dataframe of genesis points
def get_df_GENS(dfLONS, dfLATS):
    for i in range(len(dfLONS)): 
        if i ==0: 
            dfGENS = pd.DataFrame(dfLONS[i].loc[:,0].values)
            dfGENS.columns = ['firstlon']
            dfGENS['firstlat']= dfLATS[i].loc[:,0].values
            dfGENS['cluster']= dfLONS[i]['cluster'].values
            dfGENS['Year']= dfLONS[i]['Year'].values
            dfGENS['ensemble']=np.ones(len(dfLONS[i]['cluster'].values))*i
        else: 
            dfGENS2 = pd.DataFrame(dfLONS[i].loc[:,0].values)
            dfGENS2.columns = ['firstlon']
            dfGENS2['firstlat']= dfLATS[i].loc[:,0].values
            dfGENS2['cluster']= dfLONS[i]['cluster'].values
            dfGENS2['Year']= dfLONS[i]['Year'].values
            dfGENS2['ensemble']=np.ones(len(dfLONS[i]['cluster'].values))*i

            dfGENS = dfGENS.append(dfGENS2)
    return dfGENS









# visualize cluster locations
def visualize_clusters(full, dfall, dfLATS, dfLONS, num_ensembles, num_clusters): 
    df = dfall
    #first 4 subplots   
    plt.rcParams.update({'font.size': 18})
    fig, ax = plt.subplots(ncols=2,nrows=3,figsize=(18,15), sharex=False,subplot_kw={'projection': ccrs.PlateCarree(central_longitude=0)},dpi = 300)

    for c in range(num_clusters): 
        df2 = df[df['cluster']==c]
        xedges = range(240,360,4)
        yedges = range(0,90,4)
        x = df2['lon']
        y = df2['lat']
        xAll = df['lon']
        yAll = df['lat']
        H, xedges, yedges = np.histogram2d(x, y, bins=(xedges, yedges), normed = False)
        H2, xedges, yedges = np.histogram2d(xAll, yAll, bins=(xedges, yedges), normed = False)


        H = H.T
        H2=H2.T

        H3=H
        X, Y= np.meshgrid(xedges, yedges)
        first = int(c/2)
        second = int((c%2) !=0)
        cmaps = [plumT,midnightblueT,seagreenT,tomatoT]
        im = ax[first,second].pcolormesh(X, Y, np.ma.masked_where(H3==0,H3), cmap = cmaps[c],vmin = 0,vmax = 1000)

        ax[first,second].coastlines(color = 'steelblue')
        ax[first,second].set_extent([-110,10,0,70], crs=ccrs.PlateCarree())
#         ax[first,second].add_feature(cartopy.feature.OCEAN, facecolor='aliceblue')
        ax[first,second].add_feature(cartopy.feature.LAND, facecolor= 'ivory',edgecolor='lightsteelblue')
        ax[first,second].add_feature(cartopy.feature.LAKES, facecolor='white',edgecolor='lightsteelblue')
        ax[first,second].add_feature(cartopy.feature.RIVERS, edgecolor ='aliceblue')

        ax[first,second].legend(title='Cluster '+str(int(c+1)),loc = 'upper left')
        if second ==0:   
            ax[first,second].set_yticks([0,15,30,45,60], crs = ccrs.PlateCarree())
            lat_formatter = LatitudeFormatter()
            ax[first,second].yaxis.set_major_formatter(lat_formatter)

        axins = inset_axes(ax[first,second],width="3%",height="100%", loc='lower left',
                                   bbox_to_anchor=(1.02, 0., 1, 1),
                                   bbox_transform=ax[first,second].transAxes,borderpad=0) 



        if second==1:


            fig.colorbar(im, axins,ticks=[0,200,400,600,800,1000],label= 'Number of points')
        else:
            fig.colorbar(im, axins,ticks=[0,200,400,600,800,1000])

#         if first ==0:
#             if second ==0:
#                 fig.colorbar(im, axins, ticks=[0,200,400,600])
                
    #sublot 5

    for i in [0,1]:
        lon_formatter = LongitudeFormatter()

        ax[2,i].xaxis.set_major_formatter(lon_formatter)
        ax[2,i].set_xticks([ -90, -60,-30,0], crs = ccrs.PlateCarree())
        ax[2,i].coastlines(color = 'steelblue')
        ax[2,i].set_extent([-110,10,0,70], crs=ccrs.PlateCarree())
#         ax[2,i].add_feature(cartopy.feature.OCEAN, facecolor='aliceblue')
        ax[2,i].add_feature(cartopy.feature.LAND, facecolor= 'ivory',edgecolor='steelblue')
        ax[2,i].add_feature(cartopy.feature.LAKES, facecolor='white',edgecolor='lightsteelblue')
        ax[2,i].add_feature(cartopy.feature.RIVERS, edgecolor ='aliceblue')




    ax[2,0].set_yticks([0,15,30,45,60], crs = ccrs.PlateCarree())
    lat_formatter = LatitudeFormatter()
    ax[2,0].yaxis.set_major_formatter(lat_formatter)



    df=dfall

    prevyr = 1
    prevz=0
    clusterseen=np.array([False,False,False,False])

    for i in range(num_ensembles+1):
        for k in range(dfLONS[i].shape[0]):
            lats = dfLATS[i].iloc[k,:-2][~np.isnan(dfLATS[i].iloc[k,:-2])]
            lons = dfLONS[i].iloc[k,:-2][~np.isnan(dfLONS[i].iloc[k,:-2])]
            cluster = dfLATS[i]['cluster'].values[k]
            if (clusterseen[int(cluster)]==False)&((int(cluster)==0)|(clusterseen[int(cluster)-1]==True)):
                ax[2,0].plot(lons,lats,color = colorList3[int(cluster)], linewidth=0.1,transform=ccrs.PlateCarree(), label = 'Cluster '+str(int(cluster+1)))
                np.put(clusterseen, int(cluster), True, mode='clip')
            else: ax[2,0].plot(lons,lats,color = colorList3[int(cluster)], linewidth=0.1,transform=ccrs.PlateCarree())



    leg = ax[2,0].legend(loc = 'upper left')

    for line in leg.get_lines():
        line.set_linewidth(6)


    #last subplot

    df = full
    dfmean=df.groupby(['cluster']).mean()
    dfstd=df.groupby(['cluster']).std()

    for i in range(num_clusters):
        lats = [np.asarray(dfmean.iloc[i,:]['lat1']), np.asarray(dfmean.iloc[i,:]['lat5']),np.asarray(dfmean.iloc[i,:]['lat10'])]
        lons = [np.asarray(dfmean.iloc[i,:]['lon1']), np.asarray(dfmean.iloc[i,:]['lon5']),np.asarray(dfmean.iloc[i,:]['lon10'])]
        laterr = [np.asarray(dfstd.iloc[i,:]['lat1']), np.asarray(dfstd.iloc[i,:]['lat5']),np.asarray(dfstd.iloc[i,:]['lat10'])]
        lonerr = [np.asarray(dfstd.iloc[i,:]['lon1']), np.asarray(dfstd.iloc[i,:]['lon5']),np.asarray(dfstd.iloc[i,:]['lon10'])]
        cluster = i
        ax[2,1].plot(lons, lats,'H-',c=colorList3[cluster],transform=ccrs.PlateCarree(), lw=5,ms=20, label='Cluster '+str(cluster+1))
        ax[2,1].errorbar(lons, lats, xerr=lonerr, c=colorList3[cluster], yerr=laterr,transform=ccrs.PlateCarree(),lw=2)
        ax[2,1].legend(loc = 'upper left')

    plt.subplots_adjust(hspace = 0.07, wspace=0.11)
    plt.show()
    fig.savefig('/tigress/gkortum/thesisSummer/figures/clustersummary.jpg')

      


# visualize cluster locations
def visualize_clusters2(full, dfall, dfLATS, dfLONS, num_ensembles, num_clusters, Hist=False, noHist=False): 
    df = dfall
    #first 4 subplots   
    plt.rcParams.update({'font.size': 12})
    fig, ax = plt.subplots(ncols=1,nrows=4,figsize=(6,20), sharex=False,subplot_kw={'projection': ccrs.PlateCarree(central_longitude=0)},dpi = 300)
    if noHist: num_clusters = num_clusters-1
    if Hist: num_clusters =0
   
    #sublot 5

    for i in range(num_clusters):
        lon_formatter = LongitudeFormatter()

        ax[i].xaxis.set_major_formatter(lon_formatter)
        ax[i].set_xticks([ -90, -60,-30,0], crs = ccrs.PlateCarree())
        ax[i].coastlines(color = 'steelblue')
        ax[i].set_extent([-110,10,0,70], crs=ccrs.PlateCarree())
#         ax[2,i].add_feature(cartopy.feature.OCEAN, facecolor='aliceblue')
        ax[i].add_feature(cartopy.feature.LAND, facecolor= 'ivory',edgecolor='steelblue')
        ax[i].add_feature(cartopy.feature.LAKES, facecolor='white',edgecolor='lightsteelblue')
        ax[i].add_feature(cartopy.feature.RIVERS, edgecolor ='aliceblue')




        ax[i].set_yticks([0,15,30,45,60,70], crs = ccrs.PlateCarree())
        lat_formatter = LatitudeFormatter()
        ax[i].yaxis.set_major_formatter(lat_formatter)

    df=dfall

    prevyr = 1
    prevz=0
    clusterseen=np.array([False,False,False,False])

    for i in range(num_ensembles+1):
        if Hist: i = num_ensembles
        for k in range(dfLONS[i].shape[0]):
            lats = dfLATS[i].iloc[k,:-2][~np.isnan(dfLATS[i].iloc[k,:-2])]
            lons = dfLONS[i].iloc[k,:-2][~np.isnan(dfLONS[i].iloc[k,:-2])]
            cluster = dfLATS[i]['cluster'].values[k]
            if cluster ==0: 
                position =3
            elif cluster ==3: 
                position =0
            else: 
                position = cluster
            if (clusterseen[int(cluster)]==False)&((int(cluster)==0)|(clusterseen[int(cluster)-1]==True)):
                lon_formatter = LongitudeFormatter()

                ax[cluster].xaxis.set_major_formatter(lon_formatter)
                ax[cluster].set_xticks([ -90, -60,-30,0], crs = ccrs.PlateCarree())
                ax[cluster].coastlines(color = 'steelblue')
                ax[cluster].set_extent([-110,10,0,70], crs=ccrs.PlateCarree())
        #         ax[2,i].add_feature(cartopy.feature.OCEAN, facecolor='aliceblue')
                ax[cluster].add_feature(cartopy.feature.LAND, facecolor= 'ivory',edgecolor='steelblue')
                ax[cluster].add_feature(cartopy.feature.LAKES, facecolor='white',edgecolor='lightsteelblue')
                ax[cluster].add_feature(cartopy.feature.RIVERS, edgecolor ='aliceblue')

                ax[cluster].set_yticks([0,15,30,45,60,70], crs = ccrs.PlateCarree())
                lat_formatter = LatitudeFormatter()
                ax[cluster].yaxis.set_major_formatter(lat_formatter)
                ax[position].plot(lons,lats,color = colorList3[int(cluster)], linewidth=0.2,transform=ccrs.PlateCarree(), label = 'cluster '+str(int(cluster+1)))
                np.put(clusterseen, int(cluster), True, mode='clip')
                leg = ax[position].legend(loc = 'upper left')
            else: ax[position].plot(lons,lats,color = colorList3[int(cluster)], linewidth=0.2,transform=ccrs.PlateCarree())



            

    for line in leg.get_lines():
        line.set_linewidth(6)


    #last subplot



    plt.subplots_adjust(hspace = 0.07, wspace=0.11)
    plt.show()
    fig.savefig('/tigress/gkortum/thesisSummer/figures/clustersummary.jpg')

      











