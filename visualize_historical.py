
#import libraries
import numpy as np
import xarray as xr
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import MaxNLocator

from format import *
from restructure_cluster_datasets import *
from analyze_clusters import *

  
def Plot_Gen(dfHistGen,square, normed, thefilter, filename, year_div = 1996):
    # plot genesis locations for historical data
    df = dfHistGen

     #first 4 subplots   
    plt.rcParams.update({'font.size': 16})
    fig, ax = plt.subplots(ncols=3,nrows=1,figsize=(15,12), sharex=False,subplot_kw={'projection': ccrs.PlateCarree(central_longitude=0)},dpi = 300)
    for c in range(3):
        if c == 0: 
            df2 = df[df['Year']<year_div]
        else: df2=df[df['Year']>year_div-1]
        xedges = range(240,360,5)
        yedges = range(0,90,5)
        x = df2['firstlon']
        y = df2['firstlat']
    #         xAll = df['firstlon']
    #         yAll = df['firstlat']
#         if c==2: 
#             if not square: 
#                 x = sp.filters.gaussian_filter(x, sigma = 2, order = 0)
#                 y = sp.filters.gaussian_filter(y, sigma = 2, order = 0)
        H, xedges, yedges = np.histogram2d(x, y, bins=(xedges, yedges), normed = normed)
    #         H2, xedges, yedges = np.histogram2d(xAll, yAll, bins=(xedges, yedges), normed = False)
#         xy = np.column_stack([np.array(xedges).flat, np.array(yedges).flat])

        if normed: 
            factor = 0.0001
            
        else: 
            factor = 10

        H = H.T
#         H = np.pad(H, ((1,0),(0,1)), 'constant')
    #         H2=H2.T
        if c ==0: 
            H3=H/24
            if not square: 
                H3 = pd.DataFrame(H3)
                H3= H3.rolling(thefilter, axis = 0).mean().rolling(thefilter, axis = 1).mean()
                H3 = np.asarray(H3)
#                 print(H3)
#                 H3 = scipy.ndimage.uniform_filter(H3, size = thefilter)
        elif c==1: 
            H3=H/26
            if not square: 
                H3 = pd.DataFrame(H3)
                H3= H3.rolling(thefilter, axis = 0).mean().rolling(thefilter, axis = 1).mean()
                H3 = np.asarray(H3)
#                 H3 = scipy.ndimage.uniform_filter(H3, size = thefilter)
        if c ==0: 
            H4 = H3
        if c==1:
            H5 = H3
        if c ==2: 
            H3 = H5-H4
        X, Y= np.meshgrid(xedges, yedges)
#         try: 
#             grid_z = scipy.interpolate.griddata(xy,H3,(X, Y), method='linear')
#         except ValueError: 
#             print("xy = ")
#             print(xy.shape)
#             print("H3 = ")
#             print(H3.shape)
#             print("X = ")
#             print(X.shape)
#             print("Y = ")
#             print(Y.shape)
#             return
#         if not square: H3 = grid_z

        if c <2: 
            im = ax[c].pcolormesh(X, Y, H3, cmap = midnightblueT, vmin = 0, vmax =0.6*factor)
#             extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
#             im = ax[c].imshow(H3,extent = extent, cmap = midnightblueT, vmin = 0, vmax =0.6*factor)
        elif c ==2: 

            im = ax[c].pcolormesh(X, Y, np.ma.masked_where(H3==0,H3),  cmap ='seismic', vmin = -0.5*factor, vmax = 0.5*factor)

#                 levels = MaxNLocator(nbins=15).tick_values(H3.min(), H3.max())
#                 cf = ax[c].contourf(X[:-1, :-1] + dx/2.,
#                   Y[:-1, :-1] + dy/2., H3, levels=levels,
#                   cmap='seismic')
#             else:
# #         im = ax[c].imshow(H3, extent=extent, cmap ='seismic', vmin = -0.5*factor, vmax = 0.5*factor)
#                 im = ax[c].pcolormesh(X, Y, np.ma.masked_where(H3==0,H3), shading = 'gouraud', cmap ='seismic', vmin = -0.5*factor, vmax = 0.5*factor)
        ax[c].coastlines(color = 'steelblue')
        ax[c].set_extent([-110,10,0,70], crs=ccrs.PlateCarree())
    #         if c <2: 
    #             ax[c].add_feature(cartopy.feature.OCEAN, facecolor='aliceblue')
        ax[c].add_feature(cartopy.feature.LAND, facecolor= 'ivory',edgecolor='lightsteelblue')
        ax[c].add_feature(cartopy.feature.LAKES, facecolor='white',edgecolor='lightsteelblue')
        ax[c].add_feature(cartopy.feature.RIVERS, edgecolor ='aliceblue')
        titles = ['pre-'+ str(year_div), 'post-'+ str(year_div),'post-'+ str(year_div)+' minus pre-'+ str(year_div)]
        ax[c].set_title(titles[c])
        if c==0: 
            ax[c].set_yticks([0,15,30,45,60,70], crs = ccrs.PlateCarree())
            lat_formatter = LatitudeFormatter()
            ax[c].yaxis.set_major_formatter(lat_formatter)

        axins = inset_axes(ax[c],width="3%",height="100%", loc='lower left',
                                   bbox_to_anchor=(1.02, 0., 1, 1),
                                   bbox_transform=ax[c].transAxes,borderpad=0) 
        if c < 2:
            fig.colorbar(im, axins,ticks = [0,0.2,0.4,0.6])
        else: 
            fig.colorbar(im, axins,ticks = [-0.5,0,0.5], label= str(filename))

        lon_formatter = LongitudeFormatter()

        ax[c].xaxis.set_major_formatter(lon_formatter)
        ax[c].set_xticks([ -90, -60,-30,0], crs = ccrs.PlateCarree())

    

    
def Plot_Gen_one(dfHistGen,square, normed, thefilter, filename, year_div = 1996, title = "title"):
    # plot genesis locations for historical data
    df = dfHistGen

     #first 4 subplots   
    plt.rcParams.update({'font.size': 16})
    fig, ax = plt.subplots(ncols=1,nrows=1,figsize=(8,12), sharex=False,subplot_kw={'projection': ccrs.PlateCarree(central_longitude=0)},dpi = 300)
    plt.subplots_adjust(wspace=0.1, hspace=0)
    for c in range(3):
        if c == 0: 
            df2 = df[df['Year']<year_div]
        else: df2=df[df['Year']>year_div-1]
        xedges = range(240,360,5)
        yedges = range(0,90,5)
        x = df2['firstlon']
        y = df2['firstlat']
    #         xAll = df['firstlon']
    #         yAll = df['firstlat']
#         if c==2: 
#             if not square: 
#                 x = sp.filters.gaussian_filter(x, sigma = 2, order = 0)
#                 y = sp.filters.gaussian_filter(y, sigma = 2, order = 0)
        H, xedges, yedges = np.histogram2d(x, y, bins=(xedges, yedges), normed = normed)
    #         H2, xedges, yedges = np.histogram2d(xAll, yAll, bins=(xedges, yedges), normed = False)
#         xy = np.column_stack([np.array(xedges).flat, np.array(yedges).flat])
     
        factor = 0.002


        H = H.T
#         H = np.pad(H, ((1,0),(0,1)), 'constant')
    #         H2=H2.T
        if c ==0: 
            H3=H
#             H3=H/26
            if not square: 
                H3 = pd.DataFrame(H3)
                H3= H3.rolling(thefilter, axis = 0).mean().rolling(thefilter, axis = 1).mean()
                H3 = np.asarray(H3)
#                 print(H3)
#                 H3 = scipy.ndimage.uniform_filter(H3, size = thefilter)
        elif c==1: 
            H3=H
#             H3=H/26
            if not square: 
                H3 = pd.DataFrame(H3)
                H3= H3.rolling(thefilter, axis = 0).mean().rolling(thefilter, axis = 1).mean()
                H3 = np.asarray(H3)
#                 H3 = scipy.ndimage.uniform_filter(H3, size = thefilter)
        if c ==0: 
            H4 = H3
        if c==1:
            H5 = H3
        if c ==2: 
            H3 = H5-H4
        X, Y= np.meshgrid(xedges, yedges)

        if c ==2: 

            im = ax.pcolormesh(X, Y, np.ma.masked_where(H3==0,H3),  cmap ='seismic', vmin = -0.5*factor, vmax = 0.5*factor)

            ax.coastlines(color = 'steelblue')
            ax.set_extent([-110,10,0,70], crs=ccrs.PlateCarree())
        #         if c <2: 
        #             ax[c].add_feature(cartopy.feature.OCEAN, facecolor='aliceblue')
            ax.add_feature(cartopy.feature.LAND, facecolor= 'ivory',edgecolor='lightsteelblue')
            ax.add_feature(cartopy.feature.LAKES, facecolor='white',edgecolor='lightsteelblue')
            ax.add_feature(cartopy.feature.RIVERS, edgecolor ='aliceblue')

            ax.set_title(title)

            ax.set_yticks([0,15,30,45,60], crs = ccrs.PlateCarree())
            lat_formatter = LatitudeFormatter()
            ax.yaxis.set_major_formatter(lat_formatter)

            axins = inset_axes(ax,width="3%",height="100%", loc='lower left',
                                       bbox_to_anchor=(1.02, 0., 1, 1),
                                       bbox_transform=ax.transAxes,borderpad=0) 
            labels = ['Point density','Point density','Change in point density']

            cbar = fig.colorbar(im, axins)
            cbar.set_label(label = labels[c])
            cbar.formatter.set_powerlimits((0, 0))

            cbar.set_ticks([-0.001,-0.0005,0,0.0005,0.001])

            lon_formatter = LongitudeFormatter()

            ax.xaxis.set_major_formatter(lon_formatter)
            ax.set_xticks([ -90, -60,-30,0], crs = ccrs.PlateCarree())



        fig.savefig('/tigress/gkortum/thesisSummer/figures/'+str(filename)+'_historical.jpg')

def gen_plot_historical(num_ensembles,dfGENS, full,peak_thresh, square, normed, thefilter=0, one = False, title = "title"):

#     dfGENS2 = dfGENS[full['maxwind']>=peak_thresh]
    dfGENS2 = dfGENS
    dfHistGen = dfGENS2[dfGENS2['ensemble']==num_ensembles]

#     years = []
#     genLONs = []
#     genLATs=[]
#     for YR in range(1971,2020,1): 
# #         for ZSTORM in dsHist2.ZSTORM1.values: 
# #             if np.sum(np.isnan(dsHist3WIND.sel(YRT=YR, ZSTORM1=ZSTORM, YENS=num_ensembles+1).WIND))<(120-hours_at_wind_thresh/6): 

# #                 theindex = np.asarray(np.where(np.invert(np.isnan(dsHist2WIND.sel(YRT=YR, ZSTORM1=ZSTORM, YENS=num_ensembles +1).WIND)))[0])[0]
# #                 thelon= float(np.asarray(dsHist2WIND.sel(YRT=YR, ZSTORM1=ZSTORM, YENS=num_ensembles+1).LON[theindex]))
# #                 thelat= float(np.asarray(dsHist2WIND.sel(YRT=YR, ZSTORM1=ZSTORM, YENS=num_ensembles+1).LAT[theindex]))
#                 years.append(YR)
#                 genLONs.append(thelon)
#                 genLATs.append(thelat)
#     dfHistGen=pd.DataFrame(data={'Year':np.array(years), 'firstlon':np.array(genLONs),'firstlat':np.array(genLATs)})
    if one: 
        Plot_Gen_one(dfHistGen, square, normed, thefilter, 'genesis_distribution', title = title)
    else: 
        Plot_Gen(dfHistGen, square, normed, thefilter, 'genesis_distribution')




def all_plot_historical(num_ensembles, dfall, peak_thresh, square, normed, thefilter=0, year_div = 1996, one = False, title = 'title'):

#     dfall2 = dfall[dfall['maxwind']>=peak_thresh]
    dfall2 = dfall
    dfall2=dfall2[dfall2['ensemble']==num_ensembles]

    dfHistGen=pd.DataFrame(data={'Year':dfall2['Year'].values, 'firstlon':dfall2['lon'].values,'firstlat':dfall2['lat'].values})
    if one: 
        Plot_Gen_one(dfHistGen, square, normed, thefilter,'point_distribution', year_div, title = title)
    else:
        Plot_Gen(dfHistGen, square, normed, thefilter,'point_distribution', year_div)
    
def all_plot_model(num_ensembles, dfall, peak_thresh, square, normed, thefilter=0, year_div = 1996, one = False, title = 'title'):

#     dfall2 = dfall[dfall['maxwind']>=peak_thresh]
    dfall2 = dfall
    dfall2=dfall2[dfall2['ensemble']<num_ensembles]

    dfHistGen=pd.DataFrame(data={'Year':dfall2['Year'].values, 'firstlon':dfall2['lon'].values,'firstlat':dfall2['lat'].values})
    if one: 
        Plot_Gen_one(dfHistGen, square, normed, thefilter,'point_distribution', year_div, title = title)
    else: 
        Plot_Gen(dfHistGen, square, normed, thefilter,'point_distribution', year_div)

def gen_plot_model(num_ensembles,dfGENS, full,peak_thresh, square, normed, thefilter=0, one = False, title = "title"):

#     dfGENS2 = dfGENS[full['maxwind']>=peak_thresh]
    dfGENS2 = dfGENS
    dfHistGen = dfGENS2[dfGENS2['ensemble']<num_ensembles]


    if one: 
        Plot_Gen_one(dfHistGen, square, normed, thefilter, 'genesis_distribution', title = title)
    else: 
        Plot_Gen(dfHistGen, square, normed, thefilter, 'genesis_distribution')
    
def timeseries(num_ensembles1,theClusters,full, num_ensembles,wind_thresh= 17):    
    years = theClusters[num_ensembles1].dfLON()['Year']
    lons =theClusters[num_ensembles1].dfLON().iloc[:,:-2]
    lats = theClusters[num_ensembles1].dfLAT().iloc[:,:-2]
    lons2=lons.mean(skipna=True, axis = 1)
    lats2=lats.mean(skipna=True, axis = 1)
    full2 = full[full['dataset']==num_ensembles]

#     return pd.DataFrame(np.array([lons2,years]).T)

    toplot=pd.DataFrame(np.array([lons2,years]).T).groupby(1).mean().rolling(5, min_periods = 1).mean().values
    toplot3=pd.DataFrame(np.array([lats2,years]).T).groupby(1).mean().rolling(5, min_periods = 1).mean().values
    toplot2=pd.DataFrame(np.array([lons2,years]).T).groupby(1).count().rolling(5, min_periods = 1).mean().values

    fig, a = plt.subplots(ncols=1,nrows=3,figsize=(10,13), sharex=True)
    k = 52
    c = 2022
    y1 = 1970
    try: 
 
        a[0].fill_between(range(y1,c,1),toplot2.reshape([k,]),np.ones(k)*toplot2.mean(),color = 'seagreen', alpha = 0.3)
        a[0].scatter(years, pd.DataFrame(np.array([lons2,years]).T).groupby(1).count().values)

    except: print('')
    a[0].plot(range(y1,c,1),np.ones(k)*toplot2.mean(), c = 'k')
    a[0].plot(range(y1,c,1),toplot2.reshape([k,]), c = 'seagreen')
    a[0].plot([1995,1995],[0,1000], linestyle = '--', c ='slategrey')
    a[0].set_ylim([0,10])
    a[0].yaxis.set_label_coords(-.1, .5)
    a[0].set_ylabel('Number of hurricanes')

    a[1].fill_between(range(y1,c,1),toplot.reshape([k,]),np.ones(k)*toplot.mean(), color = 'navy',alpha = 0.3)
    a[1].plot(range(y1,c,1),np.ones(k)*toplot.mean(), c = 'k')
    a[1].plot(range(y1,c,1),toplot.reshape([k,]), c = 'navy')
#     a[1].set_yticks([285,290,295,300])
    a[1].plot([1995,1995],[0,1000], linestyle = '--', c ='slategrey')
    a[1].set_ylim([280,305])
    a[1].yaxis.set_label_coords(-0.1, .5)
    a[1].set_ylabel('Average longitude')

    a[2].fill_between(range(y1,c,1),toplot3.reshape([k,]),np.ones(k)*toplot3.mean(), color = 'tomato',alpha = 0.3)
    a[2].plot(range(y1,c,1),np.ones(k)*toplot3.mean(), c = 'k')
    a[2].plot(range(y1,c,1),toplot3.reshape([k,]), c = 'tomato')
    a[2].set_yticks([20,25,30,35])
    a[2].plot([1995,1995],[0,1000], linestyle = '--', c ='slategrey')
    a[2].set_ylim([20,35])
    a[2].yaxis.set_label_coords(-.1, .5)
    a[2].set_xticks(range(y1,2025,5))
    a[2].set_ylabel('Average latitude')
    a[2].set_xlabel('Year')
    fig.savefig('/tigress/gkortum/thesisSummer/figures/intoseries.jpg', dpi = 300)
