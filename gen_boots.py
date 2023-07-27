
import pickle

import xarray as xr
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy as cartopy
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import cftime as cftime
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.gridspec as gridspec
import statsmodels.api as sm
import statsmodels
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

from datetime import date

warnings.filterwarnings('ignore')
import statsmodels.api as sm

import ipynb
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})

    
def getdf(hibestens, full):
    for i in range(len(hibestens)): 
        year = 1970+i
        ensemble = int(hibestens[i])
        if i ==0:
            hibestdf = full[(full['Year']==year)&(full['dataset']==ensemble)]
        else: 
            hibestdf = hibestdf.append(full[(full['Year']==year)&(full['dataset']==ensemble)])

    hibestdf=hibestdf.reset_index().drop('index',axis =1)   
    return hibestdf

def get_hilow_bestdfsGEN(full,hibestenses,lowbestenses): 
    hibestdfsGEN=[]
    lowbestdfsGEN=[]
    for b in range(0,100,1):
#         if b==1: 
#             thestring=''
#         else: 
#             thestring= '_'+str(b)

#         with open('/tigress/gkortum/ThesisCurrent/lowbestens'+thestring, 'rb') as f:
#             lowbestens = pickle.load(f)

#         with open('/tigress/gkortum/ThesisCurrent/hibestens'+thestring, 'rb') as f:
#             hibestens= pickle.load(f)
        hibestens = hibestenses[b]
        lowbestens = lowbestenses[b]
        hibestdfsGEN.append(getdf(hibestens, full))
        lowbestdfsGEN.append(getdf(lowbestens, full))
    return hibestdfsGEN,lowbestdfsGEN
    

def getH(df): 
    xedges = range(240,360,4)
    yedges = range(0,90,4)
    X, Y= np.meshgrid(xedges, yedges)
    above=[]
    for i in range(df.shape[0]):
        if df['Year'].values[i]< 1996: 
            above.append(0)
        else: above.append(1)
    df['above']= above    
        
#     
 
    df2 = df.copy()
    dfabove = df2[df2['above']==1]
    dfbelow = df2[df2['above']==0]


#         x = df2['firstlon']
#         y = df2['firstlat']
#         xAll = df['firstlon']
#         yAll = df['firstlat']
    xa = dfabove['lon1']
    ya = dfabove['lat1']
    xb = dfbelow['lon1']
    yb = dfbelow['lat1']

    Ha, xedges, yedges = np.histogram2d(xa, ya, bins=(xedges, yedges), normed = True)
    Hb, xedges, yedges = np.histogram2d(xb, yb, bins=(xedges, yedges),normed = True)
    Ha = pd.DataFrame(Ha)
    Ha= Ha.rolling(2, axis= 0).mean().rolling(2, axis =1).mean()
    Ha = np.asarray(Ha)
    Hb = pd.DataFrame(Hb)
    Hb= Hb.rolling(2, axis =0).mean().rolling(2, axis =1).mean()
    Hb = np.asarray(Hb)

#         H2, xedges, yedges = np.histogram2d(xAll, yAll, bins=(xedges, yedges))
        
    # #     ax = fig.add_subplot(132, title='pcolormesh: actual edges', aspect='equal',subplot_kw={'projection': ccrs.PlateCarree(central_longitude=0)} )


    #     h =ax.hist2d(x, y, bins=[xedges,yedges])
    Ha = Ha.T
    Hb = Hb.T
    return Ha-Hb


def densityAll4(hibestdfsGEN, lowbestdfsGEN):
    xedges = range(240,360,4)
    yedges = range(0,90,4)
    X, Y= np.meshgrid(xedges, yedges)
    HsHi=[]
    HsLow = []
    HsDiffs = []
    for i in range(100): 
#         print(i)
        hidf = hibestdfsGEN[i]
        lowdf = lowbestdfsGEN[i]
        HsHi.append(getH(hidf))
        HsLow.append(getH(lowdf))
        HsDiffs.append(getH(hidf)-getH(lowdf))
    Hshimean = np.array(HsHi).mean(axis = 0)
    Hshistd = np.true_divide(np.array(HsHi).std(axis = 0),np.sqrt(100))
    Hslowmean = np.array(HsLow).mean(axis =0)
    Hslowstd = np.true_divide(np.array(HsLow).std(axis = 0),np.sqrt(100))
    Hsdiffmean = np.array(HsDiffs).mean(axis =0)
    Hsdiffste = np.true_divide(np.array(HsDiffs).std(axis =0),np.sqrt(100))


    fig, ax = plt.subplots(ncols=3,nrows=2,figsize=(24,10), sharex=False,subplot_kw={'projection': ccrs.PlateCarree(central_longitude=0)}, dpi = 300)
    plt.subplots_adjust(hspace = 0.2, wspace=0.22) 
#     fig.tight_layout(pad = 2)
    seagreen=matplotlib.colors.LinearSegmentedColormap.from_list('seagreen', ['white','seagreen','k'], N=256, gamma=1.0)
    
    Hshis = [Hshimean, Hshistd]
    Hslows = [Hslowmean,Hslowstd]
    Hsdiffs = [Hsdiffmean, Hsdiffste]
    cmaps=['seismic', seagreen]
    vmins = [-0.0006,0]
    vmaxs= [0.0006,0.00006]
    vmaxs2=[0.0006,0.00006]
#     ticks=[[-0.0006,0,0.0006], [0,0.00006]]
    ticks2=[[-0.0006,-0.0003,0,0.0003,0.0006], [0,0.00002,0.00004,0.00006]]
#     theLetters = np.array([['A','B','C'],['D','E','F']])
    for c in range(2):
    
        if c ==0: 
            mask_n= 0.000001
        elif c==1: 
            mask_n=0.00000001
  
        im = ax[c,0].pcolormesh(X, Y, np.ma.masked_where(np.abs(Hshis[c])< mask_n,Hshis[c]), cmap = cmaps[c],vmin = vmins[c], vmax =vmaxs[c]) 
        imb = ax[c,1].pcolormesh(X, Y, np.ma.masked_where(np.abs(Hslows[c])<mask_n,Hslows[c]), cmap = cmaps[c],vmin = vmins[c], vmax =vmaxs[c]) 
        imc = ax[c,2].pcolormesh(X, Y, np.ma.masked_where(np.abs(Hsdiffs[c])<mask_n,Hsdiffs[c]), cmap=cmaps[c],vmin = vmins[c], vmax =vmaxs2[c]) 
    #     ax[c, 0].set_title('cluster = '+str(c) + ' pre 1995')
    #     ax[c, 1].set_title('cluster = '+str(c) + ' post 1995')
    #     ax[c, 2].set_title('cluster = '+str(c) + ' post minus pre')
        ims = [im,imb,imc]
        
        for k in range(3):



            axins = inset_axes(ax[c,k],width="3%",height="100%", loc='lower left',
                               bbox_to_anchor=(1.02, 0., 1, 1),
                               bbox_transform=ax[c,k].transAxes,borderpad=0) 
            sfmt=matplotlib.ticker.ScalarFormatter(useMathText=True) 
            sfmt.set_powerlimits((0, 0))
            cbar = fig.colorbar(ims[k], axins, ticks = ticks2[c],format=sfmt)
            if k ==2: 
                cbar.set_label(label = 'Change in point density',labelpad=c*14+5)


#                 cbar.formatter.set_powerlimits((0, 0))

#             else: 
#                 fig.colorbar(ims[k], axins,ticks = ticks[c])
            ax[c,k].coastlines(color = 'k')
#             ax[c,k].legend(title = theLetters[c,k], loc = 'upper left',fontsize = 'medium')

            if k ==0:   
                ax[c,k].set_yticks([0,15,30,45,60], crs = ccrs.PlateCarree())
                lat_formatter = LatitudeFormatter()
                ax[c,k].yaxis.set_major_formatter(lat_formatter)

            lon_formatter = LongitudeFormatter()

            ax[c,k].xaxis.set_major_formatter(lon_formatter)

            ax[c,k].set_xticks([ -90, -60,-30,0], crs = ccrs.PlateCarree())
            ax[c,k].set_extent([-110,10,0,70], crs=ccrs.PlateCarree())
            ax[c,k].coastlines(color = 'steelblue')
            ax[c,k].set_extent([-110,10,0,70], crs=ccrs.PlateCarree())
            
            ax[c,k].add_feature(cartopy.feature.LAND, facecolor= 'ivory',edgecolor='steelblue')
            ax[c,k].add_feature(cartopy.feature.LAKES, facecolor='white',edgecolor='lightsteelblue')
            ax[c,k].add_feature(cartopy.feature.RIVERS, edgecolor ='aliceblue')

    ax[0,0].set_title('(A): Westward shift', fontsize = 17)
    ax[0,1].set_title('(B): Eastward shift', fontsize = 17)
    ax[0,2].set_title('(A) minus (B)', fontsize = 17)



    fig.savefig('figures/GENboots.jpg')