
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
import cmocean
from datetime import date

warnings.filterwarnings('ignore')
import statsmodels.api as sm

import ipynb
# import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})

def load_data():
    with open('data/ensemblesSLP', 'rb') as f:
        ensemblesSLP = pickle.load(f)

    hibestenses= []
    lowbestenses = []
    for b in range(0,100,1):
#         if b==1: 
#             thestring=''
#         else: 
        thestring= '_'+str(b)

        with open('data/bootstrap ensembles 3/lowbestens'+thestring, 'rb') as f:
            lowbestens = pickle.load(f)

        with open('data/bootstrap ensembles 3/hibestens'+thestring, 'rb') as f:
            hibestens= pickle.load(f)
        hibestenses.append(hibestens)
        lowbestenses.append(lowbestens)
    return hibestenses,lowbestenses, ensemblesSLP

def plotSLP(SLPhi):

    SLPhiPre=SLPhi.isel(time=slice(0,25,1)).mean('time')
    SLPhiPost = SLPhi.isel(time=slice(26,51,1)).mean('time')
    SLPhidiff = SLPhiPost-SLPhiPre
    return SLPhidiff, SLPhiPre, SLPhiPost



def getdiffs(hibestenses, lowbestenses, bs,ensemblesSLP):
    diffsHi=[]
    diffsLow = []
    SLPshiPre = []
    SLPshiPost=[]
    SLPslowPre=[]
    SLPslowPost=[]
    diffsHig=[]
    diffsLowg = []
    SLPshiPreg = []
    SLPshiPostg=[]
    SLPslowPreg=[]
    SLPslowPostg=[]

    for b in range(bs):
#         print(b)
        SLPshi=[]
        SLPslow=[]
        SLPshig=[]

        SLPslowg=[]
        for i in range(len(hibestenses[b])-1):
            theEns = int(hibestenses[b][i])
            try: 
                theSLP = ensemblesSLP[theEns][i]['SLP'].isel(time = slice(5,11,1)).mean('time')
            except IndexError: 
                theSLP = ensemblesSLP[theEns][i]['SLP'].isel(time = slice(5,11,1)).mean('time')
            theSLPgrad = ensemblesSLP[theEns][i].isel(time = slice(5,11,1)).mean('time')
            theSLPgrad['xgrad_divide']=theSLPgrad['SLPxgrad']/np.sin(np.radians(theSLPgrad['grid_yt']))
            theSLPgrad['ygrad_divide']=-theSLPgrad['SLPygrad']/np.sin(np.radians(theSLPgrad['grid_yt']))



            SLPshi.append(theSLP)
            SLPshig.append(theSLPgrad)

            theEnsL = int(lowbestenses[b][i])
            theSLPL = ensemblesSLP[theEnsL][i]['SLP'].isel(time = slice(5,11,1)).mean('time')
            theSLPLgrad = ensemblesSLP[theEnsL][i].isel(time = slice(5,11,1)).mean('time')
            theSLPLgrad['xgrad_divide']=theSLPLgrad['SLPxgrad']/np.sin(np.radians(theSLPLgrad['grid_yt']))
            theSLPLgrad['ygrad_divide']=-theSLPLgrad['SLPygrad']/np.sin(np.radians(theSLPLgrad['grid_yt']))



            SLPslow.append(theSLPL)
            SLPslowg.append(theSLPLgrad)

        SLPhi = xr.concat(SLPshi,dim = 'time')
        SLPhig = xr.concat(SLPshig,dim = 'time')

        SLPlow = xr.concat(SLPslow,dim = 'time')
        SLPlowg = xr.concat(SLPslowg,dim = 'time')


        SLPhidiff, SLPhiPre,SLPhiPost = plotSLP(SLPhi)
        SLPlowdiff, SLPlowPre, SLPlowPost = plotSLP(SLPlow)
        SLPhidiffg, SLPhiPreg,SLPhiPostg = plotSLP(SLPhig)
        SLPlowdiffg, SLPlowPreg, SLPlowPostg = plotSLP(SLPlowg)

#         SLPshiPre.append(SLPhiPre.values.flatten())
#         SLPslowPre.append(SLPlowPre.values.flatten())
#         SLPshiPost.append(SLPhiPost.values.flatten())
#         SLPslowPost.append(SLPlowPost.values.flatten())
        SLPshiPre.append(SLPhiPre)
        SLPslowPre.append(SLPlowPre)
        SLPshiPost.append(SLPhiPost)
        SLPslowPost.append(SLPlowPost)
        diffsHi.append(SLPhidiff)
        diffsLow.append(SLPlowdiff)
        
        SLPshiPreg.append(SLPhiPreg)
        SLPslowPreg.append(SLPlowPreg)
        SLPshiPostg.append(SLPhiPostg)
        SLPslowPostg.append(SLPlowPostg)
        diffsHig.append(SLPhidiffg)
        diffsLowg.append(SLPlowdiffg)

    return diffsHi,diffsLow, SLPshiPre, SLPshiPost, SLPslowPre, SLPslowPost,diffsHig,diffsLowg, SLPshiPreg, SLPshiPostg, SLPslowPreg, SLPslowPostg


def plot_SLP(hibestenses,lowbestenses, ensemblesSLP, ensmean=False):
    theNum = 100
    if ensmean ==True: 
        theNum =10
        hibestenses = []
        for k in range(theNum): 
            hibestenses.append(np.ones(len(lowbestenses))*k)
        lowbestenses = hibestenses                       
    diffsHi,diffsLow, SLPshiPre, SLPshiPost, SLPslowPre, SLPslowPost,diffsHig,diffsLowg, SLPshiPreg, SLPshiPostg, SLPslowPreg, SLPslowPostg = getdiffs(hibestenses, lowbestenses, theNum, ensemblesSLP)
    diffsList2=[]
    diffsList=[]
    diffsList2g=[]
    diffsListg=[]
    for i in range(len(diffsHi)):
        diffsList.append(diffsHi[i]-diffsLow[i])
        diffsList2.append(SLPshiPre[i]-SLPslowPre[i])
        diffsListg.append(diffsHig[i]-diffsLowg[i])
        diffsList2g.append(SLPshiPreg[i]-SLPslowPreg[i])

    hiPre= xr.concat(SLPshiPre, dim ='b').mean('b')
    lowPre= xr.concat(SLPslowPre, dim ='b').mean('b')
    hiPreg= xr.concat(SLPshiPreg, dim ='b').mean('b')
    lowPreg= xr.concat(SLPslowPreg, dim ='b').mean('b')


    himean =xr.concat(diffsHi, dim ='b').mean('b')
    lowmean =xr.concat(diffsLow, dim ='b').mean('b')
    himeang =xr.concat(diffsHig, dim ='b').mean('b')
    lowmeang =xr.concat(diffsLowg, dim ='b').mean('b')


    diffmean = xr.concat(diffsList, dim = 'b').mean('b')
    diffPre = xr.concat(diffsList2, dim = 'b').mean('b')
    diffmeang = xr.concat(diffsListg, dim = 'b').mean('b')
    diffPreg = xr.concat(diffsList2g, dim = 'b').mean('b')







    seagreen=matplotlib.colors.LinearSegmentedColormap.from_list('seagreen', ['white','seagreen','k'], N=256, gamma=1.0)
    if ensmean: 
        return

                         
    elif ensmean ==False:                            
        fig, ax = plt.subplots(ncols=3,nrows=2,figsize=(24,10), sharex=False,subplot_kw={'projection': ccrs.PlateCarree(central_longitude = 0)}, dpi = 300)
        plt.subplots_adjust(hspace = 0.2, wspace=0.22)  
    #     fig.tight_layout(pad = 3)
        vmins = [-0.8, 1005]
        vmaxs = [0.8, 1025]
        ticks = [[-0.8,-0.4,0,0.4,0.8], [1005,1010,1015,1020,1025]]
        ticks2= [[-0.8,-0.4,0,0.4,0.8], [1005,1010,1015,1020,1025]]
        vmins2=[-0.8,1005]
        vmaxs2=[0.8,1025]
        cmaps = ['seismic', cmocean.cm.dense]
        for i in range(2):
            if i ==0: 
                diffsHi1 = himean
                diffsLow1 = lowmean
                diffsdiffs = diffmean
    #             theletters =[' A ',' B ',' C ']
                b = diffsHi1.plot.contourf(levels = 40,vmin = vmins[i],vmax = vmaxs[i], cmap = cmaps[i], ax = ax[i,0], add_colorbar=False)
                axins2 = inset_axes(ax[i,0],width="3%",height="100%", loc='lower left',bbox_to_anchor=(1.02, 0., 1, 1),bbox_transform=ax[i,0].transAxes,borderpad=0)


                plt.colorbar(b,axins2,ticks = ticks2[i])
#                 skip = dict(grid_xt=slice(None,None,40),grid_yt=slice(None,None,40))
#                 himeang2=himeang.isel(skip)
#                 b = himeang2.plot.quiver(x='grid_xt', y='grid_yt',u='ygrad_divide', v='xgrad_divide',ax = ax[i,0])
                ax[i,0].coastlines()

                ax[i,0].set_extent([-110,10,0,70], crs=ccrs.PlateCarree())
                skip = dict(grid_xt=slice(None,None,40),grid_yt=slice(None,None,40))
                himeang2=himeang.isel(skip)
                himeang3=himeang2.sel(grid_yt=slice(10,90,1))
                b = himeang3.plot.quiver(x='grid_xt', y='grid_yt',u='ygrad_divide', v='xgrad_divide',ax = ax[i,0])
        #         axins = inset_axes(ax[i,0],width="3%",height="100%", loc='lower left',bbox_to_anchor=(1.02, 0., 1, 1),bbox_transform=ax[i,0].transAxes,borderpad=0)
        #         plt.colorbar(b,axins,ticks = ticks[i])
        #         ax[i,0].legend(title = theletters[0], loc = 'upper left',fontsize = 'medium')

                b=diffsLow1.plot.contourf(levels = 40,vmin = vmins[i],vmax = vmaxs[i], cmap = cmaps[i], ax = ax[i,1], add_colorbar=False)
                axins3 = inset_axes(ax[i,1],width="3%",height="100%", loc='lower left',bbox_to_anchor=(1.02, 0., 1, 1),bbox_transform=ax[i,1].transAxes,borderpad=0)


                plt.colorbar(b,axins3,ticks = ticks2[i])
                ax[i,1].coastlines()

                ax[i,1].set_extent([-110,10,0,70], crs=ccrs.PlateCarree())

                lowmeang2=lowmeang.isel(skip)
                lowmeang3=lowmeang2.sel(grid_yt=slice(10,90,1))
                b = lowmeang3.plot.quiver(x='grid_xt', y='grid_yt',u='ygrad_divide', v='xgrad_divide',ax = ax[i,1])
        #         axins = inset_axes(ax[i,1],width="3%",height="100%", loc='lower left',bbox_to_anchor=(1.02, 0., 1, 1),bbox_transform=ax[i,1].transAxes,borderpad=0)
        #         plt.colorbar(b,axins,ticks = ticks[i])
        #         ax[i,1].legend(title = theletters[1], loc = 'upper left',fontsize = 'medium')

                b=diffsdiffs.plot.contourf(levels = 40,vmin =vmins2[i],vmax = vmaxs2[i], cmap =cmaps[i], ax = ax[i,2], add_colorbar=False)
                ax[i,2].coastlines()

                ax[i,2].set_extent([-110,10,0,70], crs=ccrs.PlateCarree())
                axins = inset_axes(ax[i,2],width="3%",height="100%", loc='lower left',bbox_to_anchor=(1.02, 0., 1, 1),bbox_transform=ax[i,2].transAxes,borderpad=0)


                plt.colorbar(b,axins,ticks = ticks2[i], label = 'mbar')
                diffmeang2=diffmeang.isel(skip)
                diffmeang3=diffmeang2.sel(grid_yt=slice(10,90,1))
                b = diffmeang3.plot.quiver(x='grid_xt', y='grid_yt',u='ygrad_divide', v='xgrad_divide',ax = ax[i,2])




            elif i ==1: 
                diffsHi1=hiPre

                diffsLow1=lowPre
                diffsdiffs = diffPre
    #             theletters =[' D ',' E ',' F ']
                b = diffsHi1.plot.contourf(levels = 40,vmin = vmins2[i],vmax = vmaxs2[i], cmap = cmaps[i], ax = ax[i,0], add_colorbar=False)
                ax[i,0].coastlines()

                ax[i,0].set_extent([-110,10,0,70], crs=ccrs.PlateCarree())
                axins2 = inset_axes(ax[i,0],width="3%",height="100%", loc='lower left',bbox_to_anchor=(1.02, 0., 1, 1),bbox_transform=ax[i,0].transAxes,borderpad=0)


                plt.colorbar(b,axins2,ticks = ticks2[i])
                skip = dict(grid_xt=slice(None,None,40),grid_yt=slice(None,None,40))
                hiPreg2=hiPreg.isel(skip)
                hiPreg3=hiPreg2.sel(grid_yt=slice(10,90,1))
                b = hiPreg3.plot.quiver(x='grid_xt', y='grid_yt',u='ygrad_divide', v='xgrad_divide',ax = ax[i,0])
        #         axins = inset_axes(ax[i,0],width="3%",height="100%", loc='lower left',bbox_to_anchor=(1.02, 0., 1, 1),bbox_transform=ax[i,0].transAxes,borderpad=0)
        #         plt.colorbar(b,axins,ticks = ticks[i])
        #         ax[i,0].legend(title = theletters[0], loc = 'upper left',fontsize = 'medium')
                b=diffsLow1.plot.contourf(levels = 40,vmin = vmins2[i],vmax = vmaxs2[i], cmap = cmaps[i], ax = ax[i,1], add_colorbar=False)
                axins3 = inset_axes(ax[i,1],width="3%",height="100%", loc='lower left',bbox_to_anchor=(1.02, 0., 1, 1),bbox_transform=ax[i,1].transAxes,borderpad=0)


                plt.colorbar(b,axins3,ticks = ticks2[i])
                ax[i,1].coastlines()

                
                ax[i,1].set_extent([-110,10,0,70], crs=ccrs.PlateCarree())
                lowPreg2=lowPreg.isel(skip)
                lowPreg3=lowPreg2.sel(grid_yt=slice(10,90,1))
                b = lowPreg3.plot.quiver(x='grid_xt', y='grid_yt',u='ygrad_divide', v='xgrad_divide',ax = ax[i,1])

        #         axins = inset_axes(ax[i,1],width="3%",height="100%", loc='lower left',bbox_to_anchor=(1.02, 0., 1, 1),bbox_transform=ax[i,1].transAxes,borderpad=0)
        #         plt.colorbar(b,axins,ticks = ticks[i])
        #         ax[i,1].legend(title = theletters[1], loc = 'upper left',fontsize = 'medium')

                b=diffsdiffs.plot.contourf(levels = 40,vmin =-0.8,vmax = 0.8, cmap =cmaps[0], ax = ax[i,2], add_colorbar=False)
                ax[i,2].coastlines()
                axins = inset_axes(ax[i,2],width="3%",height="100%", loc='lower left',bbox_to_anchor=(1.02, 0., 1, 1),bbox_transform=ax[i,2].transAxes,borderpad=0)


                plt.colorbar(b,axins,ticks = ticks2[0], label = 'mbar')
                ax[i,2].set_extent([-110,10,0,70], crs=ccrs.PlateCarree())
                diffPreg2=diffPreg.isel(skip)
                diffPreg3=diffPreg2.sel(grid_yt=slice(10,90,1))
                b = diffPreg3.plot.quiver(x='grid_xt', y='grid_yt',u='ygrad_divide', v='xgrad_divide',ax = ax[i,2])



    #         ax[i,2].legend(title = theletters[2], loc = 'upper left',fontsize = 'medium')

            ax[i,0].set_yticks([0,15,30,45,60], crs = ccrs.PlateCarree())
            lat_formatter = LatitudeFormatter()
            ax[i,0].yaxis.set_major_formatter(lat_formatter)
            ax[i,0].set_ylabel(' ')

            lon_formatter = LongitudeFormatter()
 
            for c in range(3):
                ax[i,c].xaxis.set_major_formatter(lon_formatter)
                ax[i,c].set_xticks([ -90, -60,-30,0], crs = ccrs.PlateCarree())
                ax[i,c].set_xlabel(' ')
            ax[0,0].set_title('(A): Westward shift', fontsize = 17)
            ax[0,1].set_title('(B): Eastward shift', fontsize = 17)
            ax[0,2].set_title('(A) minus (B)', fontsize = 17)

        plt.show()
        fig.savefig('figures/SLPboots2.jpg')

