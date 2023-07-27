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
import seaborn as sns

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


from datetime import datetime
from datetime import timedelta


import predictions2


import SLP_boots
hibestenses,lowbestenses,ensemblesSLP = SLP_boots.load_data()

import regression4

import matplotlib.ticker as ticker

def fmt(x, pos):
    a, b = '{:.0e}'.format(x).split('e')
    b = int(b)
    if a ==0:
    
        return r'${}$'.format(a)
    elif b ==0: 
        return r'${}$'.format(b)
    else: 
        return r'${} \times 10^{{{}}}$'.format(a, b)


#     return Hs
def densitySLP_gen2(hibestdfs,genconsthiList,SLPconsthiList,predictedBootsnoWind, genc4x,SLPc4x,genc4x4x,SLPc4x4x,predictedBootsnoWind4x, norm = False, n_ens=1):
    
#         fig, ax = plt.subplots(ncols=1,nrows=1,figsize=(8,5), sharex=False,subplot_kw={'projection': ccrs.PlateCarree(central_longitude=0)}, dpi = 300)

        
        nrows = 3
        ncols = 3
        fig, ax = plt.subplots(ncols=ncols,nrows=nrows,figsize=(30,6*nrows), sharex=False,subplot_kw={'projection': ccrs.PlateCarree(central_longitude=0)}, dpi = 300)

        for c in range(nrows):
            for col in range(n_ens):
            
                count = c
                if c ==0: 

                    df2 = hibestdfs[col].copy().reset_index().drop('index',axis =1)
                    df22 = hibestdfs[col].copy().reset_index().drop('index',axis =1)
                    df222 = hibestdfs[col].copy().reset_index().drop('index',axis =1)
                elif c ==1:

                    df2 = genconsthiList[col].copy().reset_index().drop('index',axis =1)
                    df22 = SLPconsthiList[col].copy().reset_index().drop('index',axis =1)
                    df222 = predictedBootsnoWind[col].copy().reset_index().drop('index',axis =1)


                elif c ==2:
                    df2 = genc4x[col].copy()
                    df22 = SLPc4x[col].copy()
                    df222 = predictedBootsnoWind4x[col].copy().reset_index().drop('index',axis =1)
                try:    
                    df222['lon']= df222['currentlon'].values
                    df222['lat']= df222['currentlat'].values  
                except KeyError: 
                    print(" ")
                if n_ens>1: 
                    Ha, Hb, X,Y= getDiffGrid(df22,norm, diff = False)
                    Ha2, Hb2, X2,Y2= getDiffGrid(df2,norm, diff = False)
                    Ha3, Hb3, X3,Y3= getDiffGrid(df222,norm, diff = False)
                    if col==0: 
                        HaSum = Ha
                        HbSum = Hb
                        HaSum2 = Ha2
                        HbSum2 = Hb2
                        HaSum3 = Ha3
                        HbSum3 = Hb3
                    else: 
                        HaSum += Ha
                        HbSum += Hb
                        HaSum2 += Ha2
                        HbSum2 += Hb2
                        HaSum3 += Ha3
                        HbSum3 += Hb3
                        
                else:    
                    HD,HD_rolling,X,Y= getDiffGrid(df22,norm)
                    HD2,HD_rolling2,X2,Y2= getDiffGrid(df2,norm)
                    HD3,HD_rolling3,X3,Y3= getDiffGrid(df222,norm)
            if n_ens>1:
                HD2 = (HaSum2-HbSum2)/n_ens
                HD = (HaSum-HbSum)/n_ens
                HD3 = (HaSum3-HbSum3)/n_ens
            if norm==True: 
                max2=max1=0.0015

            if norm ==True:
                max3=0.0008

#             else: max3=5
#             if c ==3: max3=0.0001




            if c==0:
                if n_ens ==10: 
                    thetitle ='Average of All Ensemble Members'
                else:
                    thetitle='Target: Eastward shift'
            elif c==1:
                thetitle = 'Monthly model'
            elif c==2:
                thetitle = '6-hourly model'
                
            elif c==3:
                thetitle = '6-hourly model + predictions'
            thetitles2=[' changing genesis',' changing SLP', ' changing genesis & SLP']
            for m in range(3):
                if m ==1: 
                    if c !=0: 
                        max3= 0.0004
                    else: max3=0.0004
                else: 
                    max3=0.0004
                if m ==1:
                    HD =HD2

                elif m ==2: 
                    HD = HD3

#                 if m !=2: 
                imc = ax[c,m].pcolormesh(X, Y,np.ma.masked_where(HD==0,HD), cmap='seismic',vmin = -max3, vmax = max3)
#                 elif m ==2: 
#                     imc = ax[c,m].pcolormesh(X, Y,np.ma.masked_where(HD==0,HD), cmap='seismic')
#                 if c!=0:

#                     ax[c,m].legend(title = thetitles2[m], loc ='upper left')
                if c ==0: 
                    ax[c,m].set_title(thetitle)
                else:
                    ax[c,m].set_title(thetitle+ thetitles2[m])
                if (((c ==0)&(m==1))|((c!=0)&(m==2)))|((m==1)&(c==0)):
                    axinsc = inset_axes(ax[c,m],width="3%",height="100%", loc='lower left',
                                           bbox_to_anchor=(1.02, 0., 1, 1),
                                           bbox_transform=ax[c,m].transAxes,borderpad=0) 
                if c==0:
                    if m==1: 
                        thelabel = 'Prop. of points change'
                elif m ==2: 
                    thelabel = 'Prop. of points change'
                else: thelabel =''
                if (((c ==0)&(m==1))|((c!=0)&(m==2))):
                    cbar = fig.colorbar(imc, axinsc, ticks = [-0.0004,-0.0003,-0.0002,-0.0001,0,0.0001,0.0002,0.0003,0.0004], label =thelabel)
                    cbar.formatter.set_powerlimits((0, 0))
#                 if m==1: 
#                     if c !=0: 
#                         cbar = fig.colorbar(imc, axinsc, ticks = [-0.0004,-0.0003,-0.0002,-0.0001,0,0.0001,0.0002,0.0003,0.0004],label =thelabel)
#                         cbar.formatter.set_powerlimits((0, 0))
                ax[c,m].coastlines(color =  'steelblue')
#                 if (c==3)|(c==0): 

                if c ==2: 
                    lon_formatter = LongitudeFormatter()

                    ax[c,m].xaxis.set_major_formatter(lon_formatter)
                    ax[c,m].set_xticks([ -90, -60,-30,0], crs = ccrs.PlateCarree())

                ax[c,m].set_extent([-110,10,0,70], crs=ccrs.PlateCarree())

                ax[c,m].add_feature(cartopy.feature.LAND, facecolor= 'ivory',edgecolor='lightblue')
                ax[c,m].add_feature(cartopy.feature.LAKES, facecolor='aliceblue',edgecolor='lightblue')
                ax[c,m].add_feature(cartopy.feature.RIVERS, edgecolor ='aliceblue')
                if m==0:
                    if c!=0: 
                        ax[c,m].set_yticks([0,15,30,45,60], crs = ccrs.PlateCarree())
                ax[0,1].set_yticks([0,15,30,45,60], crs = ccrs.PlateCarree())
                lat_formatter = LatitudeFormatter()
                ax[c,m].yaxis.set_major_formatter(lat_formatter)
#                 if c==0:
#                     ax[c,m].set_yticks([0,15,30,45,60,70], crs = ccrs.PlateCarree())
#                     lat_formatter = LatitudeFormatter()
#                     ax[c,m].yaxis.set_major_formatter(lat_formatter)
                ax[0,0].set_visible(False)
                ax[0,2].set_visible(False)
        plt.subplots_adjust(hspace = 0.09, wspace=0.07)
        fig.savefig('/tigress/gkortum/thesisSummer/figures/shiftRsExamplesEXPERIMENT_wide2.jpg', bbox_inches='tight' )
#     

#     return H
def densitySLP_gen3(hibestdfs,genconsthiList,SLPconsthiList,predictedBootsnoWind, genc4x,SLPc4x,genc4x4x,SLPc4x4x,predictedBootsnoWind4x, norm = False, n_ens=1):
    
#         fig, ax = plt.subplots(ncols=1,nrows=1,figsize=(8,5), sharex=False,subplot_kw={'projection': ccrs.PlateCarree(central_longitude=0)}, dpi = 300)

        
        nrows = 2
        ncols = 3
        fig, ax = plt.subplots(ncols=ncols,nrows=nrows,figsize=(30,6*nrows), sharex=False,subplot_kw={'projection': ccrs.PlateCarree(central_longitude=0)}, dpi = 300)

        for c in range(nrows):
            for col in range(n_ens):
            
                count = c
                if c ==0: 

                    df2 = hibestdfs[col].copy().reset_index().drop('index',axis =1)
                    df22 = hibestdfs[col].copy().reset_index().drop('index',axis =1)
                    df222 = hibestdfs[col].copy().reset_index().drop('index',axis =1)
                elif c ==1:

                    df2 = genconsthiList[col].copy().reset_index().drop('index',axis =1)
                    df22 = SLPconsthiList[col].copy().reset_index().drop('index',axis =1)
                    df222 = predictedBootsnoWind[col].copy().reset_index().drop('index',axis =1)


#                 elif c ==2:
#                     df2 = genc4x[col].copy()
#                     df22 = SLPc4x[col].copy()
#                     df222 = predictedBootsnoWind4x[col].copy().reset_index().drop('index',axis =1)
                try:    
                    df222['lon']= df222['currentlon'].values
                    df222['lat']= df222['currentlat'].values  
                except KeyError: 
                    print(" ")
                if n_ens>1: 
                    Ha, Hb, X,Y= getDiffGrid(df22,norm, diff = False)
                    Ha2, Hb2, X2,Y2= getDiffGrid(df2,norm, diff = False)
                    Ha3, Hb3, X3,Y3= getDiffGrid(df222,norm, diff = False)
                    if col==0: 
                        HaSum = Ha
                        HbSum = Hb
                        HaSum2 = Ha2
                        HbSum2 = Hb2
                        HaSum3 = Ha3
                        HbSum3 = Hb3
                    else: 
                        HaSum += Ha
                        HbSum += Hb
                        HaSum2 += Ha2
                        HbSum2 += Hb2
                        HaSum3 += Ha3
                        HbSum3 += Hb3
                        
                else:    
                    HD,HD_rolling,X,Y= getDiffGrid(df22,norm)
                    HD2,HD_rolling2,X2,Y2= getDiffGrid(df2,norm)
                    HD3,HD_rolling3,X3,Y3= getDiffGrid(df222,norm)
            if n_ens>1:
                HD2 = (HaSum2-HbSum2)/n_ens
                HD = (HaSum-HbSum)/n_ens
                HD3 = (HaSum3-HbSum3)/n_ens
            if norm==True: 
                max2=max1=0.0015

            if norm ==True:
                max3=0.0008

#             else: max3=5
#             if c ==3: max3=0.0001




            if c==0:
                if n_ens ==10: 
                    thetitle ='Average of All Ensemble Members'
                else:
                    thetitle='Target: Eastward shift'
            elif c==1:
                thetitle = 'Monthly model'
#             elif c==2:
#                 thetitle = '6-hourly model'
                
#             elif c==3:
#                 thetitle = '6-hourly model + predictions'
            thetitles2=[' changing genesis',' changing SLP', ' changing genesis & SLP']
            for m in range(3):
                if m ==1: 
                    if c !=0: 
                        max3= 0.0004
                    else: max3=0.0004
                else: 
                    max3=0.0004
                if m ==1:
                    HD =HD2

                elif m ==2: 
                    HD = HD3

#                 if m !=2: 
                imc = ax[c,m].pcolormesh(X, Y,np.ma.masked_where(HD==0,HD), cmap='seismic',vmin = -max3, vmax = max3)
#                 elif m ==2: 
#                     imc = ax[c,m].pcolormesh(X, Y,np.ma.masked_where(HD==0,HD), cmap='seismic')
#                 if c!=0:

#                     ax[c,m].legend(title = thetitles2[m], loc ='upper left')
                if c ==0: 
                    ax[c,m].set_title(thetitle)
                else:
                    ax[c,m].set_title(thetitle+ thetitles2[m])
                if (((c ==0)&(m==1))|((c!=0)&(m==2)))|((m==1)&(c==0)):
                    axinsc = inset_axes(ax[c,m],width="3%",height="100%", loc='lower left',
                                           bbox_to_anchor=(1.02, 0., 1, 1),
                                           bbox_transform=ax[c,m].transAxes,borderpad=0) 
                if c==0:
                    if m==1: 
                        thelabel = 'Prop. of points change'
                elif m ==2: 
                    thelabel = 'Prop. of points change'
                else: thelabel =''
                if (((c ==0)&(m==1))|((c!=0)&(m==2))):
                    cbar = fig.colorbar(imc, axinsc, ticks = [-0.0004,-0.0003,-0.0002,-0.0001,0,0.0001,0.0002,0.0003,0.0004], label =thelabel)
                    cbar.formatter.set_powerlimits((0, 0))
#                 if m==1: 
#                     if c !=0: 
#                         cbar = fig.colorbar(imc, axinsc, ticks = [-0.0004,-0.0003,-0.0002,-0.0001,0,0.0001,0.0002,0.0003,0.0004],label =thelabel)
#                         cbar.formatter.set_powerlimits((0, 0))
                ax[c,m].coastlines(color =  'steelblue')
#                 if (c==3)|(c==0): 

                if c ==1: 
                    lon_formatter = LongitudeFormatter()

                    ax[c,m].xaxis.set_major_formatter(lon_formatter)
                    ax[c,m].set_xticks([ -90, -60,-30,0], crs = ccrs.PlateCarree())

                ax[c,m].set_extent([-110,10,0,70], crs=ccrs.PlateCarree())

                ax[c,m].add_feature(cartopy.feature.LAND, facecolor= 'ivory',edgecolor='lightblue')
                ax[c,m].add_feature(cartopy.feature.LAKES, facecolor='aliceblue',edgecolor='lightblue')
                ax[c,m].add_feature(cartopy.feature.RIVERS, edgecolor ='aliceblue')
                if m==0:
                    if c!=0: 
                        ax[c,m].set_yticks([0,15,30,45,60], crs = ccrs.PlateCarree())
                ax[0,1].set_yticks([0,15,30,45,60], crs = ccrs.PlateCarree())
                lat_formatter = LatitudeFormatter()
                ax[c,m].yaxis.set_major_formatter(lat_formatter)
#                 if c==0:
#                     ax[c,m].set_yticks([0,15,30,45,60,70], crs = ccrs.PlateCarree())
#                     lat_formatter = LatitudeFormatter()
#                     ax[c,m].yaxis.set_major_formatter(lat_formatter)
                ax[0,0].set_visible(False)
                ax[0,2].set_visible(False)
        plt.subplots_adjust(hspace = 0.09, wspace=0.07)
        fig.savefig('/tigress/gkortum/thesisSummer/figures/shiftRsExamplesEXPERIMENT_wide2.jpg', bbox_inches='tight' )

def densitySLP_gen4(hibestdfs,genconsthiList,SLPconsthiList,predictedBootsnoWind, genc4x,SLPc4x,genc4x4x,SLPc4x4x,predictedBootsnoWind4x, norm = False, n_ens=1):
    
#         fig, ax = plt.subplots(ncols=1,nrows=1,figsize=(8,5), sharex=False,subplot_kw={'projection': ccrs.PlateCarree(central_longitude=0)}, dpi = 300)

        
        nrows = 2
        ncols = 3
        fig, ax = plt.subplots(ncols=ncols,nrows=nrows,figsize=(30,6*nrows), sharex=False,subplot_kw={'projection': ccrs.PlateCarree(central_longitude=0)}, dpi = 300)

        for c in range(nrows):
            for col in range(n_ens):
            
                count = c
                if c ==0: 

                    df2 = hibestdfs[col].copy().reset_index().drop('index',axis =1)
                    df22 = hibestdfs[col].copy().reset_index().drop('index',axis =1)
                    df222 = hibestdfs[col].copy().reset_index().drop('index',axis =1)
#                 elif c ==1:

#                     df2 = genconsthiList[col].copy().reset_index().drop('index',axis =1)
#                     df22 = SLPconsthiList[col].copy().reset_index().drop('index',axis =1)
#                     df222 = predictedBootsnoWind[col].copy().reset_index().drop('index',axis =1)


                elif c ==1:
                    df2 = genc4x[col].copy()
                    df22 = SLPc4x[col].copy()
                    df222 = predictedBootsnoWind4x[col].copy().reset_index().drop('index',axis =1)
                try:    
                    df222['lon']= df222['currentlon'].values
                    df222['lat']= df222['currentlat'].values  
                except KeyError: 
                    print(" ")
                if n_ens>1: 
                    Ha, Hb, X,Y= getDiffGrid(df22,norm, diff = False)
                    Ha2, Hb2, X2,Y2= getDiffGrid(df2,norm, diff = False)
                    Ha3, Hb3, X3,Y3= getDiffGrid(df222,norm, diff = False)
                    if col==0: 
                        HaSum = Ha
                        HbSum = Hb
                        HaSum2 = Ha2
                        HbSum2 = Hb2
                        HaSum3 = Ha3
                        HbSum3 = Hb3
                    else: 
                        HaSum += Ha
                        HbSum += Hb
                        HaSum2 += Ha2
                        HbSum2 += Hb2
                        HaSum3 += Ha3
                        HbSum3 += Hb3
                        
                else:    
                    HD,HD_rolling,X,Y= getDiffGrid(df22,norm)
                    HD2,HD_rolling2,X2,Y2= getDiffGrid(df2,norm)
                    HD3,HD_rolling3,X3,Y3= getDiffGrid(df222,norm)
            if n_ens>1:
                HD2 = (HaSum2-HbSum2)/n_ens
                HD = (HaSum-HbSum)/n_ens
                HD3 = (HaSum3-HbSum3)/n_ens
            if norm==True: 
                max2=max1=0.0015

            if norm ==True:
                max3=0.0008

#             else: max3=5
#             if c ==3: max3=0.0001




            if c==0:
                if n_ens ==10: 
                    thetitle ='Average of All Ensemble Members'
                else:
                    thetitle='Target: Eastward shift'
#             elif c==1:
#                 thetitle = 'Monthly model'
            elif c==1:
                thetitle = '6-hourly model'
                
#             elif c==3:
#                 thetitle = '6-hourly model + predictions'
            thetitles2=[' changing genesis',' changing SLP', ' changing genesis & SLP']
            for m in range(3):
                if m ==1: 
                    if c !=0: 
                        max3= 0.0004
                    else: max3=0.0004
                else: 
                    max3=0.0004
                if m ==1:
                    HD =HD2

                elif m ==2: 
                    HD = HD3

#                 if m !=2: 
                imc = ax[c,m].pcolormesh(X, Y,np.ma.masked_where(HD==0,HD), cmap='seismic',vmin = -max3, vmax = max3)
#                 elif m ==2: 
#                     imc = ax[c,m].pcolormesh(X, Y,np.ma.masked_where(HD==0,HD), cmap='seismic')
#                 if c!=0:

#                     ax[c,m].legend(title = thetitles2[m], loc ='upper left')
                if c ==0: 
                    ax[c,m].set_title(thetitle)
                else:
                    ax[c,m].set_title(thetitle+ thetitles2[m])
                if (((c ==0)&(m==1))|((c!=0)&(m==2)))|((m==1)&(c==0)):
                    axinsc = inset_axes(ax[c,m],width="3%",height="100%", loc='lower left',
                                           bbox_to_anchor=(1.02, 0., 1, 1),
                                           bbox_transform=ax[c,m].transAxes,borderpad=0) 
                if c==0:
                    if m==1: 
                        thelabel = 'Prop. of points change'
                elif m ==2: 
                    thelabel = 'Prop. of points change'
                else: thelabel =''
                if (((c ==0)&(m==1))|((c!=0)&(m==2))):
                    cbar = fig.colorbar(imc, axinsc, ticks = [-0.0004,-0.0003,-0.0002,-0.0001,0,0.0001,0.0002,0.0003,0.0004], label =thelabel)
                    cbar.formatter.set_powerlimits((0, 0))
#                 if m==1: 
#                     if c !=0: 
#                         cbar = fig.colorbar(imc, axinsc, ticks = [-0.0004,-0.0003,-0.0002,-0.0001,0,0.0001,0.0002,0.0003,0.0004],label =thelabel)
#                         cbar.formatter.set_powerlimits((0, 0))
                ax[c,m].coastlines(color =  'steelblue')
#                 if (c==3)|(c==0): 

                if c ==1: 
                    lon_formatter = LongitudeFormatter()

                    ax[c,m].xaxis.set_major_formatter(lon_formatter)
                    ax[c,m].set_xticks([ -90, -60,-30,0], crs = ccrs.PlateCarree())

                ax[c,m].set_extent([-110,10,0,70], crs=ccrs.PlateCarree())

                ax[c,m].add_feature(cartopy.feature.LAND, facecolor= 'ivory',edgecolor='lightblue')
                ax[c,m].add_feature(cartopy.feature.LAKES, facecolor='aliceblue',edgecolor='lightblue')
                ax[c,m].add_feature(cartopy.feature.RIVERS, edgecolor ='aliceblue')
                if m==0:
                    if c!=0: 
                        ax[c,m].set_yticks([0,15,30,45,60], crs = ccrs.PlateCarree())
                ax[0,1].set_yticks([0,15,30,45,60], crs = ccrs.PlateCarree())
                lat_formatter = LatitudeFormatter()
                ax[c,m].yaxis.set_major_formatter(lat_formatter)
#                 if c==0:
#                     ax[c,m].set_yticks([0,15,30,45,60,70], crs = ccrs.PlateCarree())
#                     lat_formatter = LatitudeFormatter()
#                     ax[c,m].yaxis.set_major_formatter(lat_formatter)
                ax[0,0].set_visible(False)
                ax[0,2].set_visible(False)
        plt.subplots_adjust(hspace = 0.09, wspace=0.07)
        fig.savefig('/tigress/gkortum/thesisSummer/figures/shiftRsExamplesEXPERIMENT_wide3.jpg', bbox_inches='tight' )

def getDiffGrid(df2,norm, diff=True):
         #     count = 0
        above=[]
        for i in range(df2.shape[0]):

            if df2['year'].values[i]< 1995: 
                above.append(0)
            else: above.append(1)

        df2['above']= above   
#         df2=df.copy()

    
        dfabove = df2[df2['above']==1]
        dfbelow = df2[df2['above']==0]

        xedges = range(240,360,5)
        yedges = range(0,90,5)
        
        xa = dfabove['lon']
        ya = dfabove['lat']
        xb = dfbelow['lon']
        yb = dfbelow['lat']
        x = df2['lon'].values
        y = df2['lat'].values
  

        H, xedges, yedges = np.histogram2d(x, y, bins=(xedges, yedges), normed = norm)
        Ha, xedges, yedges = np.histogram2d(xa, ya, bins=(xedges, yedges), normed = norm)
        Hb, xedges, yedges = np.histogram2d(xb, yb, bins=(xedges, yedges), normed = norm)

   
        Ha = Ha.T
        Hb = Hb.T
        H = H.T
        X, Y= np.meshgrid(xedges, yedges)

#         imH = ax[count,0].pcolormesh(X, Y, H, vmin = 0, vmax = max2)
#         im = ax[count,2].pcolormesh(X, Y, Ha, vmin = 0, vmax = max1) 
#         imb = ax[count,1].pcolormesh(X, Y, Hb, vmin = 0, vmax = max1) 
        HD=Ha-Hb
        HD_rolling=np.array(pd.DataFrame(HD).rolling(window = 2, axis = 1).mean().rolling(window = 2,axis =0).mean())
        if diff: 
            return HD,HD_rolling,X,Y
        else: return Ha, Hb, X, Y
def get_HS_for_errors(nboots,hibestdfs,predictedBoots,predictedBootsnoWind,norm = True):
    nrows = 3
    Hsdf=[]
    HsBoot=[]
    HsBootnoWind=[]
    for col in range(nboots):
        for c in range(nrows):
            count = c
            if c ==0: 
                df2 = hibestdfs[col].copy()
                HD,HD_rolling,X,Y= getDiffGrid(df2,norm)
                Hsdf.append(HD)
            elif c ==1:
               
                predictedboot = predictedBoots[col][['currentlat','currentlon','year']]
                predictedboot.columns=['lat','lon','year']
                df2=predictedboot.copy()
                HD,HD_rolling,X,Y= getDiffGrid(df2,norm)
                HsBoot.append(HD)
            elif c ==2:
                predictedbootnoWind = predictedBootsnoWind[col][['currentlat','currentlon','year']]
                predictedbootnoWind.columns=['lat','lon','year']
                df2=predictedbootnoWind.copy()
                HD,HD_rolling,X,Y= getDiffGrid(df2,norm)
                HsBootnoWind.append(HD)

    return Hsdf,HsBoot,HsBootnoWind

def get_Hs_comps(nboots,hibestdfs,predictedBoots,predictedBootsnoWind,norm=True):
    
    Hsdf,HsBoot,HsBootnoWind =  get_HS_for_errors(nboots,hibestdfs,predictedBoots,predictedBootsnoWind,norm)
    rrmsBoot=[]
    rrmsBootnoWind=[]
    for b in range(nboots):
        H00=Hsdf[b].copy()
        for w in range(2):
            if w ==0: 
                Hmm= HsBoot[b].copy()
            elif w==1:
                Hmm =HsBootnoWind[b].copy()

#             H0=np.array(H0).flatten()

#             H1=np.array(Hm).flatten()


#             H2= np.invert(np.isnan(H1))

#             H0=H0[H2]
#             H1=H1[H2]
            
#             H3=np.invert(np.isnan(H0))
            
#             H1=H1[H3]
#             H0=H0[H3]
            
#             coef=sm.OLS(H0,H1).fit(cov_type='HC0').params[0]
#             r2=sm.OLS(H0,H1).fit(cov_type='HC0').rsquared

#             r = np.sqrt(r2)*np.sign(coef)

            H0=np.array(pd.DataFrame(H00).rolling(window = 2).mean().rolling(window = 2,axis =1).mean())
            Hm=np.array(pd.DataFrame(Hmm).rolling(window = 2).mean().rolling(window = 2,axis =1).mean())

            H0=H0.flatten()
            H1=Hm.flatten()
        
            H2= np.invert(np.isnan(H1))

            H0=H0[H2]
            H1=H1[H2]
            
            H3=np.invert(np.isnan(H0))
            
            H1=H1[H3]
            H0=H0[H3]
            
            coefrm=sm.OLS(H0,H1).fit(cov_type='HC0').params[0]
            r2rm=sm.OLS(H0,H1).fit(cov_type='HC0').rsquared
            rrm = np.sqrt(r2rm)*np.sign(coefrm)
            if w ==0: 
                rrmsBoot.append(rrm)
            elif w==1:
                rrmsBootnoWind.append(rrm)

    return rrmsBoot,rrmsBootnoWind


   

def Hs_errs_hists2(thelists,theArgs,hibestdfs,lowbestdfs,nboots):
    ncols = 1
    nrows=1
    fig, ax = plt.subplots(ncols=ncols,nrows=nrows,figsize=(8,3.5), sharex=False, dpi = 300)
    theTitles1 = [' Monthly model',' 6-Hourly model',' 6-Hourly model and predictions']
    model = 0
    PB=theArgs[0+model*4]
    PBL=theArgs[1+model*4]
    PBNW=theArgs[2+model*4]
    PBLNW=theArgs[3+model*4]
    rrms, rrmsNW =get_Hs_comps(nboots,hibestdfs,PB,PBNW)
    rrmsL, rrmsLNW =get_Hs_comps(nboots,lowbestdfs,PBL,PBLNW)
#         theList = [rrms,rrmsNW,rrmsL,rrmsLNW]
    theTitles = ['West shift ','East shift ']
    colors = ['midnightblue','midnightblue']
    r =1
    rrms1= np.append(rrms, rrmsL)
    rrmsN = np.append(rrmsNW, rrmsLNW)
#     rrmsN = rrmsNW
#             if r ==1:
#                 rrms=rrmsL
#                 rrmsNW = rrmsLNW

#             sns.kdeplot(rrms1, ax = ax[model],label='mean\n = '+str(round(np.mean(np.array(rrms)),3)),color = colors[r])






    thelist = thelists[model]
    rrmsGenConst= thelist[0]
    rrmsGenConstL= thelist[1]
    rrmsSLPConst= thelist[2]
    rrmsSLPConstL= thelist[3]
    rrmsG = np.append(rrmsGenConst,rrmsGenConstL)
    rrmsS = np.append(rrmsSLPConst,rrmsSLPConstL)
    rrmsG = np.append(rrmsGenConst,rrmsGenConstL)
    rrmsS = np.append(rrmsSLPConst,rrmsSLPConstL)
    theTitles = ['West shift ','East shift ']





    r = 0
#             if r ==1:
#                 rrmsGenConst=rrmsGenConstL
#                 rrmsSLPConst = rrmsSLPConstL
#                 sns.kdeplot(rrmsGenConst,ax=ax[model], bw_adjust =3, label='mean \n= '+str(round(np.mean(np.array(rrmsGenConst)),4)), c='orchid', linestyle = '--', linewidth = 1.5)
#                 sns.kdeplot(rrmsSLPConst, ax =ax[model],bw_adjust =3,label='mean \n= '+str(round(np.mean(np.array(rrmsSLPConst)),4)), c = 'seagreen', linestyle = '--', linewidth = 1.5)
#             else: 
    sns.kdeplot(rrmsG,ax=ax, bw_adjust =2, label='changing steering flow', color='seagreen', linewidth = 2, fill = True, alpha = 0.1)
    sns.kdeplot(rrmsS, ax =ax,bw_adjust =3,label='changing genesis location', color = 'tomato', linewidth = 2, fill = True, alpha = 0.15)

    sns.kdeplot(rrmsN, ax =ax,bw_adjust =1.5,label='changing genesis and steering flow',  color = colors[r], linewidth = 2, fill = True, alpha = 0.2)


    ax.legend(loc = 'upper left')
    ax.set_title(str(theTitles1[model]))
    ax.set_xlim(-0.5,1)
    if model ==0: 
        ax.set_ylim(0,5)
    ax.plot(np.zeros(2),[0,4.5], linestyle = '--', color = 'grey' )
    ax.set_xlabel('Correlation')
            
#         handles, labels = ax.get_legend_handles_labels()
#         fig.legend(handles, ['West shift \nconstant genesis', 'West shift \nconstant SLP','East shift \nconstant genesis', 'East shift \nconstant SLP'], bbox_to_anchor=(1.0, 0.9), loc='upper left')
#         fig.tight_layout()
    fig.savefig('/tigress/gkortum/figures/exphist.png',bbox_inches='tight',  dpi = 600)


class LegendTitle(object):
    def __init__(self, text_props=None):
        self.text_props = text_props or {}
        super(LegendTitle, self).__init__()

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        title = mtext.Text(x0, y0, r'\underline{' + orig_handle + '}', usetex=True, **self.text_props)
        handlebox.add_artist(title)
        return title
def plot_Hs_gens_hists(rsm1, rsm2,theArgs,hibestdfs,lowbestdfs):
    ncols = 1
    nrows=2
    nboots = 100
    fig, ax = plt.subplots(ncols=ncols,nrows=nrows,figsize=(15,15), dpi = 300, sharex=True)
    theTitles1 = [' Monthly model',' 6-Hourly model']
    rsm = rsm1
#     rrms = rrms1

    for model in range(2):
        theTitles = ['West shift ','East shift ']
        if model ==1: 
            rsm = rsm2
#             rrms = rrms2
            
        rrmsGenConst= rsm[0][0]
        rrmsGenConstL= rsm[1][0]
        rrmsSLPConst= rsm[2][0]
        rrmsSLPConstL= rsm[3][0]
        
        PB=theArgs[0+model*4]
        PBL=theArgs[1+model*4]
        PBNW=theArgs[2+model*4]
        PBLNW=theArgs[3+model*4]
        rrms, rrmsNW =get_Hs_comps(nboots,hibestdfs,PB,PBNW)
        rrmsL, rrmsLNW =get_Hs_comps(nboots,lowbestdfs,PBL,PBLNW)
        


        line1= sns.kdeplot(rrmsGenConst,ax=ax[model], bw_adjust =2, label=theTitles[0]+'changing SLP \nmean R = '+str(round(np.nanmean(np.array(rrmsGenConst)),4)), c='seagreen', linestyle = '--', linewidth = 2)

        line2=sns.kdeplot(rrmsGenConstL,ax=ax[model], bw_adjust =2, label=theTitles[1]+'changing SLP \nmean R = '+str(round(np.nanmean(np.array(rrmsGenConstL)),4)), c='seagreen', linewidth = 2)
        line3=sns.kdeplot(rrmsSLPConst, ax =ax[model],bw_adjust =2,label=theTitles[0]+'changing genesis \nmean R = '+str(round(np.nanmean(np.array(rrmsSLPConst)),4)), c = 'tomato', linestyle = '--', linewidth = 2)
        line4=sns.kdeplot(rrmsSLPConstL, ax =ax[model],bw_adjust =2,label=theTitles[1]+'changing genesis \nmean R = '+str(round(np.nanmean(np.array(rrmsSLPConstL)),4)), c = 'tomato', linewidth = 2)
        line5=sns.kdeplot(rrmsNW, ax =ax[model],bw_adjust =2,label=theTitles[0]+'changing SLP and genesis \nmean R = '+str(round(np.nanmean(np.array(rrmsNW)),4)), c = 'midnightblue', linestyle = '--', linewidth = 2)

    
        line6=sns.kdeplot(rrmsLNW, ax =ax[model],bw_adjust =2,label=theTitles[1]+'changing SLP and genesis \nmean R = '+str(round(np.nanmean(np.array(rrmsLNW)),4)), c = 'midnightblue',  linewidth = 2)


        ax[model].legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax[model].set_title(str(theTitles1[model]))
        ax[model].set_xlim(-0.5,1)
        ax[model].set_ylim(0,5)
        ax[model].plot(np.zeros(2),[0,5], linestyle = '--', color = 'grey' )
        ax[model].set_xlabel('Correlation')
            
    fig.savefig('figures/exphist.png', dpi = 300)

def Hs_gens_hists(theArgs,hibestdfs,lowbestdfs,nboots):

    rsList = []
    rrmsGenConst1=[]
    rrmsGenConstL1=[]
    rrmsSLPConst1=[]
    rrmsSLPConstL1=[]
    
    rrmsGenConst2=[]
    rrmsGenConstL2=[]
    rrmsSLPConst2=[]
    rrmsSLPConstL2=[]
    
    rrmsGenConst3=[]
    rrmsGenConstL3=[]
    rrmsSLPConst3=[]
    rrmsSLPConstL3=[]
    
    for model in range(2):

        SLPH=theArgs[0+model*4]
        SLPL=theArgs[1+model*4]
        GH=theArgs[2+model*4]
        GL=theArgs[3+model*4]
#         if model == 0: 
#             nboots =1
#         elif model == 1:
#             nboots = 1
#         elif model == 2: 
#             nboots = 1
        

        rrmsGenConst,rrmsSLPConst =get_Hs_comps_for_exp(nboots,hibestdfs,GH,SLPH,norm=True)
        if nboots==100:
            
            rrmsGenConstL,rrmsSLPConstL =get_Hs_comps_for_exp(nboots,lowbestdfs,GL,SLPL,norm=True)
        else: 
            rrmsGenConstL=1
            rrmsSLPConstL =1
        if model ==0:
            rrmsGenConst1.append(rrmsGenConst)
            rrmsGenConstL1.append(rrmsGenConstL)
            rrmsSLPConst1.append(rrmsSLPConst)
            rrmsSLPConstL1.append(rrmsSLPConstL)
        elif model ==1: 
            rrmsGenConst2.append(rrmsGenConst)
            rrmsGenConstL2.append(rrmsGenConstL)
            rrmsSLPConst2.append(rrmsSLPConst)
            rrmsSLPConstL2.append(rrmsSLPConstL)
#         elif model ==2: 
#             rrmsGenConst3.append(rrmsGenConst)
#             rrmsGenConstL3.append(rrmsGenConstL)
#             rrmsSLPConst3.append(rrmsSLPConst)
#             rrmsSLPConstL3.append(rrmsSLPConstL)

        #         theList = [rrms,rrmsNW,rrmsL,rrmsLNW]
 
    return  [rrmsGenConst1, rrmsGenConstL1,rrmsSLPConst1,rrmsSLPConstL1], [rrmsGenConst2, rrmsGenConstL2,rrmsSLPConst2,rrmsSLPConstL2]
def get_HDs(df2,norm):
    HDs=[]
    for ens in range(10):
    
        for year in range(1970,2022,1):
            df3 = df2[(df2['setens']==ens) & (df2['setyr']==year)]
            HD,HD_rolling,X,Y= getDiffGrid(df3,norm)
            HDs.append(HD) 
    return HDs
def get_HS_for_exp(nboots,hibestdfs,genconsthiList,SLPconsthiList,  norm = True):
    nrows = 3
    Hsdf=[]
    HsGenConst=[]
    HsSLPConst=[]

                
    for col in range(nboots):
        for c in range(nrows):
            count = c
            if c ==0: 
                df2 = hibestdfs[col].copy()
                HD,HD_rolling,X,Y= getDiffGrid(df2,norm)
                Hsdf.append(HD)
            elif c ==1: 
                df2 = genconsthiList[col].copy()
                HDs = get_HDs(df2,norm)
                HsGenConst.append(HDs)
            elif c ==2:
                df2 = SLPconsthiList[col].copy()
                HDs = get_HDs(df2,norm)
                HsSLPConst.append(HDs)

     
    return Hsdf,HsGenConst,HsSLPConst

def get_Hs_comps_for_exp(nboots,hibestdfs,genconsthiList,SLPconsthiList,norm=True):
    
    Hsdf,HsGenConst,HsSLPConst =  get_HS_for_exp(nboots,hibestdfs,genconsthiList,SLPconsthiList,  norm = True)
    rrmsGenConst=[]
    rrmsSLPConst=[]
    
    
    for b in range(nboots):
        print(b)
        H00=Hsdf[b].copy()
        for c in range((2022-1970)*10):

            for w in range(2):
                if w ==0: 
                    try: 
                        
                        Hmm= HsGenConst[b][c].copy()
                    except IndexError: print(str(b)+" "+str(c))
                elif w==1:
                    Hmm =HsSLPConst[b][c].copy()



                H0=np.array(pd.DataFrame(H00).rolling(window = 2).mean().rolling(window = 2,axis =1).mean())
                Hm=np.array(pd.DataFrame(Hmm).rolling(window = 2).mean().rolling(window = 2,axis =1).mean())

                H0=H0.flatten()
                H1=Hm.flatten()

                H2= np.invert(np.isnan(H1))

                H0=H0[H2]
                H1=H1[H2]

                H3=np.invert(np.isnan(H0))

                H1=H1[H3]
                H0=H0[H3]
  
                
                rrm = np.corrcoef(H1, H0)[0,1]
#                 coefrm=sm.OLS(H0,H1).fit(cov_type='HC0').params[0]
#                 r2rm=sm.OLS(H0,H1).fit(cov_type='HC0').rsquared
#                 rrm = np.sqrt(r2rm)*np.sign(coefrm)
                if w ==0: 
                    rrmsGenConst.append(rrm)
                elif w==1:
                    rrmsSLPConst.append(rrm)

    return rrmsGenConst,rrmsSLPConst


def Hs_errs_hists(theArgs,hibestdfs,lowbestdfs,nboots):
    ncols = 1
    nrows=2
    fig, ax = plt.subplots(ncols=ncols,nrows=nrows,figsize=(10,10), sharex=True, dpi = 300)
    theTitles1 = [' Monthly model',' 6-Hourly model',' 6-Hourly model and predictions']
    for model in range(2):
#         if model ==2: 
#             nboots=20
#         if model ==1: 
#             continue
            
        PB=theArgs[0+model*4]
        PBL=theArgs[1+model*4]
        PBNW=theArgs[2+model*4]
        PBLNW=theArgs[3+model*4]
        rrms, rrmsNW =get_Hs_comps(nboots,hibestdfs,PB,PBNW)
        rrmsL, rrmsLNW =get_Hs_comps(nboots,lowbestdfs,PBL,PBLNW)
#         theList = [rrms,rrmsNW,rrmsL,rrmsLNW]
        theTitles = ['West shift ','East shift ']
        linestyles= ['--','-']

        for r in range(2):
            if r ==1:
                rrms=rrmsL
                rrmsNW = rrmsLNW
            
            sns.kdeplot(rrms, ax = ax[model],label='mean\n = '+str(round(np.mean(np.array(rrms)),3)),color = 'tomato',linestyle = linestyles[r], bw_adjust = 1.5)
            sns.kdeplot(rrmsNW, ax =ax[model],label='mean\n = '+str(round(np.mean(np.array(rrmsNW)),3)), color = 'midnightblue',linestyle = linestyles[r], bw_adjust = 1.5)
            
     
            ax[model].legend(fontsize='small')
            ax[model].set_title(str(theTitles1[model]))
            ax[model].set_xlim(0,1)

            ax[model].set_ylim(0,5)
            ax[model].set_yticks([0,1,2,3,4,5])
            ax[model].set_xlabel('Correlation')
        handles, labels = ax[model].get_legend_handles_labels()
        fig.legend(handles, ['West shift \nintensities given', 'West shift \nno intensities','East shift \nintensities given', 'East shift \nno intensities'], bbox_to_anchor=(1.0, 0.9), loc='upper left')
        fig.tight_layout()
    fig.savefig('/tigress/gkortum/thesisSummer/figures/shiftRs.jpg', bbox_inches='tight')

        