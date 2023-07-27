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
plt.rcParams.update({'font.size': 18})
def get_stormIDs(predicted): 
    stormID=[]

    theCount =0
    timesince =0
    count = 0

    #keep track of the separate storms in X
    previousI = 100
    previousYear = 1971
    for i in predicted['zSTORM1'].values:
        if i != previousI: 
            theCount+=1
        elif previousYear !=predicted['year'].values[count]: 
            theCount +=1
        stormID.append(theCount)
        previousI = i
        previousYear = predicted['year'].values[count]
        count+=1
#     for i in df5['XCOUNT'].values:
#         if i ==2: 
#             theCount+=1
#         stormID.append(theCount)
    Xmany = predicted.copy()
    Xmany['stormID']=stormID
#     Xmany['constant']=np.ones(Xmany.shape[0])
    #keep track of the length of each storm in X
    Xmany['constant']=np.ones(Xmany.shape[0])
    stormlengths = Xmany.groupby('stormID').count()['constant'].values
    stormLength=[]
    for i in stormID:
         stormLength.append(stormlengths[i-1])

    Xmany['length']=stormLength
    Xmany['constant']=np.ones(Xmany.shape[0])
    return Xmany

def deviation_wholetrack(predicted,  df5, X2, X1, modelLon,modelLat):

    Xmany = get_stormIDs(df5)
    Predmany = get_stormIDs(predicted)

    latpred = []
    lonpred=[]
    latreal2=[]
    lonreal2=[]
    latnaive=[]
    lonnaive=[]

    nstorms = Xmany['stormID'].values[-1]

    for thestorm in range(1,nstorms+1,1):

        Xdrop= Xmany[Xmany['stormID']==thestorm]
        length=Xdrop['length'].values[0]


        Xpred = Predmany[Predmany['stormID']==thestorm]

        lengthPred = Xpred['length'].values[0]
        
        thelengthmissing = 120 - length 
        toadd= np.ones(thelengthmissing)*np.nan
        thelengthmissingP = 120 - lengthPred 
        toaddP= np.ones(thelengthmissingP)*np.nan

        latpred.append(np.concatenate([Xpred['currentlat'].values,toaddP]))
        lonpred.append(np.concatenate([Xpred['currentlon'].values,toaddP]))

        if len(latpred[-1])!=120:
            print(thestorm)

        latreal2.append(np.concatenate([Xdrop['lat'].values,toadd]))
        lonreal2.append(np.concatenate([Xdrop['lon'].values,toadd]))
        latnaive.append(np.concatenate([Xdrop['lat'].values[0]*np.ones(length),toadd]))
        lonnaive.append(np.concatenate([Xdrop['lon'].values[0]*np.ones(length),toadd]))

    #convert to pandas
    latpred3=pd.DataFrame(np.asarray(latpred))
    lonpred3=pd.DataFrame(np.asarray(lonpred))
    latreal3=pd.DataFrame(np.asarray(latreal2))
    lonreal3=pd.DataFrame(np.asarray(lonreal2))
    
    latnaive3=pd.DataFrame(np.asarray(latnaive))
    lonnaive3=pd.DataFrame(np.asarray(lonnaive))




    return latpred3, lonpred3, latreal3, lonreal3, latnaive3,lonnaive3

def stdangle(angles):
    sin = 0
    cos = 0
    for i in range(0,len(angles),1):

        sin += math.sin(angles[i]);
        cos += math.cos(angles[i]); 

    sin =np.true_divide(sin, len(angles))
    cos =np.true_divide(cos, len(angles))

    stddev = math.sqrt(-math.log(sin*sin+cos*cos))

    return stddev


def get_stats(diffs4,thetas4):

    radii=[]
    thethetas = []
    thethetasSTD=[]
    for m in range(1,20,1):
        radius = np.nanmean(diffs4.iloc[:,m].values, axis =0)
        thetas = thetas4.iloc[:,m].values
        thetas = thetas[~np.isnan(thetas)]
        xs = 0
        ys = 0
        for i in thetas: 
            if not np.isnan(i):
                ys+=np.sin(i)
                xs+=np.cos(i)
        meantheta = float(np.arctan2(ys,xs))
    #     thetastd = thetas4.iloc[:,m].values.std()

        thethetas.append(meantheta)
        thethetasSTD.append(stdangle(thetas))
        radii.append(radius)
    return radii, thethetas, thethetasSTD



def plot_stds(diffs4,thetas4,color,ax): 
    radii, thethetas, thethetasSTD=get_stats(diffs4,thetas4)
    for m in range(19):

     
        if m ==0: 
            rad_prev = 0
        else: 
            rad_prev = radii[m-1]

        ax.fill_between(
            np.linspace(thethetas[m]-thethetasSTD[m],thethetas[m]+thethetasSTD[m], 100),  # Need high res or you'll fill a triangle
            rad_prev,
            radii[m],
            alpha=0.2,
            color=color,linewidth = 0, zorder = -1
        )


    
def plot_errors(predicted, df5Model, X2, X1, modelLon,modelLat,predicted4x, df5Model_4x, X2_4x, X1_4x, modelLon_4x,modelLat_4x,predictedL,predicted4xL):

    latpred3, lonpred3, latreal3, lonreal3, latnaive3,lonnaive3= deviation_wholetrack(predicted, df5Model, X2, X1, modelLon,modelLat)
    latpred34x, lonpred34x, latreal3, lonreal3, latnaive3,lonnaive3= deviation_wholetrack(predicted4x, df5Model_4x, X2_4x, X1_4x, modelLon_4x,modelLat_4x)

    latpred3L, lonpred3L, latreal3, lonreal3, latnaive3,lonnaive3= deviation_wholetrack(predictedL, df5Model, X2, X1, modelLon,modelLat)
    latpred34xL, lonpred34xL, latreal3, lonreal3, latnaive3,lonnaive3= deviation_wholetrack(predicted4xL, df5Model_4x, X2_4x, X1_4x, modelLon_4x,modelLat_4x)


    # lon4x3=np.array(lon4x3)
    diffsWnaive=pd.DataFrame(np.sqrt(np.square(111*np.asarray(latnaive3-latreal3))+np.square(111*np.asarray(lonnaive3-lonreal3))))

    thetasWnaive = pd.DataFrame(np.arctan2(111*np.asarray(latnaive3-latreal3),111*np.asarray(lonnaive3-lonreal3)))


    diffsW=pd.DataFrame(np.sqrt(np.square(111*np.asarray(latpred3-latreal3))+np.square(111*np.asarray(lonpred3-lonreal3))))

    thetasW = pd.DataFrame(np.arctan2(111*np.asarray(latpred3-latreal3),111*np.asarray(lonpred3-lonreal3)))
    diffsW4x=pd.DataFrame(np.sqrt(np.square(111*np.asarray(latpred34x-latreal3))+np.square(111*np.asarray(lonpred34x-lonreal3))))


    thetasW4x = pd.DataFrame(np.arctan2(111*np.asarray(latpred34x-latreal3),111*np.asarray(lonpred34x-lonreal3)))



    diffsWL=pd.DataFrame(np.sqrt(np.square(111*np.asarray(latpred3L-latreal3))+np.square(111*np.asarray(lonpred3L-lonreal3))))

    thetasWL = pd.DataFrame(np.arctan2(111*np.asarray(latpred3L-latreal3),111*np.asarray(lonpred3L-lonreal3)))
    diffsW4xL=pd.DataFrame(np.sqrt(np.square(111*np.asarray(latpred34xL-latreal3))+np.square(111*np.asarray(lonpred34xL-lonreal3))))


    thetasW4xL = pd.DataFrame(np.arctan2(111*np.asarray(latpred34xL-latreal3),111*np.asarray(lonpred34xL-lonreal3)))



    fig = plt.figure(figsize=(10,7))

    width = 120


    width2=21

    plt.plot(range(0,6*width,6)[0:width2], np.nanmean(diffsWnaive,axis = 0)[0:width2], linewidth = 5, zorder =1, label = 'Naive model', color = 'k', linestyle = '--')



    kmpernm=1.852
    plt.plot([0,24,48,72,96,120],kmpernm*np.array([0,48,75,90, 120,150]), label = 'National Hurricane Center model', linewidth =5, c='k')


    plt.plot(range(0,6*width,6)[0:width2], np.nanmean(diffsW4x,axis = 0)[0:width2], linewidth = 5, zorder =1, label = '6-hourly model', c = 'cornflowerblue')
    yerr2=np.nanstd(diffsW4x,axis = 0)[0:width2]
    plt.fill_between(range(0,6*width,6)[0:width2], np.nanmean(diffsW4x,axis = 0)[0:width2]-yerr2, np.nanmean(diffsW4x,axis = 0)[0:width2]+yerr2, alpha = 0.2, zorder = -1, color = 'cornflowerblue')
    plt.plot(range(0,6*width,6)[0:width2], np.nanmean(diffsW,axis = 0)[0:width2], linewidth = 5, zorder =1, label = 'Monthly model', c = 'tomato')

    yerr2=np.nanstd(diffsW,axis = 0)[0:width2]
    plt.fill_between(range(0,6*width,6)[0:width2], np.nanmean(diffsW,axis = 0)[0:width2]-yerr2, np.nanmean(diffsW,axis = 0)[0:width2]+yerr2, alpha = 0.1, zorder = -1, color = 'tomato')


    plt.xlim([0,120])
    plt.xlabel('Hours since genesis')
    plt.ylabel('Distance from actual track (km)')
    plt.legend()

    fig.savefig('figures/errstime.jpg', dpi =400)

    # add a line of a naive non-moving prediction, or keeps going at same rate


    fig, ax = plt.subplots(ncols=1,nrows=1,figsize=(12,8),subplot_kw=dict(projection="polar"),dpi = 300)



    cmap = plt.cm.get_cmap('gist_rainbow', 20)

    radii, thethetas, thethetasSTD=get_stats(diffsW,thetasW)
    thethetas = np.insert(np.array(thethetas),0,0)
    radii = np.insert(np.array(radii),0,0)
    plot_stds(diffsW, thetasW, 'tomato',ax)


    ax.plot(thethetas,radii, color = 'tomato', zorder =0, linewidth =7, label = 'Monthly model')
    pc = ax.scatter(thethetas,radii, c = range(0,20*6,6),cmap = cmap,vmin =-3, vmax = 117, zorder=1,s=10)


    radii, thethetas, thethetasSTD=get_stats(diffsW4x,thetasW4x)
    thethetas = np.insert(np.array(thethetas),0,0)
    radii = np.insert(np.array(radii),0,0)
    ax.plot(thethetas,radii, color = 'cornflowerblue', zorder =1, linewidth =7, label = '6-hourly model')
    plot_stds(diffsW4x, thetasW4x, 'cornflowerblue',ax)
    pc = ax.scatter(thethetas,radii, c = range(0,20*6,6),cmap = cmap,vmin =-3, vmax =117,s=10, zorder =2)





    axins = inset_axes(ax,width="3%",height="100%", loc='upper left',
                                       bbox_to_anchor=(1.02, 0., 1, 1),
                                       bbox_transform=ax.transAxes,borderpad=2) 

    fig.colorbar(pc, axins, ticks = range(0,21*6,6), label='hours')
    ax.legend(loc = 'lower left')
    # ax[m-1].set_title(str(6*m)+' hours out')

    ax.set_rlabel_position(235)
    # ax.set_rlabel("km")
    fig.savefig('figures/errsradial.jpg')