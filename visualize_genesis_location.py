
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


    

# plot genesis locations for each cluster
def gen_plot(df,num_ensembles, num_clusters):
    prob = False
    NsHist = df[df['ensemble']==num_ensembles].groupby('cluster').count()['firstlat'].values
    NsModel = df[df['ensemble']<num_ensembles].groupby('cluster').count()['firstlat'].values
    
    

 
    plt.rcParams.update({'font.size': 18})
    fig, ax = plt.subplots(ncols=2,nrows=3,figsize=(18,15), sharex=False,subplot_kw={'projection': ccrs.PlateCarree(central_longitude=0)},dpi = 300)
    for c in range(num_clusters): 
        df2 = df[(df['cluster']==c)&(df['firstlat']<50)]
        xedges = range(240,360,4)
        yedges = range(0,90,4)
        x = df2['firstlon']
        y = df2['firstlat']
        xAll = df['firstlon']
        yAll = df['firstlat']
        H, xedges, yedges = np.histogram2d(x, y, bins=(xedges, yedges), normed = False)
        H2, xedges, yedges = np.histogram2d(xAll, yAll, bins=(xedges, yedges), normed = False)


        H = H.T
        H2=H2.T
        if prob: 
            H3= H/H2
        else: 
            H3=H
        X, Y= np.meshgrid(xedges, yedges)
        first = int(c/2)
        second = int((c%2) !=0)
        cmaps = [plumT,midnightblueT,seagreenT,tomatoT]
        im = ax[first,second].pcolormesh(X, Y, np.ma.masked_where(H3==0,H3), cmap = cmaps[c], label = 'Cluster '+str(int(c+1)),vmin = 0,vmax =50)

        ax[first,second].coastlines(color = 'steelblue')
        ax[first,second].set_extent([-110,10,0,70], crs=ccrs.PlateCarree())
#         ax[first,second].add_feature(cartopy.feature.OCEAN, facecolor='aliceblue')
        ax[first,second].add_feature(cartopy.feature.LAND, facecolor= 'ivory',edgecolor='steelblue')
        ax[first,second].add_feature(cartopy.feature.LAKES, facecolor='white',edgecolor='lightsteelblue')
        ax[first,second].add_feature(cartopy.feature.RIVERS, edgecolor ='aliceblue')

        
        ax[2,0].coastlines(color = 'steelblue')
        ax[2,0].set_extent([-110,10,0,70], crs=ccrs.PlateCarree())
#         ax[first,second].add_feature(cartopy.feature.OCEAN, facecolor='aliceblue')
        ax[2,0].add_feature(cartopy.feature.LAND, facecolor= 'ivory',edgecolor='steelblue')
        ax[2,0].add_feature(cartopy.feature.LAKES, facecolor='white',edgecolor='lightsteelblue')
        ax[2,0].add_feature(cartopy.feature.RIVERS, edgecolor ='aliceblue')
        ax[2,0].scatter(x,y,c = colorList3[c],transform=ccrs.PlateCarree(), label = 'Cluster '+str(int(c+1)), s= 2)
        ax[2,0].set_yticks([0,15,30,45,60], crs = ccrs.PlateCarree())
        lat_formatter = LatitudeFormatter()
        ax[2,0].yaxis.set_major_formatter(lat_formatter)
        lgnd =ax[2,0].legend(markerscale=10)

        ax[first,second].legend(title='Cluster '+str(int(c+1)),
                            loc = 'upper left')
#         ax[first,second].legend(title='Cluster '+str(int(c+1))+
#                             '\nN (historical) = ' +str(NsHist[c])+
#                             '\nN (model) = '+str(NsModel[c]),
#                             loc = 'upper left')
        if second ==0:   
            ax[first,second].set_yticks([0,15,30,45,60], crs = ccrs.PlateCarree())
            lat_formatter = LatitudeFormatter()
            ax[first,second].yaxis.set_major_formatter(lat_formatter)

        axins = inset_axes(ax[first,second],width="3%",height="100%", loc='lower left',
                                   bbox_to_anchor=(1.02, 0., 1, 1),
                                   bbox_transform=ax[first,second].transAxes,borderpad=0) 
        
        for i in [0,1]:
            lon_formatter = LongitudeFormatter()

            ax[2,i].xaxis.set_major_formatter(lon_formatter)
            ax[2,i].set_xticks([ -90, -60,-30,0], crs = ccrs.PlateCarree())
            
        if prob: 
            if second==1:


                fig.colorbar(im, axins,ticks = [0,0.25,0.5,0.75,1], label= 'Probability of assignment to cluster x')
            else:
                fig.colorbar(im, axins,ticks = [0,0.25,0.5,0.75,1])
        else: 
            if second==1:


                fig.colorbar(im, axins, label= 'Number of points')
            else:
                fig.colorbar(im, axins)
            if first ==1:
                if second ==1:
                    fig.colorbar(im, axins, label= 'Number of points')
                    
                    
                    
                    
         
    prob = True    
    for c in range(num_clusters):  
            # last subplot
        df2 = df[(df['cluster']==c)&(df['firstlat']<50)]
        xedges = range(240,360,4)
        yedges = range(0,90,4)
        x = df2['firstlon']
        y = df2['firstlat']
        xAll = df['firstlon']
        yAll = df['firstlat']
        H, xedges, yedges = np.histogram2d(x, y, bins=(xedges, yedges), normed = False)
        H2, xedges, yedges = np.histogram2d(xAll, yAll, bins=(xedges, yedges), normed = False)

    # #     ax = fig.add_subplot(132, title='pcolormesh: actual edges', aspect='equal',subplot_kw={'projection': ccrs.PlateCarree(central_longitude=0)} )


    #     h =ax.hist2d(x, y, bins=[xedges,yedges])
        H = H.T
        H2=H2.T
        if prob: 
            H3= H/H2
        else: 
            H3=H
        X, Y= np.meshgrid(xedges, yedges)
        first = int(c/2)
        second = int((c%2) !=0)
        cmaps = [plumTT,midnightblueTT,seagreenTT,tomatoTT]
        im = ax[2,1].pcolormesh(X, Y, np.ma.masked_where(H2<0,H3), cmap = cmaps[c],vmin =0.25,vmax = 1, label = 'cluster '+str(int(c+1)))
   
        ax[2,1].coastlines(color = 'steelblue')
        ax[2,1].set_extent([-110,10,0,70], crs=ccrs.PlateCarree())
        ax[2,1].add_feature(cartopy.feature.OCEAN, facecolor='white')
        ax[2,1].add_feature(cartopy.feature.LAND, facecolor= 'ivory',edgecolor='steelblue')
        ax[2,1].add_feature(cartopy.feature.LAKES, facecolor='white',edgecolor='lightsteelblue')
        ax[2,1].add_feature(cartopy.feature.RIVERS, edgecolor ='aliceblue')


#         if second ==0:   
#             ax[2,1].set_yticks([0,15,30,45,60], crs = ccrs.PlateCarree())
#             lat_formatter = LatitudeFormatter()
#             ax[2,1].yaxis.set_major_formatter(lat_formatter)

        axins = inset_axes(ax[2,1],width="3%",height="100%", loc='lower left',
                                   bbox_to_anchor=(1.02, 0., 1, 1),
                                   bbox_transform=ax[2,1].transAxes,borderpad=0) 

        for i in [0,1]:
            lon_formatter = LongitudeFormatter()

            ax[2,1].xaxis.set_major_formatter(lon_formatter)
            ax[2,1].set_xticks([ -90, -60,-30,0], crs = ccrs.PlateCarree())
        from matplotlib.lines import Line2D

        fig.colorbar(matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=0.25, vmax=1),cmap=blackT),axins, label= 'Prob. of assignment to cluster',ticks = [0.2,0.4,0.6,0.8,1])    

        custom_lines = [Line2D([0], [0], color='plum', lw=15),
                Line2D([0], [0], color='midnightblue', lw=15),
                Line2D([0], [0], color='seagreen', lw=15),
                   Line2D([0], [0], color='tomato', lw=15)]
        ax[2,1].legend(custom_lines, ['Cluster 1', 'Cluster 2', 'Cluster 3','Cluster 4'], loc = 'upper left')


            
        
    plt.subplots_adjust(hspace = 0.07, wspace=0.11)
    fig.savefig('/tigress/gkortum/thesisSummer/figures/genesisdistribution.jpg')

def gen_plot2(df,num_ensembles, num_clusters, Hist):
    prob = False
    NsHist = df[df['ensemble']==num_ensembles].groupby('cluster').count()['firstlat'].values
    NsModel = df[df['ensemble']<num_ensembles].groupby('cluster').count()['firstlat'].values

 
    plt.rcParams.update({'font.size': 12})
    fig, ax = plt.subplots(ncols=1,nrows=4,figsize=(6,20), sharex=False,subplot_kw={'projection': ccrs.PlateCarree(central_longitude=0)},dpi = 300)
    for c in range(num_clusters): 
        if Hist: df = df[df['ensemble']==num_ensembles]
        df2 = df[df['cluster']==c]
        xedges = range(240,360,5)
        yedges = range(0,90,5)
        x = df2['firstlon']
        y = df2['firstlat']
        if c ==0: 
            position =3
        elif c ==3: 
            position =0
        else: 
            position = c


        ax[c].coastlines(color = 'steelblue')
        ax[c].set_extent([-110,10,0,70], crs=ccrs.PlateCarree())
#         ax[first,second].add_feature(cartopy.feature.OCEAN, facecolor='aliceblue')
        ax[c].add_feature(cartopy.feature.LAND, facecolor= 'ivory',edgecolor='steelblue')
        ax[c].add_feature(cartopy.feature.LAKES, facecolor='white',edgecolor='lightsteelblue')
        ax[c].add_feature(cartopy.feature.RIVERS, edgecolor ='aliceblue')
        lon_formatter = LongitudeFormatter()

        ax[c].xaxis.set_major_formatter(lon_formatter)
        ax[c].set_xticks([ -90, -60,-30,0], crs = ccrs.PlateCarree())

#         ax[c].legend(title='cluster '+str(int(c+1))+
#                             '\nN (historical) = ' +str(NsHist[c])+
#                             '\nN (model) = '+str(NsModel[c]),
#                             loc = 'upper left')

        ax[c].set_yticks([0,15,30,45,60], crs = ccrs.PlateCarree())
        lat_formatter = LatitudeFormatter()
        ax[c].yaxis.set_major_formatter(lat_formatter)
        ax[position].scatter(x,y,color = colorList3[int(c)], transform=ccrs.PlateCarree(), label = 'cluster '+str(int(c+1)))
              
        leg = ax[position].legend(loc = 'upper left')
#         axins = inset_axes(ax[c],width="3%",height="100%", loc='lower left',
#                                    bbox_to_anchor=(1.02, 0., 1, 1),
#                                    bbox_transform=ax[c].transAxes,borderpad=0) 



            
#         if prob: 
#             if second==1:


#                 fig.colorbar(im, axins,ticks = [0,0.25,0.5,0.75,1], label= 'likelihood of being in cluster')
#             else:
#                 fig.colorbar(im, axins,ticks = [0,0.25,0.5,0.75,1])
#         else: 
#             if second==1:


#                 fig.colorbar(im, axins,ticks = [0,10,20,30,40,50,60,70,80,90], label= 'number of points')
#             else:
#                 fig.colorbar(im, axins,ticks = [0,10,20,30,40,50,60,70,80,90])
    fig.savefig('/tigress/gkortum/thesisSummer/figures/genesisdistribution.jpg')
   







 #plot likelihood of genesis plots
def gen_plot_prob(df, num_clusters):
    prob = True
    plt.rcParams.update({'font.size': 18})
    fig, ax = plt.subplots(ncols=1,nrows=1,figsize=(18,10), sharex=False,subplot_kw={'projection': ccrs.PlateCarree(central_longitude=0)},dpi = 300)
    for c in range(num_clusters): 
        df2 = df[df['cluster']==c]
        xedges = range(240,360,5)
        yedges = range(0,90,5)
        x = df2['firstlon']
        y = df2['firstlat']
        xAll = df['firstlon']
        yAll = df['firstlat']
        H, xedges, yedges = np.histogram2d(x, y, bins=(xedges, yedges), normed = False)
        H2, xedges, yedges = np.histogram2d(xAll, yAll, bins=(xedges, yedges), normed = False)

    # #     ax = fig.add_subplot(132, title='pcolormesh: actual edges', aspect='equal',subplot_kw={'projection': ccrs.PlateCarree(central_longitude=0)} )


    #     h =ax.hist2d(x, y, bins=[xedges,yedges])
        H = H.T
        H2=H2.T
        if prob: 
            H3= H/H2
        else: 
            H3=H
        X, Y= np.meshgrid(xedges, yedges)
        first = int(c/2)
        second = int((c%2) !=0)
        cmaps = [plumTT,midnightblueTT,seagreenTT,tomatoTT]
        im = ax.pcolormesh(X, Y, np.ma.masked_where(H2<5,H3), cmap = cmaps[c],vmin =0.25,vmax = 1, label = 'cluster '+str(int(c+1)))
   
        ax.coastlines(color = 'steelblue')
        ax.set_extent([-110,10,0,70], crs=ccrs.PlateCarree())
        ax.add_feature(cartopy.feature.OCEAN, facecolor='white')
        ax.add_feature(cartopy.feature.LAND, facecolor= 'ivory',edgecolor='steelblue')
        ax.add_feature(cartopy.feature.LAKES, facecolor='white',edgecolor='lightsteelblue')
        ax.add_feature(cartopy.feature.RIVERS, edgecolor ='aliceblue')


        if second ==0:   
            ax.set_yticks([0,15,30,45,60], crs = ccrs.PlateCarree())
            lat_formatter = LatitudeFormatter()
            ax.yaxis.set_major_formatter(lat_formatter)

        axins = inset_axes(ax,width="3%",height="100%", loc='lower left',
                                   bbox_to_anchor=(1.02, 0., 1, 1),
                                   bbox_transform=ax.transAxes,borderpad=0) 

        for i in [0,1]:
            lon_formatter = LongitudeFormatter()

            ax.xaxis.set_major_formatter(lon_formatter)
            ax.set_xticks([ -90, -60,-30,0], crs = ccrs.PlateCarree())
        from matplotlib.lines import Line2D

        fig.colorbar(matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=0.25, vmax=1),cmap=blackT),axins, label= 'Probability of assignment to cluster x',ticks = [0.25,0.5,0.75,1])    

        custom_lines = [Line2D([0], [0], color='plum', lw=15),
                Line2D([0], [0], color='midnightblue', lw=15),
                Line2D([0], [0], color='seagreen', lw=15),
                   Line2D([0], [0], color='tomato', lw=15)]


        ax.legend(custom_lines, ['Cluster 1', 'Cluster 2', 'Cluster 3','Cluster 4'], loc = 'upper left')
    fig.savefig('/tigress/gkortum/thesisSummer/figures/genesislikelihood.jpg')
    
 