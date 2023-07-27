
#import libraries
import numpy as np
import xarray as xr
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.gridspec as gridspec

from shapely.geometry import LineString
import sklearn
import seaborn as sns
import random
import pickle
from joblib import Parallel, delayed

from format import *
from restructure_cluster_datasets import *
from analyze_clusters import *

plt.rcParams.update({'font.size': 18})
def yearCluster01(full,n, num_clusters):
        full2 = full[(full['dataset']==n)&(full['Year']<2022)]
        yearCluster= full2[['Year','cluster']]
        yearCluster['ones']= np.ones(yearCluster.shape[0])
    
        
        list1=[]
        list2=[]
        list3=[]
        list4=[]
        lists = [list1,list2,list3,list4]
        for i in range(len(yearCluster['cluster'])):
            for j in range(4):
                if yearCluster['cluster'].values[i]==j: 
                    lists[j].append(1)
                else: lists[j].append(0)
        yearCluster['cluster0']=list1
        yearCluster['cluster1']=list2
        yearCluster['cluster2']=list3
        if num_clusters ==4: 
            yearCluster['cluster3']=list4
        
        #add for above and below 1995
        ones=np.ones(yearCluster['Year'].values.shape[0])
        zeros=np.zeros(yearCluster['Year'].values.shape[0])
        above1995=np.where(yearCluster['Year'].values>1995, ones,zeros)
        yearCluster['above1995']=above1995
        return yearCluster
    
def get_the_dfs(num_ensembles, full, num_clusters,hurricane_only = False): 
    thedfs= []
    for n in range(num_ensembles+1):
        thedfs.append(yearCluster01(full,n, num_clusters))
        
    if hurricane_only: 

        thedfsNEW = []
        for i in range(len(thedfs)):
            a = thedfs[i]
            a['maxwind']= full[full['dataset']==i]['maxwind'].values
            a = a[a['maxwind']>=33]
            a =a.drop(['maxwind'], axis =1)
            thedfsNEW.append(a)
        thedfs = thedfsNEW
    return thedfs
    
def clusteredBar2(fakedf, num_clusters):
        yearCluster = fakedf
        yearClusterSum = yearCluster.groupby('above1995').sum()
        b=np.zeros(len(yearClusterSum.index.values))
        clusterlist = ['cluster0', 'cluster1','cluster2','cluster3']
        # want to have the df rows be the clusters,
        #want to have the df columns be the model run
        #from this return a column from this model run including: percent change in sum
        # and also percentage point change in cluster proportion
        clusterChanges = []
        
        #total percent change: 
        for i in range(num_clusters):
            b = np.add(b,yearClusterSum[clusterlist[i]].values)
            height=np.divide(yearClusterSum[clusterlist[i]].values, yearClusterSum['ones'].values)
            clusterChanges.append(height[1]-height[0])
        clusterChanges.append((np.true_divide(b[1],26)-np.true_divide(b[0],26))/(0.5*((np.true_divide(b[0],26))+(np.true_divide(b[1],26)))))
             
             

        return clusterChanges
    
def get_the_df(num_ensembles, theClusters, full,hurricane_only = False):

    for i in range(num_ensembles):

        if i ==0:
            thedf = theClusters[i].yearCluster01()
            thedf['ensemble']=np.ones(thedf.shape[0])*i
        else: 
            thedf2= theClusters[i].yearCluster01()
            thedf2['ensemble']=np.ones(thedf2.shape[0])*i
            thedf=thedf.append(thedf2)


    thedf=thedf.reset_index().drop('index',axis =1)  
    if hurricane_only: 
        thedf['maxwind']= full[full['dataset']<num_ensembles]['maxwind'].values
        thedf = thedf[thedf['maxwind']>=33]
        thedf =thedf.drop(['maxwind'], axis = 1)
    return thedf

def get_the_df_HurricaneAdd(num_ensembles, theClusters, full):

    for i in range(num_ensembles+1):

        if i ==0:
            thedf = theClusters[i].yearCluster01()
            thedf['ensemble']=np.ones(thedf.shape[0])*i
        else: 
            thedf2= theClusters[i].yearCluster01()
            thedf2['ensemble']=np.ones(thedf2.shape[0])*i
            thedf=thedf.append(thedf2)


    thedf=thedf.reset_index().drop('index',axis =1)  

    thedf['maxwind']= full['maxwind'].values
#         thedf = thedf[thedf['maxwind']>=33]
#         thedf =thedf.drop(['maxwind'], axis = 1)
    thedf['hurricane'] = thedf['maxwind']>=33
    thedf['dataset']=full['dataset'].values
    return thedf

def theadd(m, num_ensembles, thedf, bootstraps, boot2, columnsFull, num_clusters):
    ensembles = []
    for k in range(1970,2022,1):

        therandom=random.randint(0, num_ensembles-1)
        ensembles.append(therandom)
        if k ==1970: 
            fakedf=thedf[(thedf['Year']==k)& (thedf['ensemble']==therandom)]
        else: 
            fakedf2=thedf[(thedf['Year']==k)& (thedf['ensemble']==therandom)]
            fakedf=fakedf.append(fakedf2)
    fakedf=fakedf.reset_index().drop('index',axis=1)

    
    bootstraps.append(fakedf)
    c = clusteredBar2(fakedf, num_clusters)
    boot2.append(c)
    columnsFull.append(np.asarray(c))


def get_columns_Full(num_boot, num_ensembles, thedf, num_clusters):
    bootstraps=[]
    boot2=[]
    columnsFull = []

    for m in range(num_boot):

        theadd(m, num_ensembles, thedf, bootstraps, boot2, columnsFull, num_clusters)
    return columnsFull

def make_df(columns, num_clusters):
    dfClusteredBar = pd.DataFrame(np.asarray(columns))
    dfClusteredBar = dfClusteredBar.multiply(100)
    dfClusteredBar = dfClusteredBar.append(dfClusteredBar.mean(), ignore_index = True)
    # dfClusteredBar['label'] = ['ensemble 1','ensemble 2','ensemble 3','ensemble 4','ensemble 5','ensemble 1a','ensemble 2a','ensemble 3a','average' ]
    if num_clusters ==4: 
        theCOLS= {0: "Proportion cluster 0",1: "Proportion cluster 1",2: "Proportion cluster 2",3: "Proportion cluster 3", 4:'total number storms'}
    else: 
        theCOLS={0: "Proportion cluster 0",1: "Proportion cluster 1",2: "Proportion cluster 2",3:'total number storms'}
    dfClusteredBar=dfClusteredBar.rename(columns=theCOLS)
    # dfClusteredBar = dfClusteredBar.set_index('label')
    return dfClusteredBar

def plot_cluster_hists(dfClusteredBar, thedfs, num_ensembles, num_clusters, smooth_param=1): 
    fig,ax = plt.subplots(ncols = 2,nrows = 2,figsize=(19,12),dpi = 300)
    for i in range(num_clusters):
        c = i
        first = int(c/2)
        second = int((c%2) !=0)
    #     dfClusteredBar = make_df(columnsFull)
        # sns.kdeplot(dfClusteredBar['Proportion cluster 0'].values)
        # plt.plot(np.zeros(10),range(0,10,1))

        m = clusteredBar2(thedfs[num_ensembles], num_clusters)[i]*100
    #     print(m)
        # plt.plot(np.ones(10)*m,range(0,10,1))



        y = dfClusteredBar['Proportion cluster '+str(i)].values
        kde = sklearn.neighbors.KernelDensity(kernel='gaussian', bandwidth=smooth_param).fit(y.reshape(-1, 1))

        X_plot = np.linspace(-17,17,1000).reshape(-1,1)
        log_dens = kde.score_samples(X_plot)

        

#         ax[first,second].plot(X_plot, np.exp(log_dens), lw = 4, c = colorList3[i], label = 'bootstrap distribution')
        sns.kdeplot(y,shade = True, alpha = 0.25, linewidth = 0.2, bw_adjust = smooth_param,color = colorList3[i], ax = ax[first,second])
        sns.kdeplot(y,shade = False, alpha = 1, linewidth = 4, bw_adjust = smooth_param,color = colorList3[i], ax = ax[first,second], label = 'Bootstrap distribution')
        ax[first,second].set_ylim(0,0.13)
        ax[first,second].set_xlim(-17,17)


        fmt = '%.0f%%' # Format you want the ticks, e.g. '40%'
        xticks = matplotlib.ticker.FormatStrFormatter(fmt)
        ax[first,second].xaxis.set_major_formatter(xticks)

        first_line = LineString(np.column_stack((np.ones(10)*m,range(0,10,1))))


        second_line = LineString(np.column_stack((X_plot, np.exp(log_dens))))
        intersection = first_line.intersection(second_line)

        if intersection.geom_type == 'MultiPoint':
            plt.plot(*LineString(intersection).xy, '*')
        elif intersection.geom_type == 'Point':

            the_y = [0,intersection.y]

            ax[first,second].plot(np.ones(2)*intersection.x,the_y, c = colorList3[i], lw = 5,linestyle = '--', label = 'Observed')

            ax[first,second].plot(*intersection.xy, '*', ms=23, markerfacecolor ='w',markeredgecolor='k',markeredgewidth = 2)

        l = ax[first,second].legend(title = 'Cluster '+ str(int(c+1)),loc = 'upper left')

        ax[first,second].plot(np.zeros(2),[0,0.13], linestyle = '--', c ='slategrey' )
        if first ==1:
            ax[first,second].set_xlabel('Difference in % of storms in cluster')

        if first ==0: 
            ax[first,second].set_xticks([])
        if second ==0: 
            ax[first,second].set_ylabel('Probability density')
        if second ==1: 
            ax[first,second].set_ylabel('')
            ax[first,second].set_yticks([])
    plt.subplots_adjust(hspace = 0.07, wspace=0.11)
    fig.savefig('/tigress/gkortum/thesisSummer/figures/PDFs_clustershift_boot.jpg')
    
def plot_freq_hist(dfClusteredBar, thedfs, num_ensembles, num_clusters):

    fig,ax = plt.subplots(ncols = 1,nrows = 1,figsize=(9,6),dpi = 300)

    # dfClusteredBar = make_df(columnsFull)




    y = dfClusteredBar['total number storms'].values
    kde = sklearn.neighbors.KernelDensity(kernel='gaussian', bandwidth=0.8).fit(y.reshape(-1, 1))

    X_plot = np.linspace(0,100,1000).reshape(-1,1)
    log_dens = kde.score_samples(X_plot)



    ax.plot(X_plot, np.exp(log_dens), lw = 1, c = 'navy', alpha = 0)
    sns.kdeplot(y,shade = True, alpha = 0.25, linewidth = 2, color = 'cornflowerblue', ax = ax,bw_adjust = 1)
    sns.kdeplot(y,shade = False, alpha = 1, linewidth = 5, color = 'cornflowerblue', ax = ax, label ='Bootstrap distribution',bw_adjust = 1)
    ax.set_ylim(0,0.06)
    ax.set_xlim(0,100)
#     ax.set_xticks([0,10,20,30,40,50,60,70,80,90,100])



    # m = thechange*100
#     m = 51
    m = clusteredBar2(thedfs[num_ensembles], num_clusters)[num_clusters]*100
    first_line = LineString(np.column_stack((np.ones(10)*m,range(0,10,1))))


    second_line = LineString(np.column_stack((X_plot, np.exp(log_dens))))
    intersection = second_line.intersection(first_line)

    if intersection.geom_type == 'MultiPoint':
        plt.plot(*LineString(intersection).xy, '*')
        print('a')
    elif intersection.geom_type == 'Point':

        the_y = [0,intersection.y]
        ax.plot(np.ones(2)*intersection.x,the_y, c = 'cornflowerblue', lw = 5,linestyle = '--', label = 'Observed')

        ax.plot(intersection.x,intersection.y, '*', ms=23, markerfacecolor ='w',markeredgecolor='k',markeredgewidth = 2)



    ax.set_ylabel('Probability density')  
    ax.set_xlabel('% Change in storms per year')
    fmt = '%.0f%%' # Format you want the ticks, e.g. '40%'
    xticks = matplotlib.ticker.FormatStrFormatter(fmt)
    ax.xaxis.set_major_formatter(xticks)


    # ax.plot(np.ones(2)*m,[0,0.05], linewidth = 5, linestyle = 'dotted', label = 'historical', c='mediumvioletred')

    ax.legend(loc = 'upper left')

    fig.savefig('/tigress/gkortum/thesisSummer/figures/PDF_totalshift_boot.jpg')

    
    
    
### PART 2: extreme scenario bootstrap ensemble members


def theadd_Best(m, thedf, num_ensembles):
    ensembles = []
    for k in range(1970,2022,1):

        therandom=random.randint(0, num_ensembles-1)
        ensembles.append(therandom)
        if k ==1970: 
            fakedf=thedf[(thedf['Year']==k)& (thedf['ensemble']==therandom)]
        else: 
            fakedf2=thedf[(thedf['Year']==k)& (thedf['ensemble']==therandom)]
            fakedf=fakedf.append(fakedf2)
    fakedf=fakedf.reset_index().drop('index',axis=1)
    
    a = fakedf.groupby('above1995').sum()
    try: pre1995prop0 = np.true_divide((a['cluster1'].values[0]),(a['ones'].values[0]))
    except KeyError: print(a)
    post1995prop0 = np.true_divide((a['cluster1'].values[1]),(a['ones'].values[1]))
    pre1995prop2 = np.true_divide((a['cluster2'].values[0]),(a['ones'].values[0]))
    post1995prop2 = np.true_divide((a['cluster2'].values[1]),(a['ones'].values[1]))
#     post1995prop = post1995prop1+post1995prop2
#     pre1995prop = pre1995prop1+pre1995prop2
    changepct0 = np.true_divide((post1995prop0-pre1995prop0), 0.5*(pre1995prop0+post1995prop0))
    changepct2 = np.true_divide((post1995prop2-pre1995prop2), 0.5*(pre1995prop2+post1995prop2))
#     changepct3= np.true_divide((post1995prop3-pre1995prop3), pre1995prop3)
#     changepct = changepct3-changepct2
    changepct = changepct0-changepct2
    return [changepct, ensembles]
    




def dumpBoots(thedf, num_ensembles, num_samples):
    hibestens = []
    lowbestens = []
    hichangessofar=[]
    lowchangessofar=[]
    lowchangesofar = 0
    hichangesofar = 0
    n = 0
    changesEnsembles=Parallel(n_jobs=80, verbose = 60)(delayed(theadd_Best)(m, thedf, num_ensembles) for m in range(num_samples))
    
    dftosort = pd.DataFrame(np.array(changesEnsembles))
    dftosort.columns = ['changepct','ensembles']
    dfsorted = dftosort.sort_values(by = 'changepct').reset_index().drop('index', axis = 1)
    changes = dfsorted['changepct'].values
    hibestens=dfsorted['ensembles'].values
    for i in range(0,100,1):            
        with open('data/bootstrap ensembles 3/lowbestens_'+str(i), 'wb') as f:

            pickle.dump(hibestens[i],f)

        with open('data/bootstrap ensembles 3/hibestens_'+str(i), 'wb') as f:
            k = -1*(i+1)
            pickle.dump(hibestens[k],f)
        


