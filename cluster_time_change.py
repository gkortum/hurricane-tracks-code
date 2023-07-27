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


from format import *
from restructure_cluster_datasets import *
from analyze_clusters import *


    
def get_percent_change(full, number, hurricane_only=False):
    if hurricane_only: 
        full2 = full[full['maxwind']>=33]
    else: full2 = full
    full_hist_counts=full2[full2['dataset']==number]
    full_hist_counts=full_hist_counts[full_hist_counts['Year']<2022]
    full_hist_counts['above']=np.where(full_hist_counts['Year'].values>1995, 1,0)
    thechange =np.true_divide((



        np.true_divide(full_hist_counts.groupby('above').count().iloc[1,0], 26)-


        np.true_divide(full_hist_counts.groupby('above').count().iloc[0,0],26)),



        0.5*(np.true_divide(full_hist_counts.groupby('above').count().iloc[1,0], 26)+


        np.true_divide(full_hist_counts.groupby('above').count().iloc[0,0],26)))


    return thechange

def get_df_for_clusteredBar(theClusters, full, num_ensembles, hurricane_only = False): 
    columns1 = []
    for i in range(num_ensembles+1):
        c = theClusters[i]
        columns1.append(np.asarray(c.clusteredBar(num_ensembles, full, hurricane_only)))
    dfClusteredBar = pd.DataFrame(np.asarray(columns1))

    dfClusteredBar = dfClusteredBar.multiply(100)
    dfClusteredBar2=dfClusteredBar.iloc[:-1, :]
    dfClusteredBar2 = dfClusteredBar2.append(dfClusteredBar2.mean(), ignore_index = True)
    dfClusteredBar2 = dfClusteredBar2.append(dfClusteredBar.iloc[-1,:], ignore_index=True)
    
    
    
    percentchanges = [] 
    for i in range(num_ensembles+1):

        percentchanges.append(get_percent_change(full, i, hurricane_only))
        if i ==num_ensembles-1: 
            percentchanges.append(np.mean(np.array(percentchanges)))


    dfClusteredBar2['totalchange']= np.array(percentchanges)*100
    dfClusteredBar = dfClusteredBar2
    return dfClusteredBar










def plot_ClusteredBar(dfClusteredBar, num_ensembles, num_clusters): 
    


    plt.rcParams.update({'font.size': 20})   
    fig,ax = plt.subplots(ncols = 2,nrows = 2,figsize=(19,12),dpi = 300, sharex = True, sharey = True)
#     fig2 = plt.figure(constrained_layout=True,figsize=(9*5,9))
#     widths = [1,1,1,1, 1]
#     spec2 = gridspec.GridSpec(ncols=5, nrows=1, figure=fig2, width_ratios = widths)


    colors = ['cornflowerblue','cornflowerblue','cornflowerblue','cornflowerblue','cornflowerblue','cornflowerblue','cornflowerblue','cornflowerblue','cornflowerblue','cornflowerblue','cornflowerblue','cornflowerblue']

    edges = ['','','','','','','','','midnightblue','k','midnightblue','maroon','midnightblue','maroon','midnightblue','maroon']
    num_list = ['Ensemble 1','Ensemble 2','Ensemble 3','Ensemble 4','Ensemble 5','Ensemble 6','Ensemble 7','Ensemble 8','Ensemble 9', 'Ensemble 10']

    dfClusteredBar['ensemble member'] = num_list[0:num_ensembles]+['Ensemble Mean','Observed']
    if num_clusters ==4: 
        theCOLS = {0: "Cluster 1",1: "Cluster 2",2: "Cluster 3",3: "Cluster 4", 4:'Number of storms'}
    else: theCOLS = {0: "Proportion cluster 1",1: "Proportion cluster 2",2: "Proportion cluster 3", 3:'Number of storms'}
    dfClusteredBar=dfClusteredBar.rename(columns=theCOLS)
    dfClusteredBar = dfClusteredBar.set_index('ensemble member')






    # bar2 = dfClusteredBar.drop(columns = ['Number of storms']).T.plot(kind='bar', ax = a, width = 0.6, color = colors)
    thecols = ['plum','midnightblue','seagreen','tomato']
    for i in range(num_clusters):
        if i ==0: 
            a = ax[0,0]
        elif i == 1: 
            a = ax[0,1]
        elif i ==2: 
            a = ax[1,0]
        elif i ==3: 
            a = ax[1,1]
#         a = fig2.add_subplot(spec2[i])
        bar2 = dfClusteredBar.iloc[:,i].plot(kind = 'bar', ax = a, width = 0.9, color = thecols[i])
        a.legend([""],[""],title = "Cluster "+str(i+1), loc = 'lower left')
        a.set_ylim([-15,12])
        a.set_yticks([-12,-8,-4,0,4,8,12])

        if ((i ==0)|(i==2)): 
            a.set_ylabel('Difference in % of storms in cluster', fontsize =17)
            
        plt.rcParams.update({'font.size': 20}) 
        hatches = ['','','','','','','','','','','///','**','','','','','','','','','','','///','**','','','','','','','','','','','///','**','','','','','','','','','','','///','**']  

        a.plot(range(-10,20,1), np.zeros(30), 'k', linewidth = 0.5)
        for i,thisbar in enumerate(bar2.patches):
            # Set a different hatch for each bar


            thisbar.set_edgecolor('white')
            thisbar.set_hatch(hatches[i])

        a.set_xlabel(" ")
        fmt = '%.0f%%' # Format you want the ticks, e.g. '40%'
        yticks = matplotlib.ticker.FormatStrFormatter(fmt)
        a.yaxis.set_major_formatter(yticks)
#     a.legend(loc = 'lower left', ncol = 1, fontsize = 'small')

    for tick in a.get_xticklabels():
        tick.set_rotation(90)
    a.set_xlabel(" ")
    plt.subplots_adjust(hspace = 0.07, wspace=0.11)    
    
    plt.show()
    fig.savefig('/tigress/gkortum/thesisSummer/figures/ensembles_bar_shift.jpg', dpi =300)    
    fig,a1 = plt.subplots(ncols = 1,nrows = 1,figsize=(9,6),dpi = 300)  
#     a1 = fig2.add_subplot(spec2[4])
    a1.yaxis.set_major_formatter(yticks)
    columns1 = ['Cluster 1','Cluster 2',"Cluster 3","Cluster 4"]
    hatches = ['','','','','','','','','','','//','*']    
    bar = dfClusteredBar.drop(columns = (columns1[0:num_clusters])).plot(kind='bar', ax = a1, color = colors, edgecolor = 'white', width = 0.9)
    for i,thisbar in enumerate(bar.patches):
        # Set a different hatch for each bar

        thisbar.set_color(colors[i])
        thisbar.set_edgecolor('white')
        try:
            thisbar.set_hatch(hatches[i])
        except: print(i)

    a1.plot(range(-10,10,1), np.zeros(20), 'k', linewidth = 0.5)
    a1.set_ylabel('% Change in storms per year')
    a1.get_legend().remove()
    a1.set_ylim(0,50)
    for tick in a1.get_xticklabels():
        tick.set_rotation(90)
    a1.set_xlabel(" ")
#     a1.set_xticks(['1','2','1','2','1','2','1'])
    plt.show()
    fig.savefig('/tigress/gkortum/thesisSummer/figures/ensembles_bar_shift_total.jpg', dpi =300)

    
    
    



    
    
    

