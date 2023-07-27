
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

from joblib import Parallel, delayed
import multiprocessing
import time
import pickle

from format import *
from restructure_cluster_datasets import *
from analyze_clusters import *

def read_netcdfs(files, dim,transform_func=None, transform_func2=None):
    def process_one_path(path,i):
        # use a context manager, to ensure the file gets closed after use
        with xr.open_dataset(path) as ds:
            # transform_func should do some sort of selection or
            # aggregation
            if i ==1:
                ds = transform_func(ds)
            elif i ==2:
                ds = transform_func2(ds)
            # load all data from the transformed dataset, to ensure we can
            # use it after closing each original file
            ds.load()
            return ds

    paths = files
    datasets = [process_one_path(paths,2),process_one_path(paths,1)]
    return xr.concat(datasets, dim)

    
def sendtonc(e,year):
    if year ==2021: 
        exp = 'amipHadISSTlong_chancorr_tigercpu_intelmpi_18_1080PE_extend2021'
    elif year ==2020: 
        exp = 'amipHadISSTlong_chancorr_tigercpu_intelmpi_18_1080PE_extend2020'
    else: 
        exp = 'amipHadISSTlong_chancorr_tigercpu_intelmpi_18_1080PE'
    number = e+1
    if number ==10: 
        file = '/tigress/wenchang/MODEL_OUT/AM2.5C360/'+exp+'/en'+str(number)+'/POSTP/'+str(int(year))+'0101.atmos_month.nc'
    else: 
        file = '/tigress/wenchang/MODEL_OUT/AM2.5C360/'+exp+'/en0'+str(number)+'/POSTP/'+str(int(year))+'0101.atmos_month.nc'
   
    combined = read_netcdfs(files=file, dim='grid_xt',
                            transform_func=lambda ds: ds.slp[:,360:720,960:1440],
                            transform_func2=lambda ds: ds.slp[:,360:720,0:240])
    return combined



def save_SLP_data():

    ensemblesSLP=[]
    for e in range(11): 
        if e ==10: 
            e = 9
        datasetsSLP=[]
        for i in range(1970,2022,1):

            SLPds= sendtonc(e,i)


            ygrad = np.true_divide(SLPds.differentiate('grid_yt'), 111000)
            ygrad = ygrad.rolling(grid_yt= 40, 
                                  min_periods = 2,
                                  center = True).mean().dropna("grid_yt").rolling(grid_xt= 40,
                                     min_periods = 2,center = True).mean().dropna("grid_xt")

            xgrad = np.true_divide(SLPds.differentiate('grid_xt'),np.cos(np.radians(np.asarray(SLPds['grid_yt']))).reshape([1,360,1])*111000)
            xgrad = xgrad.rolling(grid_xt= 40, 
                                  min_periods = 2,
                                  center = True).mean().dropna("grid_xt").rolling(grid_yt= 40,
                                     min_periods = 2,center = True).mean().dropna("grid_yt")
            DS=xr.Dataset({'SLP': SLPds, 'SLPxgrad':xgrad, 'SLPygrad': ygrad})

            datasetsSLP.append(DS)
            print(str(e) +' and '+str(i))
        ensemblesSLP.append(datasetsSLP)

    with open('/tigress/gkortum/thesisSummer/data/ensemblesSLP', 'wb') as f:
        pickle.dump(ensemblesSLP,f)
        
