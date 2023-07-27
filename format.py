
#import libraries
import matplotlib
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy as cartopy
from matplotlib import colors



#colors
colorList = []
for color in matplotlib.colors.CSS4_COLORS:
    colorList.append(color)
colorList2 = ['mediumpurple','hotpink','yellowgreen','tomato','aqua','navy']
colorList3= ['plum','midnightblue','seagreen','tomato','aqua','hotpink']




#custom colormaps
plum=matplotlib.colors.LinearSegmentedColormap.from_list('plum',['white','plum'], N=256, gamma=1.0)
midnightblue=matplotlib.colors.LinearSegmentedColormap.from_list('midnightblue', ['white','midnightblue'], N=256, gamma=1.0)
seagreen=matplotlib.colors.LinearSegmentedColormap.from_list('seagreen', ['white','seagreen'], N=256, gamma=1.0)
tomato=matplotlib.colors.LinearSegmentedColormap.from_list('tomato', ['white','tomato'], N=256, gamma=1.0)


# custom colormaps with transparency
def cmap(color):
    c = colors.to_rgba(color)
    cdict = {'red':   [(0.0,  1.0, 1.0),

                       (1.0,  c[0], c[0])],

             'green': [(0.0,  1.0, 1.0),
               
                       (1.0,  c[1], c[1])],

             'blue':  [(0.0,  1.0, 1.0),
 
                       (1.0,  c[2], c[2])],
             'alpha':  [(0.0,  0.0, 0.0),
                         (0.1,  1.0, 1.0),
                        (0.25,  1.0, 1.0),
                       (0.5,  1.0, 1.0),
                       (1.0,  1.0, 1.0)],
                }
    

    theCmap= matplotlib.colors.LinearSegmentedColormap(color, cdict, N=256, gamma=1.0)
    return theCmap
    
plumT = cmap('plum')
midnightblueT = cmap('midnightblue')
seagreenT = cmap('seagreen')
tomatoT = cmap('tomato')



# custom colormaps with more transparency 
def cmapT(color):
    c = colors.to_rgba(color)
    cdict = {'red':   [(0.0,  1.0, 1.0),

                       (1.0,  c[0], c[0])],

             'green': [(0.0,  1.0, 1.0),
               
                       (1.0,  c[1], c[1])],

             'blue':  [(0.0,  1.0, 1.0),
 
                       (1.0,  c[2], c[2])],
             'alpha':  [(0.0,  0.0, 0.0),
                
                       (1.0,  1.0, 1.0)],
                }
    

    theCmap= matplotlib.colors.LinearSegmentedColormap(color, cdict, N=256, gamma=1.0)
    return theCmap
plumTT = cmapT('plum')
midnightblueTT = cmapT('midnightblue')
seagreenTT = cmapT('seagreen')
tomatoTT = cmapT('tomato')
darkTT = cmapT('darkslategrey')
magentaTT = cmapT('mediumvioletred')


#another transparent colormap 
def cmap2(color):
    c = colors.to_rgba(color)
    cdict = {'red':   [(0.0,  (3*1.0+c[0])/4,(3*1.0+c[0])/4),

                       (1.0,  c[0], c[0])],

             'green': [(0.0,  (3*1.0+c[1])/4, (3*1.0+c[1])/4),
               
                       (1.0, c[1],c[1])],

             'blue':  [(0.0,  (3*1.0+c[2])/4, (3*1.0+c[2])/4),
 
                       (1.0,  c[2],c[2])],
             'alpha':  [(0.0,  0.0, 0.0),
                
                       (1.0,  1.0, 1.0)],
                }
    

    theCmap= matplotlib.colors.LinearSegmentedColormap(color, cdict, N=256, gamma=1.0)
    return theCmap


blackT = cmap2('k')


#map formatting
def mapsetup(central=0,minlat=0,maxlat=70,minlon=-110,maxlon=10, big = False):
    if big: fig=plt.figure(figsize=(15,20))
    else:
        fig=plt.figure(figsize=(5,7))
    plt.rcParams.update({'font.size': 18})
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=central), xlim = (-60,60),ylim = (-60,60))
    ax.set_extent([minlon,maxlon,minlat,maxlat], crs=ccrs.PlateCarree())
    ax.add_feature(cartopy.feature.OCEAN, facecolor='aliceblue')
    ax.add_feature(cartopy.feature.LAND, facecolor= 'ivory',edgecolor='lightblue')
    ax.add_feature(cartopy.feature.LAKES, facecolor='aliceblue',edgecolor='lightblue')
    ax.add_feature(cartopy.feature.RIVERS, edgecolor ='aliceblue')
    ax.set_xticks([-110, -90, -60,-30,0,10], crs = ccrs.PlateCarree())
    ax.set_yticks([0,15,30,45,60,70], crs = ccrs.PlateCarree())
 
    lon_formatter = LongitudeFormatter()
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    
    
    return ax


