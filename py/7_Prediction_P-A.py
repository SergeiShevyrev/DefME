#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 15:57:16 2021

@author: geolog
"""

import gdal,ogr #OpenGIS Simple Features Reference Implementation
import numpy as np
from mygdal_functions0_9 import *
import matplotlib.pyplot as plt
import os
import pandas as pd
import time;
from mygdal_functions0_9 import *
from sklearn.preprocessing import scale
from sklearn import decomposition
import copy;
from sklearn.cluster import KMeans
from matplotlib.colors import LinearSegmentedColormap
from scipy.sparse import csr_matrix 
import pickle

###
import numpy as np
"""
Give, two x,y curves this gives intersection points,
autor: Sukhbinder
5 April 2017
Based on: http://uk.mathworks.com/matlabcentral/fileexchange/11837-fast-and-robust-curve-intersections
"""


def _rect_inter_inner(x1, x2):
    n1 = x1.shape[0]-1
    n2 = x2.shape[0]-1
    X1 = np.c_[x1[:-1], x1[1:]]
    X2 = np.c_[x2[:-1], x2[1:]]
    S1 = np.tile(X1.min(axis=1), (n2, 1)).T
    S2 = np.tile(X2.max(axis=1), (n1, 1))
    S3 = np.tile(X1.max(axis=1), (n2, 1)).T
    S4 = np.tile(X2.min(axis=1), (n1, 1))
    return S1, S2, S3, S4


def _rectangle_intersection_(x1, y1, x2, y2):
    S1, S2, S3, S4 = _rect_inter_inner(x1, x2)
    S5, S6, S7, S8 = _rect_inter_inner(y1, y2)

    C1 = np.less_equal(S1, S2)
    C2 = np.greater_equal(S3, S4)
    C3 = np.less_equal(S5, S6)
    C4 = np.greater_equal(S7, S8)

    ii, jj = np.nonzero(C1 & C2 & C3 & C4)
    return ii, jj


def intersection(x1, y1, x2, y2):
    """
    INTERSECTIONS Intersections of curves.
   Computes the (x,y) locations where two curves intersect.  The curves
   can be broken with NaNs or have vertical segments.
   usage:
    x,y=intersection(x1,y1,x2,y2)
    Example:
    a, b = 1, 2
    phi = np.linspace(3, 10, 100)
    x1 = a*phi - b*np.sin(phi)
    y1 = a - b*np.cos(phi)
    x2=phi
    y2=np.sin(phi)+2
    x,y=intersection(x1,y1,x2,y2)
    plt.plot(x1,y1,c='r')
    plt.plot(x2,y2,c='g')
    plt.plot(x,y,'*k')
    plt.show()
    """
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    y1 = np.asarray(y1)
    y2 = np.asarray(y2)

    ii, jj = _rectangle_intersection_(x1, y1, x2, y2)
    n = len(ii)

    dxy1 = np.diff(np.c_[x1, y1], axis=0)
    dxy2 = np.diff(np.c_[x2, y2], axis=0)

    T = np.zeros((4, n))
    AA = np.zeros((4, 4, n))
    AA[0:2, 2, :] = -1
    AA[2:4, 3, :] = -1
    AA[0::2, 0, :] = dxy1[ii, :].T
    AA[1::2, 1, :] = dxy2[jj, :].T

    BB = np.zeros((4, n))
    BB[0, :] = -x1[ii].ravel()
    BB[1, :] = -x2[jj].ravel()
    BB[2, :] = -y1[ii].ravel()
    BB[3, :] = -y2[jj].ravel()

    for i in range(n):
        try:
            T[:, i] = np.linalg.solve(AA[:, :, i], BB[:, i])
        except:
            T[:, i] = np.Inf

    in_range = (T[0, :] >= 0) & (T[1, :] >= 0) & (
        T[0, :] <= 1) & (T[1, :] <= 1)

    xy0 = T[2:, in_range]
    xy0 = xy0.T
    return xy0[:, 0], xy0[:, 1]

###

#1 Settings

#files for processing, input and output directory
pathrowfolder="111_028"
datefolder="2019_10_16"
imgfilepath=os.path.join("..","Landsat_8_OLI",pathrowfolder,datefolder); 
fileext="tif"; #extention for files
outdir=os.path.join("..","Landsat_8_OLI_Processed",pathrowfolder,datefolder);

dir_products="products_topo" 
dir_cropped="cropped_bands_topo" 
dir_products_path=os.path.join(outdir,dir_products);
dir_cropped_path=os.path.join(outdir,dir_cropped);

#aoi_shp_filepath=os.path.join("..","shp","105_029_small.shp"); #105_029_small_NO_SEA.shp
#points_shp_filepath=os.path.join("..","shp","Points_Ore_UTM.shp");

#we use rasterized points for P-A plotting, cause these data were used for teaching
ore_raster_path=os.path.join("..","Landsat_8_OLI_Processed",pathrowfolder,datefolder,dir_products,\
                         "OreContour_{}{}.tif".format(pathrowfolder,\
                                              datefolder));

predicted_file_path=os.path.join(dir_products_path,\
    "predicted_oneClassSVM{}_".format('')+pathrowfolder+"_"+datefolder+".tif");  
                                 
ndvi_file_path=os.path.join(dir_products_path,\
    "ndvi_classes_{}".format('')+pathrowfolder+"_"+datefolder+".tif");                                   
                                 
selects_NDVI_classes=[1,2]; #open soils and rare vegetation                         
class_boundaries_value_filename='cbv.values'

#2 Load class boundary values
with open(class_boundaries_value_filename, 'rb') as f:
    class_boundaries_values = pickle.load(f);
#round and insert 0 value in front of array
#class_boundaries_values=np.insert(np.round(class_boundaries_values,2),0,0);
#class_boundaries_values=np.append(class_boundaries_values,1);  
class_boundaries_values=np.arange(0,1.1,0.1);  
    
#3 Open geotiff data
#prospectivity    
MOPM_gdal_object = gdal.Open(predicted_file_path)
MOPM_array = MOPM_gdal_object.GetRasterBand(1).ReadAsArray() #rectanrular MOPM array
MOPM_array[np.isnan(MOPM_array)]=0; #replace nan values with 0       
    
#ore objects  
ORE_gdal_object = gdal.Open(ore_raster_path)
ORE_array = ORE_gdal_object.GetRasterBand(1).ReadAsArray() #rectanrular MOPM array
ORE_array[np.isnan(ORE_array)]=0; #replace nan values with 0  

#ndvi mask read 
NDVI_classes_gdal_object = gdal.Open(ndvi_file_path)
NDVI_classes_array = NDVI_classes_gdal_object.GetRasterBand(1).ReadAsArray() #rectanrular MOPM array

#create mask
rect_ind_learn=(np.zeros(np.shape(NDVI_classes_array))==1) #create bool matrix of False
for cl in selects_NDVI_classes:
    rect_ind_learn=rect_ind_learn | (NDVI_classes_array==cl); 

#set 0 for non-learning classes,  apply NDVI mask on data
ORE_array[~rect_ind_learn]=0;
MOPM_array[~rect_ind_learn]=0;

#output rasters 
fig, (ax1, ax2,ax3) = plt.subplots(1,3);
ax1.imshow(MOPM_array); ax1.set_title('Prospectivity');
ax2.imshow(ORE_array); ax2.set_title('Points');
ax3.imshow(NDVI_classes_array); ax3.set_title('NDVI');
plt.show();

#4 iterating through class_boundaries_values and defind % of deposits pixels and % study area
total_ore_pixels=np.sum(ORE_array); 

#total_area_pixels=np.size(ORE_array); #for the total area
total_area_pixels=np.sum(rect_ind_learn); #for the only training area

deposits_percent=np.array([]); #perc of deposits
area_percent=np.array([]); # (1 - (perc_of_area))*100

for cl in class_boundaries_values:
    ind=MOPM_array>=cl;
    #
    mineral_points=100*np.sum(ORE_array[ind])/total_ore_pixels;       
    deposits_percent=np.append(deposits_percent,mineral_points);
    #
    ap=(1-(np.sum(MOPM_array>=cl)/total_area_pixels))*100
    area_percent=np.append(area_percent,ap);

#5 draw plots of area/prediction rate
#start of drawing plots
xticks_values=np.arange(0,1,0.1);
yticks_reversed=np.arange(100,-20,-20);    
x, y = intersection(class_boundaries_values, deposits_percent, class_boundaries_values, area_percent)
print(x,y);    
textbox='(%0.2f,%0.2f)'%(x,y);
#fig=plt.figure();
fig, ax1 = plt.subplots();
plt.plot(class_boundaries_values,deposits_percent,'r-',label='Prediction rate');
plt.plot(class_boundaries_values,area_percent,'g-',label='Area');
plt.text(x[0]-0.1, y[0]+7, textbox, bbox=dict(facecolor='white', alpha=0.1));
plt.plot([x[0],x[0]],[0,y[0]],'r--');
plt.plot([0,x[0]],[y[0],y[0]],'r--');
plt.ylim([-1, 102]);
plt.xlim([class_boundaries_values[1]-0.1, 1]);
plt.plot(x,y,'bo');
ax1.tick_params(axis=u'both', which=u'both',length=0)

plt.legend();
plt.grid();
plt.ylabel('Percentage of known mineralization points (%)');

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
#plt.xticks(class_boundaries_values);
plt.xticks(xticks_values);
#plt.yticks(yticks_reversed);
plt.xlabel('MOMP (prospectivity) value');
plt.ylabel('Percentage of study area (%)');
plt.ylim([-1, 102]);
ax2.invert_yaxis()  # labels read top-to-bottom
ax2.tick_params(axis=u'both', which=u'both',length=0)

plt.savefig('P-A_plot_Kunashir.svg',dpi=300);
plt.savefig('P-A_plot_Kunashir.png',dpi=300);
plt.show();
#end of drawing plots

    
#print report 
print(('Значение MOMP≥{:.2f} выделяет {:.2f}% известных объектов\
       на {:.2f}% территории').format(x[0],y[0],100-y[0]));

#Значение MOMP≥0.87 выделяет 89.06% известных объектов       на 10.94% территории       