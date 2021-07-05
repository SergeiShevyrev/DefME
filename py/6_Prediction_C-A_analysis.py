#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
##written and composed by Sergei L Shevyrev http://lefa.geologov.net
File aimed to perform MPM classification according to fractal diagram
log(C) - log(A). 
Source: Yousefi, Carranza, 2015
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

#one class SVM
from sklearn import svm, metrics
#selecting training and testing set
from sklearn.model_selection import train_test_split
import random
from sklearn.inspection import plot_partial_dependence, partial_dependence
from scipy.signal import argrelextrema

#1 Settings

#files for processing, input and output directory
pathrowfolder="100_028"
datefolder="2023_10_16"
imgfilepath=os.path.join("..","Landsat_8_OLI",pathrowfolder,datefolder); 
fileext="tif"; #extention for files
outdir=os.path.join("..","Landsat_8_OLI_Processed",pathrowfolder,datefolder);

dir_products="products_topo" 
dir_cropped="cropped_bands_topo" 
dir_products_path=os.path.join(outdir,dir_products);
dir_cropped_path=os.path.join(outdir,dir_cropped);

class_boundaries_value_filename='cbv.values'

#step for log-transformed value bounding
step_value=0.005;
filter_extrema=1; #absolute threshold of first area derivative for extrema filtering
start_value_perc=0.98; #98% of values will be ignored for differentiation



#file name of the predicted ore probability 
predicted_file_path=os.path.join(dir_products_path,\
    "predicted_oneClassSVM{}_".format('')+pathrowfolder+"_"+datefolder+".tif");

                                 

#2 Opening of the predicted PROBABILITY file
MOPM_gdal_object = gdal.Open(predicted_file_path)
MOPM_array = MOPM_gdal_object.GetRasterBand(1).ReadAsArray() #rectanrular MOPM array
MOPM_array[np.isnan(MOPM_array)]=0; #replace nan values with 0

MOPM_array_flat=np.ndarray.flatten(MOPM_array[MOPM_array>0]);  #flat MOPM array for >0

#3 Compute log-transformed values
log_MOPM=np.log10(MOPM_array_flat);
log_MOPM.sort();

#iterate log_MOPM values, compute areas for them
quantity_log_MOMP=np.arange(np.min(log_MOPM),np.max(log_MOPM)+step_value,step_value);
start_value=int(start_value_perc*len(quantity_log_MOMP)); #start value for differentiation

area_log_array=np.array([]);
for i in quantity_log_MOMP:
    ind=np.where(log_MOPM>=i);
    area_log_array=np.append(area_log_array,np.log10(len(ind[0])));
        



#compute derivatives d_logArea/d_logMOPM
#first derivative
d_logArea=(area_log_array[1:]-area_log_array[0:-1])/step_value;

#second derivative
d2_logArea=(d_logArea[1:]-d_logArea[0:-1])/step_value;

#find peaks for 2nd derivatives
minima = np.array(argrelextrema(d2_logArea[start_value:], np.less));
maxima = np.array(argrelextrema(d2_logArea[start_value:], np.greater));

#extrema filtration by threshold values
min_ind=np.abs(d_logArea[start_value:][minima])>1
minima_filtered=minima[min_ind];
max_ind=np.abs(d_logArea[start_value:][maxima])>1
maxima_filtered=maxima[max_ind];

#MOPM class boundaries
class_boundary_ind=np.append(minima_filtered,maxima_filtered);
class_boundary_ind.sort();

#4 draw logarithmic C-A plot
plt.plot(quantity_log_MOMP[start_value:],area_log_array[start_value:]);
plt.title('C-A plot for values indexes starting start_value');
plt.xlabel('Log transform MOPM values');
plt.ylabel('Log(Area)');
plt.plot(quantity_log_MOMP[start_value:][class_boundary_ind],area_log_array[start_value:][class_boundary_ind], "xr");
plt.plot(quantity_log_MOMP[start_value:][class_boundary_ind],area_log_array[start_value:][class_boundary_ind], "xr");
for ind in class_boundary_ind:
    x=[quantity_log_MOMP[start_value:][ind],quantity_log_MOMP[start_value:][ind]];
    y=[np.nanmin(area_log_array[start_value:][area_log_array[start_value:] != -np.inf]),area_log_array[start_value:][ind]];
    plt.plot(x,y,'r--');
plt.savefig('C-A_plot.png',dpi=300);
plt.savefig('C-A_plot.svg',dpi=300);
plt.show(); 


plt.plot(quantity_log_MOMP[start_value:-1],d_logArea[start_value:]);
plt.title('first derivative');
plt.plot(quantity_log_MOMP[start_value:-2][minima_filtered],d_logArea[start_value:][minima_filtered], "xr");
plt.plot(quantity_log_MOMP[start_value:-2][maxima_filtered],d_logArea[start_value:][maxima_filtered], "o");
plt.show();


plt.plot(quantity_log_MOMP[start_value:-2],d2_logArea[start_value:]);
plt.title('second derivative');
plt.plot(quantity_log_MOMP[start_value:-2][minima_filtered],d2_logArea[start_value:][minima_filtered], "xr");
plt.plot(quantity_log_MOMP[start_value:-2][maxima_filtered],d2_logArea[start_value:][maxima_filtered], "o");
plt.show();

#find and report class boundary values
class_boundary_values_log=quantity_log_MOMP[start_value:][class_boundary_ind];
class_boundary_values=10**class_boundary_values_log;
print('Values of boundary class values:{}'.format(class_boundary_values));
#Values of boundary class values:[0.89369736 0.90404589 0.92510383 0.94665226 0.95761398 0.96870263]

#save class boundaries values into pickle file
with open(class_boundaries_value_filename, 'wb') as f:
    pickle.dump(class_boundary_values, f);  


# Map image output                                
plt.imshow(MOPM_array);
plt.colorbar();
plt.title('MOPM');
plt.show();                                 
                                 