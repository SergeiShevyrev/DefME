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

"""
written and composed by Sergei L Shevyrev http://lefa.geologov.net

Thanks to:
    
sparse to numpy https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.toarray.html            
numpy to sparse matrix https://stackoverflow.com/questions/7922487/how-to-transform-numpy-matrix-or-array-to-scipy-sparse-matrix            
"""


#1 Settings

#files for processing, input and output directory
pathrowfolder="100_028"
datefolder="2019_10_16"
imgfilepath=os.path.join("..","Landsat_8_OLI",pathrowfolder,datefolder); 
fileext="tif"; #extention for files
outdir=os.path.join("..","Landsat_8_OLI_Processed",pathrowfolder,datefolder);

dir_products="products_topo" 
dir_products_path=os.path.join(outdir,dir_products);

file_model_data_name='model_data_{}{}'.format(pathrowfolder,datefolder);
file_model_data_name_path=os.path.join(dir_products_path,file_model_data_name);

cov_ratios_dpca_save_iron=os.path.join(dir_products_path,"ratios_dpca_cov_stat_iron.xls");
cov_ratios_dpca_save_clay=os.path.join(dir_products_path,"ratios_dpca_cov_stat_clay.xls");

loadings_filemask='loadings_DPCA_{}.xls';

#select all besides water
#selects_NDVI_classes=[1,2,3]; #select open soils (1) and rare vegetation (2)
selects_NDVI_classes=[1,2];
prefix=["dpc1","dpc2"]; #only two values are allowed


#2 Data processing NDVI
files_for_processing=[];

try:
    for file in os.listdir(dir_products_path):         #exclude 3band tif files
        #file=file.lower();
        if file.lower().endswith("."+fileext.lower())  \
        and (file.lower().startswith(prefix[0]) or file.lower().startswith(prefix[1])\
             or file.lower().startswith("ndvi_c") or \
        file.lower().startswith("orecontour")):  #and ('goetite' in file.lower())==False

            files_for_processing.append(file);
            print(file+" was added to data collecting queue.");

           
except(FileNotFoundError):
        print("Input image folder doesn\'t exist...");

#3 Open files
raster_data={};  #dictionary storing dpc names and raster values 
for myfile in files_for_processing:
        try:
            key=myfile.split('_')[0]+'_'+myfile.split('_')[1]; #crop strings before second underline
            gdal_object = gdal.Open(os.path.join(dir_products_path,myfile)) #as new gdal_object was created, no more ColMinInd,RowMinInd
            raster_data.update({key:gdal_object.GetRasterBand(1).ReadAsArray()});
        except:
            print("Error! Can not read DPC/NDVI data!")

#4 data dictionary            

#rect_ind_learn=(raster_data['ndvi_classes']==selects_NDVI_classes[0]) | (raster_data['ndvi_classes']==selects_NDVI_classes[1]) #| - поэлементное "или" & - поэлементное "и"
rect_ind_learn=(np.zeros(np.shape(raster_data['ndvi_classes']))==1) #create bool matrix
for cl in selects_NDVI_classes:
    rect_ind_learn=rect_ind_learn | (raster_data['ndvi_classes']==cl);


#координаты точек для обучения
ri,ci=np.where(rect_ind_learn==True);       #ri[:,None] - single row transposition

#
plt.figure();
plt.imshow(rect_ind_learn); 
plt.title('Area for the model teaching (1,2,3 NDVI classes)');
plt.show();

model_data={};

#adding entry for the id/row/columns indexes
model_data.update({'id':[]});
model_data.update({'row':[]});
model_data.update({'col':[]});

#add dictionaries to model data according to opened rasters
for keyval in [*raster_data]:
    model_data.update({keyval:[]});

#обходим rc,ci попарно, выбираем данные в словарь базы данных
id=0;
for rn,cn in zip(ri,ci):
    #print(rn,cn);
    model_data['id'].append(id);
    model_data['row'].append(rn);
    model_data['col'].append(cn);
    id+=1;
    #picking points from every satellite image 
    for keyval in [*raster_data]:
        model_data[keyval].append(raster_data[keyval][rn,cn]);
        
       
#конвертируем model_data в DataFrame pamdas
model_data_df=pd.DataFrame(model_data);
    

#вывод фрейма с данными в файл pickle (так как возможностей excel не хватит)
with open(file_model_data_name_path, 'wb') as f:
    pickle.dump(model_data_df, f)
        
    
#show ore points 
plt.imshow(raster_data[keyval]);
plt.show();
    
        
