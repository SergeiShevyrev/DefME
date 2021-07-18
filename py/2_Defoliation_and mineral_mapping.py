#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 15:57:43 2020

Defoliation script with mineral principal components computing.

Считается на основе спектральных каналов, прошедших топографическую коррекцию.
Вычисляются Direct Principal Components, на основе которых рассчитывается модель
оконтуривания месторождений
 
Вычисление компонентов и их нагрузок отсюда https://scentellegher.github.io/machine-learning/2020/01/27/pca-loadings-sklearn.html

"""

import gdal,ogr #OpenGIS Simple Features Reference Implementation
import numpy as np
from mygdal_functions0_9 import *
import matplotlib.pyplot as plt
import os
import pandas as pd
import time;
from sklearn.preprocessing import scale
from sklearn import decomposition
import copy;
from sklearn.cluster import KMeans
from matplotlib.colors import LinearSegmentedColormap
from sklearn.model_selection import train_test_split

from mygdal_functions0_9 import *
from configuration import *

#1 Settings

#files for processing, input and output directory

cov_ratios_dpca_save_iron=os.path.join(dir_products_path,"ratios_dpca_cov_stat_iron.xls");
cov_ratios_dpca_save_clay=os.path.join(dir_products_path,"ratios_dpca_cov_stat_clay.xls");

loadings_dpca_save_iron=os.path.join(dir_products_path,"loadings_stat_iron.xls");
loadings_dpca_save_clay=os.path.join(dir_products_path,"ratios_dpca_cov_stat_clay.xls");
loadings_filemask='loadings_DPCA_{}.xls';
loadings_filemask2='loadings_DPCA_2_{}.xls';
variance_filemask='variance_DPCA_{}.xls';

band_number_inname='_b%N%.' #%N% - for band number e.g. LC81050292016143LGN00_B6.TIF NOT A CASE SENSITIVE
band_number_inname=band_number_inname.lower();
ColMinInd=0; RowMinInd=0; #because we work on already cropped pictures

#list of bands to be used in computations
bands=[2,3,4,5,6,7];

DO_KMEANS=False; #if this flag set to FALSE image will be neglected (*-1), if TRUE
                #k-means classififed
DO_NORM_NEG_NORM=True; #нормализует данные 0 - 1, шкалирует 1 - 0 в случае 
                        #отрицательных нагрузок loadings 
NDVI_CLASS_MASK=0; #ndvi class to be excluded from computations -1 if unused


#bands to be used (Carranza, Hale, 2002)
#Alteration mineral to map	Band ratio images input to DPC (Landsat 8 OLI)

#Quartz	3/4, 7/1
#muscovite	 3/4, 6/7
#kaolinite	5/1, 7/4
#Chlorite	3/4, 7/5
#hematite 	5/4, 7/1
#limonite 3/4, 4/2


#2 data processing
files_for_processing=[];

try:
    for file in os.listdir(dir_cropped_path):         #exclude 3band tif files
        #file=file.lower();
        if file.lower().endswith("."+fileext.lower())  \
        and not file.lower().startswith("b") and not file.lower().startswith("true")\
        and not file.lower().startswith("cum"): 
            
            files_for_processing.append(file);
            print(file+" was added to data collecting queue.");
except(FileNotFoundError):
        print("Input image folder doesn\'t exist...");

#создание словаря каналов bands и загрузка туда файлов
bands={};  #dictionary storing band names 
for myfile in files_for_processing:
    for N in range(1,9):
        #populating bands dictionary
        if band_number_inname.replace('%n%',str(N),1) in myfile:
            try:
                gdal_object = gdal.Open(os.path.join(dir_cropped_path,myfile)) #as new gdal_object was created, no more ColMinInd,RowMinInd
                bands['band'+str(N)]=gdal_object.GetRasterBand(1).ReadAsArray() ;
            except:
                print("Error! Can not read cropped bands!")
#print("Bands dictionary output:")
#print(bands) 

#Compute J.Carranza band rations and their PCs
#mineral band ratios were updated according to Prof. Carranza letter of 2021/02/05

m,n=bands['band4'].shape;

#iron oxides
ratio67=bands['band6']/bands['band7']; #limonite      
ratio54=bands['band5']/bands['band4']; #vegetation 
#quartz
ratio61=bands['band6']/bands['band1']; #minerals          
ratio54=bands['band5']/bands['band4']; #vegetation
#muscovite
ratio64=bands['band6']/bands['band4']; #mineral    
ratio31=bands['band3']/bands['band1']; #vegetation 
#kaolinite 
ratio74=bands['band7']/bands['band4']; #kaolinite     
ratio31=bands['band3']/bands['band1']; #vegetation  
#chlorite 
ratio57=bands['band5']/bands['band7']; #minerals    
ratio34=bands['band3']/bands['band4']; #vegetation  
#hematite 
ratio41=bands['band4']/bands['band1']; #minerals          
ratio54=bands['band5']/bands['band4']; #vegetation          


#ndvi ratio for area classes
ndvi=(bands['band5']-bands['band4'])/(bands['band5']+bands['band4']);

#nvdi classes using K-means
#km = KMeans(n_clusters=4)  #7 classes
#km.fit(ndvi.flatten().reshape(-1,1))  #reshape(-1,1) - transpose of the single-dimension array
#km.predict(ndvi.flatten().reshape(-1,1))
#ndvi_classes = km.labels_.reshape(m,n);
ndvi_classes=kmeans_sort(ndvi,do_reverse=0,n_clusters=4);
             
#show k-maens classes ndvi


colors = [(0, 0, 1), (0, 0.5, 0.2), (0, 1, 0),(1, 0, 0)]  # R -> G -> B
n_bins = 4  # Discretizes the interpolation into bins
cmap_name = 'my_list'
cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins);

#tell the colorbar to tick at integers
im=plt.imshow(ndvi_classes,cmap=cm);
plt.colorbar(im,ticks=np.arange(np.min(ndvi_classes),np.max(ndvi_classes)+1))

plt.savefig('ndvi_classes_Kunashir.png',dpi=300);
plt.savefig('ndvi_classes_Kunashir.svg',dpi=300);
plt.show();

#save ndvi classes
saveGeoTiff(ndvi_classes,os.path.join(dir_products_path,"ndvi_classes{}_".format('')+pathrowfolder+"_"+datefolder+".tif"),gdal_object,ColMinInd,RowMinInd);

"""
NDVI classes (empirical description)
0 - water and ice
1 - rare vegetation
2 - dense vegetation
3- open soil
"""


#assessment of channel standart deviation, selecting channels with the higher deviation
minerals={}
vegetation={}


minerals.update({'quartz':ratio61});
vegetation.update({'quartz':ratio54});

minerals.update({'muscovite':ratio64});
vegetation.update({'muscovite':ratio31});

minerals.update({'kaolinite':ratio74});
vegetation.update({'kaolinite':ratio31});

minerals.update({'chlorite':ratio57});
vegetation.update({'chlorite':ratio34});

minerals.update({'hematite':ratio41});
vegetation.update({'hematite':ratio54});

minerals.update({'limonite':ratio67});
vegetation.update({'limonite':ratio54});

#compute leaves/minerals DPC's
#compute DPCA for minerals vs vegetation 5/4
#1 flatten image matrixes
print("Started to compute DPCA for the iron oxides and clay minerals...")
print("Flatten image matrix...")

mineral_vegetation_flat={};  #compute flattened matrix for all minerals
for key in [*minerals]:
    mineral_vegetation_flat.update({key:mat4pca((minerals[key],vegetation[key]))});     

#2 compute PCA for minerals
pca_models_minerals={};
pca_dpca_minerals={};

"""
mineral_vegetation_flat[key] - минерал и его вегетация в виде таблицы с двумя столбцами

NDVI_CLASS_MASK=0

mask_index=(ndvi_classes.flatten()!=NDVI_CLASS_MASK); #булевы индексы "не-вода"
table4pca=mineral_vegetation_flat[key][mask_index] #выбираем в таблицу для PCA только классы суши

"""

#selecting/masking non-water NDVI pixels 
for key in [*minerals]:
    pca_models_minerals.update({key:decomposition.PCA(n_components=2)});
    #updates of 2021/02/26
    mask_index=(ndvi_classes.flatten()!=NDVI_CLASS_MASK); #булевы индексы "не-вода"
    blank_table_pca=np.zeros(mineral_vegetation_flat[key].shape);
    table4pca=mineral_vegetation_flat[key][mask_index]
    found_pca=pca_models_minerals[key].fit_transform(table4pca); #table with pca inside    
    blank_table_pca[mask_index]=found_pca; #place computed DPCA into blank table, mask stays zero
    pca_dpca_minerals.update({key:blank_table_pca});
    

#3 loadings for minerals/vegetation (eigenvectors/eigenvalues)
                                            
                                           
loadings_minerals={};
for key in [*minerals]:
    #split 30% of data set for computation loadings
    #mineral_veg_train, mineral_veg_test = train_test_split(mineral_vegetation_flat[key],\
    #                                test_size=0.66, random_state=42);
    loadings = get_comp_loadings2(pca_models_minerals[key],\
        features=[key,'vegetation'],columns=['DPC1', 'DPC2']);
    print(loadings);                              
    loadings_minerals.update({key:loadings}); #    

#В ЭТОЙ ВЕРСИИ ИСПРАВЛЕНО ЗНАЧЕНИЕ МНОЖИТЕЛЕЙ ДЛЯ negation 
#4 DPCA for minerals
DPCA_minerals={};
for key in [*minerals]:
    pca_image=get_pca_image(pca_dpca_minerals[key],m,n,n_components=2);
    pca_image_classes={};
    for pckey in pca_image: #classify PCA images
        if loadings_minerals[key][('DPC'+pckey)][key]<0:
            do_reverse=1; #должно быть обусловлено знаками DPCA ЕСЛИ НАГРУЗКА ОТРИЦАТЕЛЬНА, МНОЖИМ на -1
            multiplier=-1;
        else:
            do_reverse=0;
            multiplier=1; #NO negation
        
        
        if DO_KMEANS==False:
            #no K-means classification!!!!!!!!!! 
            if DO_NORM_NEG_NORM==False:
                img_class=pca_image[pckey]*multiplier;
            else:
                #print('multiplier={}'.format(multiplier));
                if multiplier==1:
                    img_class=(pca_image[pckey]-np.min(pca_image[pckey]))/\
                        (np.max(pca_image[pckey])-np.min(pca_image[pckey]));
                    #print('img_class={}'.format(img_class));
                else: #multiplier==-1
                    img_class=(np.max(pca_image[pckey])-pca_image[pckey])/\
                        (np.max(pca_image[pckey])-np.min(pca_image[pckey]));
                    #print('img_class={}'.format(img_class));
        else:
            #uncomment line below if you wish k-means classification
            img_class=kmeans_sort(pca_image[pckey],do_reverse=do_reverse,n_clusters=10);
        
        #img_class=mean_shift_sort(pca_image[pckey],do_reverse=do_reverse);
        pca_image_classes.update({pckey:img_class});
        #pca_image_classes.update({pckey:get_image_classes(pca_image[pckey])});
    DPCA_minerals.update({key:pca_image_classes});    


#5 save loadings
for key in [*minerals]:
    loadings_minerals[key].to_excel(os.path.join(dir_products_path,\
                     loadings_filemask.format(key)),index=True);
    
                    
#6 save DPCs into geotiff and matplotlib image
for key in [*minerals]:
    for n in DPCA_minerals[key]: #save normalize values of DPCA matrix
        #img=normalize_matrix(DPCA_minerals[key][n]);
        #apply negation 
        img=DPCA_minerals[key][n];
        
        #if loadings_minerals[key][key]['DPC'+n]<0 and loadings_minerals[key]['vegetation']['DPC'+n]>0:
        #    print('negation for mineral %s, %s'%(key,'DPC'+n));
            #img=np.ones(img.shape)*np.max(img)-img;
        #    img=np.ones(img.shape)-img;
        
        #DPCA_minerals[key][n]=img; #replace minerals
        
        saveGeoTiff(img,os.path.join(dir_products_path,"DPC{}_{}_".format(n,key)+pathrowfolder+"_"+datefolder+".tif"),gdal_object,ColMinInd,RowMinInd);        


plt.figure();
plt.subplot(231);
plt.imshow(DPCA_minerals['limonite']['1'],cmap='gray');
plt.title('\'limonite\' image');
plt.axis('off');
plt.subplot(232);
plt.imshow(DPCA_minerals['quartz']['1'],cmap='gray');
plt.title('\'quartz\' image');
plt.axis('off');
plt.subplot(233);
plt.imshow(DPCA_minerals['hematite']['1'],cmap='gray');
plt.title('\'hematite\' image');
plt.axis('off');
plt.subplot(234);
plt.imshow(DPCA_minerals['muscovite']['1'],cmap='gray');
plt.title('\'muscovite\' image');
plt.axis('off');
plt.subplot(235);
plt.imshow(DPCA_minerals['kaolinite']['1'],cmap='gray');
plt.title('\'kaolinite\' image');
plt.axis('off');
plt.subplot(236);
plt.imshow(DPCA_minerals['chlorite']['1'],cmap='gray');
plt.title('\'chlorite\' image');
plt.axis('off');
plt.savefig('mineral_maps_DPC1.png',dpi=300);
plt.savefig('mineral_maps_DPC1.svg',dpi=300);
plt.show();

###

plt.figure();
plt.subplot(231);
plt.imshow(DPCA_minerals['limonite']['2'],cmap='gray');
plt.title('\'limonite\' image');
plt.axis('off');
plt.subplot(232);
plt.imshow(DPCA_minerals['quartz']['2'],cmap='gray');
plt.title('\'quartz\' image');
plt.axis('off');
plt.subplot(233);
plt.imshow(DPCA_minerals['hematite']['2'],cmap='gray');
plt.title('\'hematite\' image');
plt.axis('off');
plt.subplot(234);
plt.imshow(DPCA_minerals['muscovite']['2'],cmap='gray');
plt.title('\'muscovite\' image');
plt.axis('off');
plt.subplot(235);
plt.imshow(DPCA_minerals['kaolinite']['2'],cmap='gray');
plt.title('\'kaolinite\' image');
plt.axis('off');
plt.subplot(236);
plt.imshow(DPCA_minerals['chlorite']['2'],cmap='gray');
plt.title('\'chlorite\' image');
plt.axis('off');
plt.savefig('mineral_maps_DPC2.png',dpi=300);
plt.savefig('mineral_maps_DPC2.svg',dpi=300);
plt.show();

