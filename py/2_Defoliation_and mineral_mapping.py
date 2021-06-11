#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 15:57:43 2020

Defoliation script with mineral principal components computing.

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
from sklearn.model_selection import train_test_split

#1 Settings

#files for processing, input and output directory
pathrowfolder="100_027"
datefolder="2023_10_16"
imgfilepath=os.path.join("..","Landsat_8_OLI",pathrowfolder,datefolder); 
fileext="tif"; #extention for files
outdir=os.path.join("..","Landsat_8_OLI_Processed",pathrowfolder,datefolder);
dir_cropped="cropped_bands_topo" #dir for AOI cropped
dir_crop_path=os.path.join(outdir,dir_cropped);

dir_products="products_topo" 
dir_products_path=os.path.join(outdir,dir_products);

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



#2 Обработка данных
#создание списка файлов для открытия
files_for_processing=[];

try:
    for file in os.listdir(dir_crop_path):         #exclude 3band tif files
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
                gdal_object = gdal.Open(os.path.join(dir_crop_path,myfile)) #as new gdal_object was created, no more ColMinInd,RowMinInd
                bands['band'+str(N)]=gdal_object.GetRasterBand(1).ReadAsArray() ;
            except:
                print("Error! Can not read cropped bands!")

m,n=bands['band4'].shape;

#iron oxides
ratio57=bands['band5']/bands['band7']; #goethite          
ratio54=bands['band5']/bands['band4']; #vegetation         
#quartz
ratio51=bands['band5']/bands['band1']; #minerals           
ratio54=bands['band5']/bands['band4']; #vegetation
#alunite
ratio61=bands['band6']/bands['band1']; #mineral   
ratio54=bands['band5']/bands['band4']; #vegetation  
#illite 
ratio53=bands['band5']/bands['band3']; #illite             
ratio54=bands['band5']/bands['band4']; #vegetation
#chlorite 
ratio62=bands['band6']/bands['band2']; #minerals             
ratio54=bands['band5']/bands['band4']; #vegetation
#epidote 
ratio72=bands['band7']/bands['band2']; #minerals            
ratio54=bands['band5']/bands['band4']; #vegetation    


#ndvi ratio for area classes
ndvi=(bands['band5']-bands['band4'])/(bands['band5']+bands['band4']);

#nvdi classes using K-means
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

#assessment of channel standart deviation, selecting channels with the higher deviation
minerals={}
vegetation={}


minerals.update({'quartz':ratio51});
vegetation.update({'quartz':ratio54});

minerals.update({'alunite':ratio61});
vegetation.update({'alunite':ratio54});

minerals.update({'illite':ratio53});
vegetation.update({'illite':ratio54});

minerals.update({'chlorite':ratio62});
vegetation.update({'chlorite':ratio54});

minerals.update({'epidote':ratio72});
vegetation.update({'epidote':ratio54});

minerals.update({'goethite':ratio57});
vegetation.update({'goethite':ratio54});

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
plt.imshow(DPCA_minerals['goethite']['1'],cmap='gray');
plt.title('\'goethite\' image');
plt.axis('off');
plt.subplot(232);
plt.imshow(DPCA_minerals['quartz']['1'],cmap='gray');
plt.title('\'quartz\' image');
plt.axis('off');
plt.subplot(233);
plt.imshow(DPCA_minerals['epidote']['1'],cmap='gray');
plt.title('\'epidote\' image');
plt.axis('off');
plt.subplot(234);
plt.imshow(DPCA_minerals['alunite']['1'],cmap='gray');
plt.title('\'alunite\' image');
plt.axis('off');
plt.subplot(235);
plt.imshow(DPCA_minerals['illite']['1'],cmap='gray');
plt.title('\'illite\' image');
plt.axis('off');
plt.subplot(236);
plt.imshow(DPCA_minerals['chlorite']['1'],cmap='gray');
plt.title('\'chlorite\' image');
plt.axis('off');
plt.savefig('mineral_maps_Iturup1.png',dpi=300);
plt.savefig('mineral_maps_Iturup1.svg',dpi=300);
plt.show();

###

plt.figure();
plt.subplot(231);
plt.imshow(DPCA_minerals['goethite']['2'],cmap='gray');
plt.title('\'goethite\' image');
plt.axis('off');
plt.subplot(232);
plt.imshow(DPCA_minerals['quartz']['2'],cmap='gray');
plt.title('\'quartz\' image');
plt.axis('off');
plt.subplot(233);
plt.imshow(DPCA_minerals['epidote']['2'],cmap='gray');
plt.title('\'epidote\' image');
plt.axis('off');
plt.subplot(234);
plt.imshow(DPCA_minerals['alunite']['2'],cmap='gray');
plt.title('\'alunite\' image');
plt.axis('off');
plt.subplot(235);
plt.imshow(DPCA_minerals['illite']['2'],cmap='gray');
plt.title('\'illite\' image');
plt.axis('off');
plt.subplot(236);
plt.imshow(DPCA_minerals['chlorite']['2'],cmap='gray');
plt.title('\'chlorite\' image');
plt.axis('off');
plt.savefig('mineral_maps_Iturup2.png',dpi=300);
plt.savefig('mineral_maps_Iturup2.svg',dpi=300);
plt.show();


