# -*- coding: utf-8 -*-

#this script has been composed and written by Sergei L Shevirev http://lefa.geologov.net
#some methods are from https://pcjericks.github.io/py-gdalogr-cookbook/vector_layers.html

import gdal,ogr #OpenGIS Simple Features Reference Implementation
import numpy as np
from mygdal_functions0_9 import *
import matplotlib.pyplot as plt
import os
import pandas as pd
import time;

from skimage.transform import resize #function for SRTM resize according to Landsat

import elevation
import richdem as rd

time_start=time.time()

#some parameters for the topocorrection
is_topocorrection=True; #topocorrection flag
SunElevation=33.69564372      #28.411
SunAzimuth=162.48098752    #163.93


SolarZenith=90-SunElevation;


#files for processing, input and output directory
pathrowfolder="100_028"
datefolder="2023_10_16"  #

shapefilename='AOI.shp'; # sea area
imgfilepath=os.path.join("..","Landsat_8_OLI",pathrowfolder,datefolder); 
shpfilepath=os.path.join("..","shp",shapefilename);
#shpfilepath=os.path.join("..","shp","AOI_tmp"+".shp");
fileext="tif"; #extention for files
outdir=os.path.join("..","Landsat_8_OLI_Processed",pathrowfolder,datefolder);
dir_cropped="cropped_bands_topo" #dir for AOI cropped
dir_crop_path=os.path.join(outdir,dir_cropped);
dir_products="products_topo" 
dir_products_path=os.path.join(outdir,dir_products);
band_number_inname='_b%N%.' #%N% - for band number e.g. LC81050292016143LGN00_B6.TIF NOT A CASE SENSITIVE
band_number_inname=band_number_inname.lower();

#drm for topocorrection
drm_name="srtm_UTM.tif";
drm_folder=os.path.join("..","DRM","tiff");
drm_filepath=os.path.join(drm_folder,drm_name);


file_for_crop=[];

try:
    for file in os.listdir(imgfilepath):
        #file=file.lower();
        if file.lower().endswith("."+fileext.lower()):
            file_for_crop.append(file);
            print(file+" was added to crop queue.");
except(FileNotFoundError):
        print("Input image folder doesn\'t exist...");

#STEP 0. Prepare for the topocorrection

try:
    shp_extent=get_shp_extent(shpfilepath);
except:
    print("Can not read shp AOI file.")
            
#crop dem file
if is_topocorrection==True:
    print("Perform cropping of srtm");
    
    #read DEM geotiff
    srtm_gdal_object = gdal.Open(drm_filepath)
    srtm_band = srtm_gdal_object.GetRasterBand(1)
    srtm_band_array = srtm_band.ReadAsArray() 
    
    #get spatial resolution
    srtm_gt=srtm_gdal_object.GetGeoTransform()
    srtm_xsize = srtm_gdal_object.RasterXSize
    srtm_ysize = srtm_gdal_object.RasterYSize #x and y raster size in pixels
    srtm_ext=GetExtent(srtm_gt,srtm_ysize,srtm_xsize) #[[влx,влy],[нлx,нлy],[нпy, нпy],[впx, впy]]
    #resolution in meters
    srtm_dpx=(srtm_ext[3][0]-srtm_ext[0][0])/srtm_xsize
    srtm_dpy=(srtm_ext[0][1]-srtm_ext[2][1])/srtm_ysize
    
    if check_shp_inside_raster(srtm_ext,shp_extent):
#        sampleSrtmImage,ColMinIndSRTM,RowMinIndSRTM =crop_by_shp(shp_extent,srtm_ext,\
#                                                    srtm_dpx,srtm_dpy,srtm_band_array); 
        srtm_band = rd.LoadGDAL(drm_filepath);

        slope = rd.TerrainAttribute(srtm_band, attrib='slope_degrees')
        aspect = rd.TerrainAttribute(srtm_band, attrib='aspect')

        rd.SaveGDAL(os.path.join("..","tif","aspectInitialRes.tif"), aspect);
        rd.SaveGDAL(os.path.join("..","tif","SlopeInitialRes.tif"), slope);
    else:
        print("AOI shp file" +shpfilepath + "is not inside of DEM"+drm_filepath+". Exiting.");
        input('Press Enter for exit...')
        exit;    

    #reopening SRTM products
    #read srtm products
    aspect_gdal_object = gdal.Open(os.path.join("..","tif","aspectInitialRes.tif"))       #aspect
    aspect_band = aspect_gdal_object.GetRasterBand(1)
    aspect_band_array = aspect_band.ReadAsArray() 
    
    slope_gdal_object = gdal.Open(os.path.join("..","tif","SlopeInitialRes.tif"))        #slope
    slope_band = slope_gdal_object.GetRasterBand(1)
    slope_band_array = slope_band.ReadAsArray() 
    
    #get PRODUCTS spatial resolution
    srtm_gt,srtm_xsize,srtm_ysize,srtm_ext,srtm_dpx,srtm_dpy=getGeotiffParams(aspect_gdal_object);
        
    
    #check if SRTM products inside of SHP AOI ad crop it
    if check_shp_inside_raster(srtm_ext,shp_extent):
        #do image crop
        aspect_cropped,ColMinInd,RowMinInd =crop_by_shp(shp_extent,srtm_ext,srtm_dpx,srtm_dpy,aspect_band_array)
        slope_cropped,ColMinInd,RowMinInd =crop_by_shp(shp_extent,srtm_ext,srtm_dpx,srtm_dpy,slope_band_array)
        
        #for testing purporses 
        saveGeoTiff(slope_cropped,'test_crop_slope.tif',slope_gdal_object,ColMinInd,RowMinInd) #tryna save cropped geotiff
    
    else:
        print("SRTM is outside of the AOI, exiting...")
        exit();

was_corrected=False; #flag to check if resolution and scale were corrected to landsat8        
#STEP 1. CROP geotiffs one by one with AOI shape file
print("Step. 1. Starting geotiff crop operation...")        
for myfile in file_for_crop:
    
    if myfile.lower().endswith('_b8.tif'): #to skip 8th channel
        continue;
    
    #read geotiff
    gdal_object = gdal.Open(os.path.join(imgfilepath,myfile))
    band = gdal_object.GetRasterBand(1)
    band_array = band.ReadAsArray() 
    
    #get spatial resolution
    #do image crop
    gt,xsize,ysize,ext,dpx,dpy=getGeotiffParams(gdal_object)
 
    
    #check shp posiiton inside of tiff
    if check_shp_inside_raster(ext,shp_extent):
        #do image crop
        sampleImage,ColMinInd,RowMinInd =crop_by_shp(shp_extent,ext,dpx,dpy,band_array)
        
    else:
        print("AOI shp file" +shpfilepath + "is not inside of tiff"+myfile+". Exiting.");
        input('Press Enter for exit...')
        exit;
        
    #topocorrection
    if is_topocorrection==True: #topocorrection flag    
        if was_corrected==False:
            
            #коррекция aspect по Landsat8
            #adjust srtm resolution to landsat8
            [hlc,wlc]=np.shape(sampleImage);
            aspect_band_cropped=resize(aspect_cropped,(hlc,wlc),preserve_range=True,mode="wrap") #it works with scikit-image resize
    
            #коррекция slope по Landsat8
            slope_band_cropped=resize(slope_cropped,(hlc,wlc),preserve_range=True,mode="wrap") #it works with scikit-image resize
            
            
            Cos_i=np.cos(np.deg2rad(slope_band_cropped))*np.cos(np.deg2rad(SolarZenith))+\
            np.sin(np.deg2rad(slope_band_cropped))*np.sin(np.deg2rad(SolarZenith))*\
            np.cos(np.deg2rad(SunAzimuth-aspect_band_cropped));
            

            (b,a)=np.polyfit(Cos_i.ravel(),sampleImage.ravel(),1);
            C=a/b;                                                                   
            was_corrected=True; #switch the flag to true                                                                                        
        
        print("Performing topographic correction.. Please, wait..")
        #Sun-Canopy-Sensor Correction (SCS)+C
        band_array=np.uint16(sampleImage*\
                ((np.cos(np.deg2rad(SolarZenith))*np.cos(np.deg2rad(slope_band_cropped))+C)\
                 /(C+Cos_i)));
        pic_show(sampleImage,"landsat initial");
        hist_show(sampleImage);
        pic_show(band_array,"landsat SCS corrected");
        hist_show(band_array);                       
    else: #no topocorrection
        print("No topocorrection was selected..")
        #band_array=sampleImage+0; #no operation

            
    #drop image to the disk
    print("drop image to the disk")
    outfilename=os.path.join(dir_crop_path,"crop_"+myfile.lower());
    if not os.path.isdir(dir_crop_path):
        os.makedirs(dir_crop_path) #create output directory if none
    try:
        saveGeoTiff(band_array,outfilename,gdal_object,ColMinInd,RowMinInd) #save topocorrected Landsat crop
    except:
        print("Can not write on a disk... and/or error(s) in saveGeoTiff function")
          
    

        
#STEP 2. COMPUTE pseudocolor RGB stacks and satellite indexes

print("Step. 2. Getting names of the cropped files...")        
#getting names of the cropped files, aquire band names
file_for_processing=[];
try:
    for file in os.listdir(dir_crop_path): #набираем файлы из папки с кадрированными изображениями
        file=file.lower();
        if file.endswith("."+fileext.lower()):
            file_for_processing.append(file);
            print(file+" was added to crop queue.");
except(FileNotFoundError):
        print("Input image folder doesn\'t exist...");

bands={};  #dictionary storing band names 
for myfile in file_for_processing:
    for N in range(1,9):
        #populating bands dictionary
        if band_number_inname.replace('%n%',str(N),1) in myfile:
            try:
                gdal_object = gdal.Open(os.path.join(dir_crop_path,myfile)) #as new gdal_object was created, no more ColMinInd,RowMinInd
                bands['band'+str(N)]=gdal_object.GetRasterBand(1).ReadAsArray() ;
            except:
                print("Error! Can not read cropped bands!")
print("Bands dictionary output:")
print(bands) 

#create RGB stacks:
#truecolor
truecolorRGB=image_stack(bands['band4'],bands['band3'],bands['band2'],do_norm8=1,do_show=1)  
b742RGB=image_stack(bands['band7'],bands['band4'],bands['band2'],do_norm8=1,do_show=1)
b652RGB=image_stack(bands['band6'],bands['band5'],bands['band2'],do_norm8=1,do_show=1)
b453RGB=image_stack(bands['band4'],bands['band5'],bands['band3'],do_norm8=1,do_show=1)

#after Aydal, 2007
b642RGB=image_stack(bands['band6'],bands['band4'],bands['band2'],do_norm8=1,do_show=1)   
b765RGB=image_stack(bands['band7'],bands['band6'],bands['band5'],do_norm8=1,do_show=1)   
b764RGB=image_stack(bands['band7'],bands['band6'],bands['band4'],do_norm8=1,do_show=1)   


#create indexes
NDVI=(bands['band5']-bands['band4'])/(bands['band5']+bands['band4']) #NDVI
IOA=(bands['band4']/bands['band2']) #Iron oxide alteration [Doğan Aydal, 2007]
HA=(bands['band7']/bands['band2'])#Hydroxyl alteration [Doğan Aydal, 2007]
CM=(bands['band7']/bands['band6']) #Clay minerals [Doğan Aydal, 2007]


index_composite=image_stack(HA,IOA,(HA+IOA)/2,1,1)


#GENERAL OUTPUT
#print("Prepare to show PCA images")

#later incorporate path into functions
if not os.path.isdir(dir_products_path):
            os.makedirs(dir_products_path) #create output products directory if none
            
fig_save_cumsum_path=os.path.join(dir_products_path,"variance_cumsum.svg");
fig_save_pca_path=os.path.join(dir_products_path,"pca_comp.png");
tab_save_pca_variance=os.path.join(dir_products_path,"comp_variances.xls");

#COMPUTE Landsat and PCA stat for the CROSTA METHOD

stat_bands_save=os.path.join(dir_products_path,"bands_stat.xls");
cor_bands_save=os.path.join(dir_products_path,"bands_cor_stat.xls");
#cov_bands_pca_save=os.path.join(dir_products_path,"bands_pca_cov_stat.xls");

print("Saving band stat to {}".format(stat_bands_save));
save_landsat_bands_stat(bands,stat_bands_save);

print("Saving bands mutual correlation to {}".format(cor_bands_save));
save_landsat_mutual_cor(bands,cor_bands_save);


#save RGB's and index to the disk
print("Saving products on a disk")
if not os.path.isdir(dir_products_path):
    os.makedirs(dir_products_path) #create output directory if none
try:
    print("Saving RGBs...")
    ColMinInd=0; RowMinInd=0; #because we work on already cropped pictures
    saveGeoTiff(truecolorRGB,os.path.join(dir_products_path,"truecolorRGB"+pathrowfolder+"_"+datefolder+".tif"),gdal_object,ColMinInd,RowMinInd);     
    saveGeoTiff(b742RGB,os.path.join(dir_products_path,"b742RGB"+pathrowfolder+"_"+datefolder+".tif"),gdal_object,ColMinInd,RowMinInd);
    saveGeoTiff(b652RGB,os.path.join(dir_products_path,"b652RGB"+pathrowfolder+"_"+datefolder+".tif"),gdal_object,ColMinInd,RowMinInd);
    saveGeoTiff(b453RGB,os.path.join(dir_products_path,"b453RGB"+pathrowfolder+"_"+datefolder+".tif"),gdal_object,ColMinInd,RowMinInd);
     #Aydal pseudocolor:
    saveGeoTiff(b642RGB,os.path.join(dir_products_path,"b642RGB"+pathrowfolder+"_"+datefolder+".tif"),gdal_object,ColMinInd,RowMinInd);
    saveGeoTiff(b765RGB,os.path.join(dir_products_path,"b765RGB"+pathrowfolder+"_"+datefolder+".tif"),gdal_object,ColMinInd,RowMinInd);
    saveGeoTiff(b764RGB,os.path.join(dir_products_path,"b764RGB"+pathrowfolder+"_"+datefolder+".tif"),gdal_object,ColMinInd,RowMinInd);
    print("Saving Indexes...")
    saveGeoTiff(NDVI,os.path.join(dir_products_path,"NDVI"+pathrowfolder+"_"+datefolder+".tif"),gdal_object,ColMinInd,RowMinInd);
    saveGeoTiff(IOA,os.path.join(dir_products_path,"IOA"+pathrowfolder+"_"+datefolder+".tif"),gdal_object,ColMinInd,RowMinInd);
    saveGeoTiff(HA,os.path.join(dir_products_path,"HA"+pathrowfolder+"_"+datefolder+".tif"),gdal_object,ColMinInd,RowMinInd);
    saveGeoTiff(CM,os.path.join(dir_products_path,"CM"+pathrowfolder+"_"+datefolder+".tif"),gdal_object,ColMinInd,RowMinInd);
    saveGeoTiff(index_composite,os.path.join(dir_products_path,"CumulativeAlteration"+pathrowfolder+"_"+datefolder+".tif"),gdal_object,ColMinInd,RowMinInd);
 

    print("Products data were saved.")
except:
    print("Can not write PRODUCTS on a disk... and/or error(s) in saveGeoTiff function")

print("Operations were finished. It took {} sec".format(time.time()-time_start))