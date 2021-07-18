# DefMe scripts set

#some methods are from https://pcjericks.github.io/py-gdalogr-cookbook/vector_layers.html
#this script has been composed and written by Sergei L Shevirev http://lefa.geologov.net

import gdal,ogr #OpenGIS Simple Features Reference Implementation
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import time;
import copy;

from skimage.transform import resize #function for SRTM resize according to Landsat resolution
import elevation
import richdem as rd

from mygdal_functions0_9 import *
from configuration import *

"""
libraries version required:
    gdal 2.3.3 from conda install
    scikit-learn 0.23
    richdem 0.3.4 from pip install 
    elevation 1.1.3  pip install
"""
time_start=time.time()

#topocorrection flag
is_topocorrection=True; #defaut on 

#files for processing, input and output directory
# are taking from configuration.py

#no edits beyong that line ->

file_for_crop=[];
metadata=[];

try:
    for file in os.listdir(imgfilepath):
        #file=file.lower();
        if file.lower().endswith("."+fileext.lower()):
            file_for_crop.append(file);
            print(file+" was added to crop queue.");
        if file.lower().endswith("."+metafileext.lower()):
            print(file+" was defined as metadata file.");
            try:
                metadata=parse_MTL(os.path.join(imgfilepath,file));
            except:
                print('Unable to parse metadata from {}'.format(os.path.join(imgfilepath,file)));
except(FileNotFoundError):
        print("Input image folder doesn\'t exist...");

SunElevation=metadata['SUN_ELEVATION'];      
SunAzimuth=metadata['SUN_AZIMUTH']; 
SolarZenith=90-SunElevation;

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
    srtm_ext=GetExtent(srtm_gt,srtm_ysize,srtm_xsize); 
    #resolution in meters
    srtm_dpx=(srtm_ext[3][0]-srtm_ext[0][0])/srtm_xsize
    srtm_dpy=(srtm_ext[0][1]-srtm_ext[2][1])/srtm_ysize
    
    if check_shp_inside_raster(srtm_ext,shp_extent):
#        sampleSrtmImage,ColMinIndSRTM,RowMinIndSRTM =crop_by_shp(shp_extent,srtm_ext,\
#                                                    srtm_dpx,srtm_dpy,srtm_band_array); 
        srtm_band = rd.LoadGDAL(drm_filepath);

        slope = rd.TerrainAttribute(srtm_band, attrib='slope_degrees')
        aspect = rd.TerrainAttribute(srtm_band, attrib='aspect')
        
        if os.path.exists(os.path.join("..","tif"))==False:
            os.mkdir(os.path.join("..","tif"));
        
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
    
    #PARSING OF MTL FILE 
    #atmospheric correction (DOS1 - dark object subtraction)
    #https://semiautomaticclassificationmanual.readthedocs.io/en/latest/remote_sensing.html#dos1-correction
    #1) Compute radiance for 1% dark object
    if len(metadata)>0: #if metadata where red, haze removal should be applied
        try:
            band_number=int(myfile.split('.')[0].lower().split('_b')[1]);
        except:
            band_number=1;
            print('Band name is not understood, default band number (1) was taken...');
        #L_MIN_LAMBDA=np.iinfo(sampleImage.dtype).min;
        #L_MAX_LAMBDA=np.iinfo(sampleImage.dtype).max;
        L_MIN_LAMBDA=metadata['RADIANCE_MINIMUM_BAND_'+str(band_number)];
        L_MAX_LAMBDA=metadata['RADIANCE_MAXIMUM_BAND_'+str(band_number)];
        QCALMIN=metadata['QUANTIZE_CAL_MIN_BAND_'+str(band_number)]; #DN minimal value for band
        QCALMAX=metadata['QUANTIZE_CAL_MAX_BAND_'+str(band_number)]; #DN maximum value for band
        one_perc_sum=np.sum(sampleImage)*0.0001;
        
        for dn in range(int(QCALMIN),int(QCALMAX)):      #finding DNmin
            if np.sum(sampleImage[sampleImage<=dn])>=one_perc_sum: 
                DNmin=copy.copy(dn); #brightness for 1% dark object
                break;
        M_L=metadata['RADIANCE_MULT_BAND_'+str(band_number)]; #multiplicative rescaling factor, sensor gain (Richards, 2013)
        A_L=metadata['RADIANCE_ADD_BAND_'+str(band_number)];  #additive rescaling factor
        
        #0Spectral radiance on satellite sensor
        L_lambda= (M_L*sampleImage)+A_L;
        E_SUN=((np.pi*metadata['EARTH_SUN_DISTANCE']**2)*\
               metadata['RADIANCE_MAXIMUM_BAND_'+str(band_number)])/\
               metadata['REFLECTANCE_MAXIMUM_BAND_'+str(band_number)];#coefficient of solar extraatmospheric radiance
        
        
        #1 Path radiance,  this path radiance value is then subtracting from each pixel value in the image, DOS method http://gsp.humboldt.edu/OLM/Courses/GSP_216_Online/lesson4-1/radiometric.html
        L_P=(M_L*DNmin)+A_L-(0.01*E_SUN*np.cos(np.deg2rad(SolarZenith)))/(np.pi*metadata['EARTH_SUN_DISTANCE']**2);
        
        #2 Sensor radiance corrected by subtracting path radiance L_P
        #L_lambda=L_lambda-L_P;
        
        #Band digital numbers corrected to atmosperic effects
        dDN=L_P/M_L; #correction for image brightness
        sampleImage_correct=sampleImage-dDN; #so, we continue working with DN
        
        print('Band number:'+str(band_number));
        print('dDN:'+str(dDN));
        
                
    #topocorrection
    if is_topocorrection==True: #topocorrection flag    
        if was_corrected==False:
            
            #коррекция aspect по Landsat8
            #adjust srtm resolution to landsat8
            #[hlc,wlc]=np.shape(sampleImage);
            [hlc,wlc]=np.shape(sampleImage_correct);
            aspect_band_cropped=resize(aspect_cropped,(hlc,wlc),preserve_range=True,mode="wrap") #it works with scikit-image resize
    
            #коррекция slope по Landsat8
            slope_band_cropped=resize(slope_cropped,(hlc,wlc),preserve_range=True,mode="wrap") #it works with scikit-image resize
            
            
            Cos_i=np.cos(np.deg2rad(slope_band_cropped))*np.cos(np.deg2rad(SolarZenith))+\
            np.sin(np.deg2rad(slope_band_cropped))*np.sin(np.deg2rad(SolarZenith))*\
            np.cos(np.deg2rad(SunAzimuth-aspect_band_cropped));
            
            #ЭТОТ РАСЧЕТ КОС I РАССМАТРИВАЕТ ВСЕ СКЛОНЫ КАК ОСВЕЩЕННЫЕ ПОД ПРЯМЫМ УГЛОМ!                                                                            
            #Cos_i=np.cos(np.deg2rad(SolarZenith-slope_band_cropped));
                        
            #Do SCS+C correction anyway
           
            #(b,a)=np.polyfit(Cos_i.ravel(),sampleImage.ravel(),1);
            (b,a)=np.polyfit(Cos_i.ravel(),sampleImage_correct.ravel(),1);
            C=a/b;                                                                   
            was_corrected=True; #switch the flag to true                                                                                        
        
        print("Performing topographic correction.. Please, wait..")
        #Sun-Canopy-Sensor Correction (SCS)+C
        
        sampleImage_correct=np.float64(sampleImage_correct*\
                ((np.cos(np.deg2rad(SolarZenith))*np.cos(np.deg2rad(slope_band_cropped))+C)\
                 /(C+Cos_i)));
        
        
        #band_array=np.float32(L_lambda); 
        pic_show(sampleImage,"landsat initial ");
        hist_show(sampleImage);
        pic_show(sampleImage_correct,"landsat  corrected to atmosphere and topo");
        hist_show(sampleImage_correct);
        #band_array_out=copy.copy(sampleImage);   #reflectance                    
    else: #no topocorrection
        print("No topocorrection was selected, only DOS correction applied..")
            
    #drop image to the disk
    print("drop image to the disk")
    outfilename=os.path.join(dir_cropped_path,"crop_"+myfile.lower());
    if not os.path.isdir(dir_cropped_path):
        os.makedirs(dir_cropped_path) #create output directory if none
    try:
        saveGeoTiff(sampleImage_correct,outfilename,gdal_object,ColMinInd,RowMinInd) #save topocorrected Landsat crop
    except:
        print("Can not write on a disk... and/or error(s) in saveGeoTiff function")
          
        
#STEP 2. COMPUTE pseudocolor RGB stacks and satellite indexes
"""
Autodetect BANDs for default names, if the user has not specified names (for now, default names),
skip index or RGB stack if we don't find BAND NUMBER
"""
print("Step. 2. Getting names of the cropped files...")        
#getting names of the cropped files, aquire band names
file_for_processing=[];
try:
    for file in os.listdir(dir_cropped_path): #We collect files from the folder with cropped images
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
                gdal_object = gdal.Open(os.path.join(dir_cropped_path,myfile)) #as new gdal_object was created, no more ColMinInd,RowMinInd
                bands['band'+str(N)]=gdal_object.GetRasterBand(1).ReadAsArray() ;
            except:
                print("Error! Can not read cropped bands!")
print("Bands dictionary output:")
print(bands) 

#create RGB stacks:
#truecolor
truecolorRGB=image_stack(bands['band4'],bands['band3'],bands['band2'],do_norm8=1,do_show=1)  
#Комбинация 7-4-2. Изображение близкое к естественным цветам, позволяет анализировать состояние атмосферы и дым. Здоровая растительность выглядит ярко зеленой, ярко розовые участки детектируют открытую почву, коричневые и оранжевые тона характерны для разреженной растительности.
b742RGB=image_stack(bands['band7'],bands['band4'],bands['band2'],do_norm8=1,do_show=1)
#Комбинация 5-4-1. Изображение близкое к предыдущему, позволяет анализировать сельскохозяйственные культуры
b652RGB=image_stack(bands['band6'],bands['band5'],bands['band2'],do_norm8=1,do_show=1)
#Комбинация 4-5-3. Изображение позволяет четко различить границу между водой и сушей, с большой точностью будут детектироваться водные объекты внутри суши. Эта комбинация отображает растительность в различных оттенках и тонах коричневого, зеленого и оранжевого, дает возможность анализа влажности и полезны при изучении почв и растительного покрова.
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


#create cumulative image composite image of the hydroxyl image(red band), the iron oxide image
#(green band) and the average of these two images (blue band).

index_composite=image_stack(HA,IOA,(HA+IOA)/2,1,1)


#GENERAL OUTPUT
#print("Prepare to show PCA images")

#later incorporate path into functions
if not os.path.isdir(dir_products_path):
            os.makedirs(dir_products_path) #create output products directory if none
            
fig_save_cumsum_path=os.path.join(dir_products_path,"variance_cumsum.svg");
fig_save_pca_path=os.path.join(dir_products_path,"pca_comp.png");
tab_save_pca_variance=os.path.join(dir_products_path,"comp_variances.xls");

#num_comp=show_pca_cumsum(pca,fig_save_cumsum_path); #pca variance cumsum to determine right number of components
#show_pca_images(eigenvalues,mean_X,m,n,fig_save_pca_path) #show pca component images

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