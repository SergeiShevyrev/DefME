#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 14:19:23 2020

@author: geolog
"""

import gdal
import ogr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
import os #file system tools

from scipy.sparse import csr_matrix #https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html
from scipy.ndimage.morphology import binary_dilation
from skimage.morphology import dilation
from skimage.morphology import disk
from skimage.draw import circle

from mygdal_functions0_9 import *

#some solutions were taken from
##https://pcjericks.github.io/py-gdalogr-cookbook/vector_layers.html 

#0 Some functions
def objects_radius(ic,jc,h,w,rad,pxs,pys):      #ic,jc - локальные координаты точки пикс;
                                        #rad - max radius to point h,w - ширина, высота изображения
    #Пример: расчет Ed (евклидовых расстояний) для точки (ic,jc), pxs,pys - размеры пикселя
    rad_m=rad*np.sqrt(pxs**2+pys**2);
    Ed=np.zeros((h,w),dtype=np.float64);
    for i in range(0,h):
        for j in range(0,w):
            l=np.float64(np.sqrt(((i-ic)*pys)**2+((j-jc)*pxs)**2)); #расстояние до точки (ic,jc) от данной
            if (l<rad_m):
            #if (np.sqrt((i-ic)**2<(rad*pxs/pys))) and (np.sqrt((j-jc)**2<(rad*pys/pxs))):
                Ed[i,j]=1; #object is here
            else:
                Ed[i,j]=0;
    return Ed;

#draw objects
def draw_circle(img,r,c,rad):
    y,x=circle(r, c, rad, shape=None);
    img[y,x]=1;
    return img;

#1 Пути к исходным файлам
pathrowfolder="100_028"
datefolder="2023_10_16"
dir_products="products_topo" ;  
dir_cropped="cropped_bands_topo";  
aoi_shp_filepath=os.path.join("..","shp","AOI.shp");

points_shp_filepath=os.path.join("..","data","Shp",\
            "pnts_only.shp");

sample_folder_path= os.path.join("..","Landsat_8_OLI_Processed",pathrowfolder,datefolder,dir_cropped); #to get resolution from
outdir=os.path.join("..","Landsat_8_OLI_Processed",pathrowfolder,datefolder);
raster_path=os.path.join("..","Landsat_8_OLI_Processed",pathrowfolder,datefolder,dir_products,\
                         "OreContour_{}{}.tif".format(pathrowfolder,\
                                              datefolder));
fileext="tif"; #extention for files

radius=300/30; #радиус объекта в пикселях - ROC-AUC 0.605138
ColMinInd=0; RowMinInd=0; #because we work on already cropped pictures

#2 AOI shp
try:
    aoi_ext=get_shp_extent(aoi_shp_filepath); #x_min, x_max, y_min, y_max;
    print(aoi_ext)
except:
    print("Can not read shp AOI file.")
    
#3 Чтение Points shp
try:
    pnts_ext=get_shp_extent(points_shp_filepath);
    print(pnts_ext)
except:
    print("Can not read shp Points file.")

#3.5 Getting pixels resolution
try:
    for file in os.listdir(sample_folder_path):         #exclude 3band tif files
        if file.lower().endswith("."+fileext.lower()): 
           gdal_object = gdal.Open(os.path.join(sample_folder_path,file)) #as new gdal_object was created, no more ColMinInd,RowMinInd
           bands1=gdal_object.GetRasterBand(1).ReadAsArray() ; 
           
           #pixel_x_size,pixel_y_size=GetResolutionMeters(gdal_object);  #неверное разрешени
           pixel_x_size,pixel_y_size=gdal_object.GetGeoTransform()[1],gdal_object.GetGeoTransform()[5];
           print('x res=%.4f, y res=%.4f'%(pixel_x_size,pixel_y_size));
           break;
           
except(FileNotFoundError):
        print("Input image folder doesn\'t exist...");  
       
   

    
#4 Чтение объектов из точечного слоя
driver = ogr.GetDriverByName('ESRI Shapefile')

dataSource = driver.Open(points_shp_filepath, 0) # 0 means read-only. 1 means writeable.
#количество объектов в слое
layer = dataSource.GetLayer()
projection=layer.GetSpatialRef();
featureCount = layer.GetFeatureCount()
print ("Number of features in " + str(featureCount))

#5 Select points inside AOI
points_inside=list(); #список точек внутри экстента
for feature in layer:
     #shp_ext (ЛX,ПХ,НY,ВY)
    geom=feature.GetGeometryRef();
    mx,my=geom.GetX(), geom.GetY()  #coord in map units
    print("point on the map:")
    print(mx,my);
    if (mx<=aoi_ext[1]) and (mx>=aoi_ext[0]) and (my>=aoi_ext[2]) and (my<=aoi_ext[3]):
        points_inside.append((mx,my));
#convert global coordinates to local
lpntx,lpnty=list(),list(); #
for point in points_inside:
    lpntx.append(point[0]-aoi_ext[0]); #
    lpnty.append(aoi_ext[3]-point[1]);  #


#graphic output
plt.plot(lpntx,lpnty,'o')
plt.show()

#нахождение разрешения 
x_res = int((aoi_ext[1] - aoi_ext[0]) / abs(pixel_x_size));
y_res = int((aoi_ext[3] - aoi_ext[2]) / abs(pixel_y_size));



#6 
#making sparse matrix

pntx=np.int32(np.array(lpntx)/abs(pixel_x_size));
pnty=np.int32(np.array(lpnty)/abs(pixel_y_size));
pntval=np.ones(pnty.size);


obj_raster=np.zeros([y_res-1, x_res-1]);

for r,c in zip(pnty,pntx):
    try:
        obj_raster=draw_circle(obj_raster,r,c,radius);
    except IndexError:
        print('Index error, possibly marginal position of point?');


#7 Output Geotiff

obj_raster[np.isnan(obj_raster)==True]=65536
obj_raster=np.uint16(obj_raster)


#based on SHP (works incorrectly)
driver = gdal.GetDriverByName("GTiff")
outdata = driver.Create(raster_path, x_res, y_res, 1, gdal.GDT_UInt16)
outdata.SetGeoTransform((aoi_ext[0], pixel_x_size, 0, aoi_ext[3], 0, pixel_y_size));
outdata.SetProjection(projection.ExportToWkt())##sets same projection as input
outdata.GetRasterBand(1).WriteArray(obj_raster);


outdata.GetRasterBand(1).SetNoDataValue(0);
outdata.FlushCache();



#8 Демонстрация изображения
plt.imshow(obj_raster)
plt.colorbar()
plt.show()

plt.imshow(bands1)
plt.colorbar()
plt.show()