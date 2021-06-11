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


##written and composed by Sergei L Shevyrev http://lefa.geologov.net

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

file_model_data_name='model_data_{}{}'.format(pathrowfolder,datefolder);
file_model_data_name_path=os.path.join(dir_products_path,file_model_data_name);

file_model_data_name_prob='model_data_prob{}{}'.format(pathrowfolder,datefolder);
file_model_data_name_path_out=os.path.join(dir_products_path,file_model_data_name_prob);

model_filename=os.path.join('..','data','maxent_model_DPCA_OCSVM.p');

#chhosen due to eigenvalues and different signs
#goetite dpc2, kaolinite dpc2, chlorite dpc2, hematite dpc1 

#2 Загрузка данных для обучения
with open(file_model_data_name_path, 'rb') as f:
    model_data_df = pickle.load(f)
    
#getting resolution from file    
for file in os.listdir(dir_cropped_path):         #exclude 3band tif files
        if file.lower().endswith("."+fileext.lower()):
            gdal_object = gdal.Open(os.path.join(dir_cropped_path,file)); #as new gdal_object was created, no more ColMinInd,RowMinInd
            break;                          

row,col=gdal_object.GetRasterBand(1).ReadAsArray().shape;    
    
#3 Создание выборки для обучения модели MaxEnt
#find ore column
for key in model_data_df.columns:
    if key.lower().startswith("ore")==True:
        ore_key=key;
        break;
        
#ore_data=model_data_df[model_data_df[ore_key]==1]; #sub data frame for ore pixels only


#select DPC1 only indexes
dpc1_columns=[];
for col_name in [*model_data_df]:
   
               
    if col_name == 'DPC2_limonite' or col_name == 'DPC2_hematite'\
                or col_name == 'DPC2_quartz' or col_name == 'DPC2_muscovite'\
                or col_name == 'DPC2_kaolinite' or col_name == 'DPC2_chlorite':
                dpc1_columns.append(col_name);
    
          
#arrange columns in the necessary order into model_np_array
dpc1_columns=sorted(dpc1_columns, reverse=False);

#
all_data_np_array=np.array(np.array(model_data_df['id']).reshape(-1,1));
all_data_np_array=np.hstack([all_data_np_array,np.array(model_data_df['row']).reshape(-1,1),\
                             np.array(model_data_df['col']).reshape(-1,1)]);

for colname in dpc1_columns:
    col_np=np.array(model_data_df[colname]);
    all_data_np_array=np.hstack([all_data_np_array,col_np.reshape(-1,1)]);


#select ores only records
idx_ore=np.array(model_data_df[ore_key]==1);    
ore_data_np_array=all_data_np_array[idx_ore,:];  #select np array with data corresponding to ore only 

roc_auc_list=[];

for rs in range(41,42):
#rs=831;
    print('current random state is %s'%(rs));
    #divide into train and test matrix
    col_names=['id','row','col']+dpc1_columns; #concatenated lists
    params_train,params_test=train_test_split(ore_data_np_array[:,3:], test_size=0.2,random_state=rs); #,\
                                             # random_state=42);  #
    
    ###
    #teach OneSVM model
    
    # Standardize features
    mean = params_train[:,0:].mean(axis=0) #mean value for the each layer
                                            #
    std = params_train[:,0:].std(axis=0)   #
    train_cover_std = (params_train[:,0:] - mean) / std          #
                                                                #
                                                                #
    # Fit OneClassSVM
    print(" - fit OneClassSVM ... ", end='')
    clf = svm.OneClassSVM(nu=0.3, kernel="rbf", gamma=0.6)  #model регрессор svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.4) 

    clf.fit(train_cover_std)        #
                                    #

    all_data_std=(all_data_np_array[:,3:] - mean) / std;                         
    pred = clf.decision_function(all_data_std) #
    
    #
    pred_norm=(pred-pred.min())/(pred.max()-pred.min());
    
 
    
  
    non_ore_idx=model_data_df.index[model_data_df[ore_key] == 0].tolist()
    #bgrd_rand_idx=random.sample(non_ore_idx, int(len(non_ore_idx)/2));
    bgrd_rand_idx=random.sample(non_ore_idx, int(len(params_test[:,0])));
    pred_background = pred[bgrd_rand_idx];
    
    

    train_cover_std_test = (params_test[:,0:] - mean) / std
    
    all_data_std_train,all_data_std_test=train_test_split(all_data_std, test_size=0.01,random_state=42);
    
    pred_test = clf.decision_function(train_cover_std_test)
    
    
    scores = np.r_[pred_test, pred_background]
    y = np.r_[np.ones(pred_test.shape), np.zeros(pred_background.shape)]
    fpr, tpr, thresholds = metrics.roc_curve(y, scores,pos_label=1,drop_intermediate=True)
    roc_auc = metrics.auc(fpr, tpr)
    #plt.text(-35, -70, "AUC: %.3f" % roc_auc, ha="right")
    print("\n Area under the ROC curve : %f" % roc_auc)
    
    fig=plt.figure();
    plt.plot(fpr, tpr, label='ROC-curve');
    plt.plot([0,1],[0,1],'--')
    plt.xlabel('False positive rate');
    plt.ylabel('True positive rate');
    plt.title('Area under the ROC curve : {0:.{1}f}'.format(roc_auc,4));
    plt.legend()
    plt.savefig('ROC-curve_prediction_probability.png',dpi=300);
    plt.savefig('ROC-curve_prediction_probability.svg',dpi=300);
    plt.show();
    
    roc_auc_list.append(roc_auc);    
    max_iter=np.where(roc_auc_list==max(roc_auc_list))[0][0]; #step number for max ROC_AUC value
    print('Highest ROC-AUC computed so far: %s on iteration(RS)= %s'%(max(roc_auc_list),max_iter));
    
    if roc_auc>=0.7:
        print('ROC-AUC >= than 0.7 was reached on RS=%s. Breaking...'%(rs));
        break;
        
    print('Iteration was finished. Highest ROC-AUC is %s'%(max(roc_auc_list)));



#render predicted values into raster
predicted_raster=np.empty([row,col],dtype=np.float64);
predicted_raster[:]=np.nan;

for i in range(0,len(all_data_np_array[:,0]),1):
    r,c=all_data_np_array[i,1],all_data_np_array[i,2];
    predicted_raster[int(r),int(c)]=pred_norm[i];

#save geotiff with normalized predicted data
saveGeoTiff(predicted_raster,os.path.join(dir_products_path,\
    "predicted_oneClassSVM{}_".format('')+pathrowfolder+"_"+datefolder+".tif"),\
    gdal_object,0,0);

data_output_df=model_data_df.assign(Prob=pred);  
data_output_df=data_output_df.assign(Prob_norm=pred_norm);  

#вывод фрейма с данными в файл pickle (так как возможностей excel не хватит)
with open(file_model_data_name_path_out, 'wb') as f:
    pickle.dump(data_output_df, f)


#save model into file
with open(model_filename, 'wb') as f:
    pickle.dump(clf, f);  

#plot partial dependence plots
    
feature_names=dpc1_columns;
features = np.arange(0,len(feature_names),1)

fig, ax = plt.subplots(figsize=(12, 6))
#fig, ax = plt.subplots(figsize=(2, 2))
part_dep=plot_partial_dependence(clf, all_data_std_test,features,feature_names,n_cols=3,\
                                 grid_resolution=50,ax=ax)
fig = plt.gcf();
plt.setp(ax.get_yticklabels(), fontsize=3);
plt.setp(ax.get_xticklabels(), fontsize=3);
fig.suptitle('Response curves for the target area')
fig.subplots_adjust(hspace=0.7)
#fig.text(0.06, 0.5, 'common ylabel', ha='center', va='center', rotation='vertical')
fig.savefig('Partial_dependence_training.svg',dpi=300);
fig.savefig('Partial_dependence_training.png',dpi=300);    
    
    
#save model into file
with open(model_filename, 'wb') as f:
    pickle.dump(clf, f);    
