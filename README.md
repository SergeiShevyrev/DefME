DefMe is a set of Python scripts that provides users and developers with tools for atmospheric and topographic correction of spaceborne images, synthesis of indexes (products), implementing vector points of in-situ observation data, configuring and training of MaxEnt model and computing raster of target variable distribution. Scripts are organized accordingly to stages of research and purposes of each stage (Table 1). General description of the script set is given in Section 2.1. Subsequent details of software functions are presented in Section 2.2.

Description of files in the script set:

Number	File name	Stage of research	Description
1	0_Batch_crop_geotiff_folder.py	Preprocessing	Parsing of files in the directory, batch processing, including cropping to AOI, atmospheric and topographic corrections, computing of RGB composite images. 
2	1_Generate_ore_points_from_shp.py	Preprocessing	Creating of raster tiff layer from vector points of target variable observation.
3	2_Defoliation_and_mineral_mapping.py	Defoliation	Implementing of software defoliant technique, computing of directed principal components.  
4	3_Model_CollectingData.py	Dataset building	Composing of dataset for further manipulation.
5	4_MaxEnt_model_teaching.py	Model training	Configuring and training of MaxEnt model, rendering target variable distribution layer, model export into file.
6	5_MaxEnt_model_predicting.py	Model deployment
(optional)	Model loading from the saved file, its deployment to the new area. Requires previously collected dataset.
7	5_Prediction_C-A_analysis.py	Classification	Classification of target variable output, determining of class boundaries. 
8	6_Prediction_P-A.py	Assessment 	Assessment of model efficacy, drawing of Prediction-Area curve
9	mygdal_functions0_9.py	-	User defined functions for spatial data operation 
10	configuration.py	-	Storing analysis configuration variables

