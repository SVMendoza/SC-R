## SC-R: a supervised classification methodology

Maps of land use, carbon levels, forest degradation, or pests damage are crucial for climate change adaptation and restoration strategies for territorial planning. These maps derived from earth observation data help both local farmers and governments make informed management decisions. However, they are often built using unbalanced data and/or pseudo-replicates (signatures), which can create problems related to the poor performance of supervised classification algorithms. 

![myimage](../SVMendoza/SC-R/blob/main/Process.png)



SC-R is presented, a processing system designed to address supervised image classification problems from satellites or drones using machine learning techniques. SC-R integrates a set of strategies aimed at maximizing the performance of learning algorithms, considering the commonly used correlated data for training and computational limitations, with the goal of generating reliable results.


SC-R allows the user to choose between training a single machine learning model or using multiple models, as well as deciding whether to combine the predictions obtained through an ensemble result. This ensemble can be performed using a meta model approach (Stacking) or using votes. The pipeline performe a metrics measure to evaluate performe (F1-score). The models available in SC-R include Random Forest, Support Vector Machine (SVM) with polynomial kernel, SVM with sigmoid kernel, XGBoost, and a neural network.


The hyperparameters of the different algorithms are fixed, allowing for optimization of computational time and facilitating the reproducibility of the models. However, it is possible that for certain datasets, the models may have poor performance and difficulty generalizing. This system is simple but efficient, allowing good results to be obtained even with small datasets and good computational efficiency. Model ensembling allows for the generation of a final consensus map that incorporates the best features of individual models and provides performance metrics based on a confusion matrix.


The pipeline offers an image preprocessing tool using convolution (a typical Deep Learning process), using filters of 16, 32, 64, or 124 channels. If you choose convolution, it is recommended to apply a subsequent dimensionality reduction using Principal Component Analysis (PCA). This technique seeks to simplify the feature space, eliminate noise, and improve the computational efficiency of learning algorithms. If you decide to use a convolutional neural network (CNN), dimensionality reduction may not be advisable due to the high computational cost involved in using a CNN. In this case, PCA techniques could be applied to reduce the number of bands before image convolution. PCA could also be applied in the first stage of convolution if the user is training a deep network.



How the pipeline works: The main approach is a model assembly method, where machine learning algorithms are trained separately using one or more randomly selected observations from each polygon. This process is repeated *n* times, generating multiple models, which allows for better generalization and helps avoid overfitting by introducing variability into the training dataset. The predictive power of each model is evaluated using cross-validation, with the observations that were not used in training (i.e., a model is trained and the unused data is used for validation). In the case of Random Forest, the top 10 models are selected and assembled into a single Random Forest.


SC-R is a developed in **R** and is structured in modules using a functional approach. This structure facilitates its development and allows for constant updating and improvement. Currently, SC-R only supports data in Shapefile format, specifically with polygon geometry that is projected in the same way as the image to be classified.

### Functionality

A main function **classiFunction** is presented, allows the application of the model assembly methodology. The user must define some arguments. Along with additional functions that can be used separately or prior to using classiFunction.


**PrinCompMap** &nbsp;&nbsp;&nbsp;Allows the application of dimensionality reduction analysis using principal components.


**ConvolMap** &nbsp;&nbsp;&nbsp;Allows the application of convolution to the image. The user defines some arguments and whether they want the result to be after applying dimensionality reduction.


**DeepMap** &nbsp;&nbsp;&nbsp;Under testing, a general deep network algorithm for classification. The network has three convolutional layers passing through 16, 32, and 64 feature maps (112 characteristics maps). 


**Usage**

classiFunction(name.shape, file.img, name.CLASES, OPEN, SAVE, n.core, propVal, nsize=NULL, sel.n, ndt, dt.balance,=FALSE, Normalize=TRUE, selModel=c(), epochs=NULL, batch=NULL, gpu=NULL)


**Arguments**

name.shape 		&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; name of the shapefile with the **.shp** extension. 

file.img 		&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; name of the image to be classified with the .tif extension. 

OPEN 			&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; working directory where name.shape and file.img are stored. 

name.CLASES 		&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; name of the response/classification variable that identifies the class in the shapefile attribute table. 

SAVE 			&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; directory where all processing generated by the system will be saved. 

n.core	 		&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; number of cores (processors) to use, by default NULL (uses all available cores). 

propVal 		&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; by default, is NULL, defines the number of observations to use for model validation. 

Nsize 			&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; number of models to fit; in the case of XGBoost, only a single model is fitted. 

ndt 			&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; only for Random Forest. By default, is 100, and represents the number of trees per model to build. 

sel.n 			&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; number of observations per polygon to be selected. **: Directory where the R scripts are located.

dt.balance 		&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; is a Boolean value, default is FALSE. If TRUE, the algorithm forces the data subsets to be balanced by category. 

Normalize 		&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; is a Boolean value, default is TRUE, which allows the image to be normalized to a scale between 0 and 1.

selModel 		&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; selects the model(s) that should be trained and used for classification. 

epochs 			&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; is NULL by default, and is only for adjusting a neural network. 

batch 			&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; is NULL by default, and is only for adjusting a neural network.

gpu 			&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; is NULL by default. If you are running a neural network and have an NDVIA GPU, you can set it to TRUE. Otherwise, the network will be trained using the CPU.


# Required packages for modeling and data processing:

```plaintext
install.packages(c('randomForest', 'e1071', 'xgboost', 'data.table', 'future', , 'snow', 'terra', 'compiler'))

# Package required for neural networks and convolution operations:
# `torch` requires CUDA for GPU acceleration. Please make sure CUDA is installed and configured properly for GPU usage.
install.packages('torch')
```



```plaintext
# NOT RUN

rm(list=ls()): Removes all objects in the R environment.
gc(): Calls garbage collection to free memory.

source.dir<-'/home/packages/ClassificModels.R' # Directory where the R scripts are located. Facilitates platform use. 
source(source.dir) 

name.shape<-'clean.shp'
file.img<-'mm.tif'
OPEN<-'/home/classify/'
name.CLASES<-'ley'
SAVE<-'/home/classify/output/'
n.core<-8
propVal<-0.40
nsize<-200
ndt<-100 #nmodles only randomForest
sel.n<-1

classiFunction(name.shape, file.img, name.CLASES, OPEN, SAVE, 
                                n.core, propVal, nsize, sel.n, ndt, 
                                dt.balance=FALSE, Normalize=TRUE, 
                                selModel= 'rf',
                                epochs=NULL, batch=NULL)
```



## Apply convolution to the image.


Img
block_size
out_channels
device
kernel_size
stride
padding


```plaintext
# NOT RUN

rm(list=ls()): Removes all objects in the R environment.
gc(): Calls garbage collection to free memory.

source.dir<-'/home/packages/ClassificModels.R' # Directory where the R scripts are located. Facilitates platform use. 
source(source.dir) 

name.shape<-'clean.shp'
file.img<-'mm.tif'
OPEN<-'/home/classify/'
name.CLASES<-'ley'
SAVE<-'/home/classify/output/'
n.core<-8
propVal<-0.40
nsize<-200
ndt<-100 #nmodles only randomForest
sel.n<-1

classiFunction(name.shape, file.img, name.CLASES, OPEN, SAVE, 
                                n.core, propVal, nsize, sel.n, ndt, 
                                dt.balance=FALSE, Normalize=TRUE, 
                                selModel= 'rf',
                                epochs=NULL, batch=NULL)
```
