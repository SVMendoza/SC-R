#######################################################################
#######################################################################
#
#			DEPENDENCIAS
#
#######################################################################
#######################################################################
library(randomForest)
library(e1071)  
library(xgboost)
#library(neuralnet)
#library(nnet)
#library(deepNN)
library(torch)

library(data.table)
library(parallel)
library(snow)
library(future.apply)

library(rgdal)
library(terra)
library(compiler)

#######################################################################
#######################################################################
#
#		llamar todas las funciones
#
#######################################################################
#######################################################################

  FuncionesLocal<-'/home/sergio/clasificacion/'
  source(paste0(FuncionesLocal, 'PrepararDatos.r'))
  source(paste0(FuncionesLocal, 'propuestaModels.r')) 
  source(paste0(FuncionesLocal, 'otrasFuncionalidades.R'))
  source(paste0(FuncionesLocal, 'predicciones.R'))
  
#######################################################################
#######################################################################
#
#			CLASIFICACIÓN
#
#######################################################################
#######################################################################
classiFunction<-cmpfun(function(name.shape, file.img, name.CLASES, OPEN, SAVE, 
                                n.core, propVal, nsize, sel.n, ndt, 
                                dt.balance=FALSE, conv=FALSE, PC=FALSE,
                                selModel=c('rf', 'svm_Poy', 'svm_rad', 'xgboots','nn', 'all', 'none'),
                                epochs=NULL, batch=NULL, gpu=FALSE, block_size, out_channels, kernel_size, stride, padding,Normalize=TRUE){
 
 
 FuncionesLocal<-strsplit(source.dir,'ClassificModels.R')
 #FuncionesLocal<-"/home/sergio/clasificacion/"
  
  # Directorio temporal
  temp_dir <- file.path(tempdir(), "MyDIRPrinc/")
  dir.create(temp_dir, showWarnings = FALSE)
  on.exit(unlink(temp_dir, recursive = TRUE), add = TRUE)
  
  
 
#  if(isTRUE(Conv)) { 
  
 # }
  
 # if(isTRUE(conv)){
  
  
 # }
 
 
  
  #DAT.todos<-extract.dat(file.img, name.shape, name.CLASES,OPEN,SAVE, n.core, Normalize=TRUE)
  D.All<-d.val(propVal, extract.dat(file.img, name.shape, name.CLASES,OPEN,SAVE=temp_dir, n.core, Normalize=Normalize))  
  
  itrain<-f.train(D.All$d.train, name.CLASES, sel.n, ndt, n.core)  
  
  if(dt.balance) {
    itrain<-f.trainBal(data, name.CLASES, ndt, nsize, n.core)
  } else {
    itrain<-f.train(D.All$d.train, name.CLASES, sel.n, ndt, n.core)
  }
  
  nmodel<-length(itrain)
  DAT.test<-D.All$d.Val  
  
  if(toupper(selModel)=='ALL')
  {
    if(is.null(epochs)) num_epochs<-100 else num_epochs<-epochs
    if(is.null(batch)) batch_size<-1000 else batch_size<-batch
    mod<-AllModels()
    
  } else if(toupper(selModel)=='RF'){
    
    mod<-FunctionRF(itrain, 
                    name.CLASES, SAVE,DAT.test, OPEN, 
                    file.img, FuncionesLocal)
                    
  } else if(toupper(selModel)=='SVM_POLY'){
    
    mod<-FunctionPoly(itrain, name.CLASES, SAVE,DAT.test, OPEN, 
                             file.img, FuncionesLocal)
                             
  }else if(toupper(selModel)=='SVM_RAD'){
    
    mod<-FunctionRadIncrement(itrain, name.CLASES, SAVE,DAT.test, OPEN,
                     file.img, FuncionesLocal)
  } else if(toupper(selModel)=='XGBOOTS'){
    
    mod<-FunctionXGB(itrain, name.CLASES, 
                     SAVE,DAT.test, OPEN, file.img, FuncionesLocal)
                     
  } else if(toupper(selModel)=='NN'){
    
    if(is.null(epochs)) num_epochs<-100 else num_epochs<-epochs
    if(is.null(batch)) batch_size<-1000 else batch_size<-batch
    
    mod<-nnNet.fun(itrain, name.CLASES, 
    SAVE,DAT.test, OPEN, file.img, num_epochs, batch_size, FuncionesLocal)
    
  } else {
    mod<-D.All
  }
  mod  
})


AllModels<-cmpfun(function(){
  mRF<-FunctionRF(itrain, 
                  name.CLASES, SAVE,DAT.test, OPEN, 
                  file.img, FuncionesLocal)
  mPoly<-FunctionPoly(itrain, name.CLASES, SAVE,DAT.test, OPEN, 
                      file.img, FuncionesLocal) 
  mRad<-FunctionRad(itrain, name.CLASES, SAVE,DAT.test, OPEN,
                    file.img, FuncionesLocal)
  mXG<-FunctionXGB(itrain, name.CLASES, 
                   SAVE,DAT.test, OPEN, file.img, FuncionesLocal)
  #Dnn<-nnNet.fun(itrain, name.CLASES, SAVE,DAT.test, OPEN, 
  #               file.img, num_epochs, batch_size, FuncionesLocal)
  list(mRF, mPoly, mRad, mXG)
})

###############################################################


##		Convolucion funcion global


##############################################################

