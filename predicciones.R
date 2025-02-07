######################################################################
#                        ---------------------
                          ## Predicciones ##
#                        ---------------------
######################################################################
Predictmodel<-cmpfun(function(model, file, SAVE, n.core,type,...){

  dimRast<-dim(file)[1]*dim(file)[2]/8000000
  
  print("Starting prediction on the raster...")
  
  if(dimRast>1.4){
    block_size<-2000
    
    temp_dir <- file.path(tempdir(), "MyDIRPrinc/")
    dir.create(temp_dir, showWarnings = FALSE)
    on.exit(unlink(temp_dir, recursive = TRUE), add = TRUE)
    
    print(temp_dir)
    
    cutImage(file, block_size, SAVE=temp_dir) 
    rm('file')
    imgPred<-PredictMore1raster(model,n.core, temp_dir, Var.class, OR.labels,device, Yresp)
    
  } else 
    {
    
    imgPred<-Predict1raster(model,n.core, Var.class, OR.labels,device, Yresp)
    
  }
  writeRaster(imgPred, file=paste0(SAVE, 'rasterPredict.tif'), overwrite=TRUE)
})

######################################################################
#---------------------
  ## predicion cortando el raster ##
#  ---------------------
  ######################################################################
PredictMore1raster<-cmpfun(function(model,n.core, temp_dir, Var.class, OR.labels,device, Yresp){
  L<-list.files(temp_dir)
  #print(L)
  if(toupper(type)=='RF')
  {
    
    for(LL in L) {
      file0<-rast(paste0(temp_dir, LL))
     Predmodel.RF(model, file0, n.core, SAVE=temp_dir, namRast=LL)
      }
      
     L<-list.files(temp_dir, pattern='pred_')
    sal0<-list()
    for(LL in L)  sal0[[LL]]<-rast(paste0(temp_dir, LL)); gc()
    sal<-MosaicImg(sal0)
  }
else if(toupper(type)=='SVM_RAD' | toupper(type)=='SVM_POLY')
  {
  sal0<-list()
  for(LL in L) {
    file0<-rast(paste0(temp_dir, LL))
    sal0[[LL]]<-Predmodel.SVM(model, file0, n.core)
  }
  sal<-MosaicImg(sal0)
 
}
else if(toupper(type)=='XGBOOTS')
  {
  sal0<-list()
  for(LL in L) {
    file0<-rast(paste0(temp_dir, LL))
    sal0[[LL]]<-Predmodel.XGBoots(model, file0, n.core, Var.class, OR.labels)
  }
  sal<-MosaicImg(sal0)
  } else if(toupper(type)=='NN')
    { 
    sal0<-list()
    for(LL in L) {
      file0<-rast(paste0(temp_dir, LL))
      sal0[[LL]]<-PredmodelNN(model, file0, n.core, device, Yresp)
    }
    sal<-MosaicImg(sal0)
 } else{
  sal<-list()
  
  sal0<-list()
  sal1<-list()
  sal2<-list()
  sal3<-list()
  sal4<-list()
  sal5<-list()
  for(LL in L) {
    file0<-rast(paste0(temp_dir, LL))
    sal0[[LL]]<-Predmodel.RF(model[[1]], file0, n.core)
    sal1[[LL]]<-Predmodel.SVM(model[[2]], file0, n.core)
    sal2[[LL]]<-Predmodel.SVM(model[[3]], file0, n.core)
    sal3[[LL]]<-Predmodel.XGBoots(model[[4]], file0, n.core, Var.class, OR.labels)
    sal4[[LL]]<-PredmodelNN(model[[5]], file0, n.core, device, Yresp)
    }
  sal[['rf']]<-MosaicImg(sal0)
  sal[['svm_poly']]<-MosaicImg(sal1)
  sal[['svm_Rad']]<-MosaicImg(sal2)
  sal[['xgboots']]<-MosaicImg(sal3)
  sal[['deepNN']]<-MosaicImg(sal4)
  sal<-do.call(c, sal)
}
 sal 
})

######################################################################
#---------------------
  ## predicion solo un raster ##
# ---------------------
  ######################################################################
Predict1raster<-cmpfun(function(model,n.core, Var.class, OR.labels,device, Yresp){
  if(toupper(type)=='RF')
  {
    
    sal<-Predmodel.RF(model, file, n.core)
  }
  else if(toupper(type)=='SVM_RAD' | toupper(type)=='SVM_POLY')
  {
    sal<-Predmodel.SVM(model, file, n.core)
  }
  else if(toupper(type)=='XGBOOTS')
  {
    sal<-PredmodeldeepNN(model, file, n.core, Var.class, OR.labels)
  } else if(toupper(type)=='NN'){ 
    sal<-PredmodelNN(model, file, n.core, device, Yresp)
  } else{
    sal<-list()
    sal[['rf']]<-Predmodel.RF(model[[1]], file, n.core)
    sal[['svm_poly']]<-Predmodel.SVM(model[[2]], file, n.core)
    sal[['svm_Rad']]<-Predmodel.SVM(model[[3]], file, n.core)
    sal[['xgboots']]<-Predmodel.XGBoots(model[[4]], file, n.core, Var.class, OR.labels)
    sal[['deepNN']]<-PredmodeldeepNN(model[[5]], file, n.core, device, Yresp)
    sal<-do.call(c, sal)
  }
  sal
})




########################################################################
###############################   RF    ###############################
########################################################################
Predmodel.RF<-cmpfun(function(model, file, n.core, SAVE, namRast){
  print("Starting prediction on the raster...")
writeRaster(predict(file, model, cores=n.core, cpkgs="randomForest"),
file=paste0(SAVE, 'pred_', namRast))

})

########################################################################
############################### SVM      ###############################
########################################################################
Predmodel.SVM<-cmpfun(function(model, file, n.core){
  print("Starting prediction on the raster...")
predSVMfun <- cmpfun(function(m, d, ...) predict(m, newdata=d, ...))
  
r<-predict(file, model, fun=predSVMfun, na.rm=TRUE, cores = n.core, cpkgs = "e1071") 
})

########################################################################
############################### XGBOOTS  ###############################
########################################################################

Predmodel.XGBoots<-cmpfun(function(model, file, n.core, Var.class, OR.labels){
 
  pred.xgb<-cmpfun(function(model, file, Var.class, ...) {
    Xgb.p <-cmpfun(function(m, d, ...)  round(predict(m, newdata=as.matrix(d),iterationrange=c(1,1))))
    CLASI.Xgb<-predict(file, model, fun=Xgb.p, cores=n.core)
    CLASI.Xgb<-as.factor(CLASI.Xgb)
    names(OR.labels)<-c(names(CLASI.Xgb),'class')
    addCats(CLASI.Xgb1, OR.labels, merge=TRUE, layer=1)
    
  })
  
pred.xgb(mXGboots, RAST,Var.class)
})

########################################################################
############################### deepNNTorch  ###############################
########################################################################

PredmodeldeepNN<-cmpfun(function(model, file, n.core, device, Yresp){

device <- torch::torch_device(device)
raster_array <- as.array(file)
raster_tensor <- torch_tensor(raster_array)
raster_tensor <- raster_tensor$to(device = device)

model<-model$to(device = device)
predictions <- model(raster_tensor)
predictions<-as.array(predictions$to(device = device))

resultado_raster_list <- lapply(1:dim(predictions)[2], function(i) {
  Img(as.matrix(predictions[,i], nrow=dim(raster_array)[1], ncol=dim(raster_array)[2]))
})


resultado_raster <- do.call(c, resultado_raster_list)
names(resultado_raster) <- levels(Yresp)
ext(resultado_raster) <- ext(file)
crs(resultado_raster) <- crs(file)
app(resultado_raster, which.max)

})




####

predictMode<-cmpfun(function(modelo, r, namRast, SAVE, n.core){
  
writeRaster(predict(r, modelo, cores=n.core, cpkgs="randomForest"),
              file=paste0(SAVE, 'pred_', namRast))
  
  })
