#######################################################################
#######################################################################
#				____________
# 				randomForest 
#				------------
#######################################################################
#######################################################################
FunctionRF<-cmpfun(function(...){
source(paste0(FuncionesLocal, 'CombineRF.r'))
source(paste0(FuncionesLocal, 'ConfuTable.r'))

n.class<<-nlevels(itrain[[1]][,get(eval(name.CLASES))])
m<-as.formula(paste(paste(name.CLASES[1], '~'), paste(names(itrain[[1]])[-c(1:2)], collapse='+')))
m.rf<-cmpfun(function(x) randomForest(m, x,ntree=100))


 print("Starting the training...")

plan(multicore, workers = n.core) 
options(future.rng.onMisuse="ignore")
#modls<-future_lapply(itrain,FUN=m.rf)
modls <- future_lapply(seq_along(itrain), function(i) {

     print(paste("Training model", i, "of", length(itrain))) 

    m.rf(itrain[[i]])
    
  })
print("All models have been trained.")



Data<-DAT.test
Var.class<-Data[,get(eval(name.CLASES))]
salida<-f.confusionAll(Data, modls, Var.class)
errrrr<-sapply(salida, function(x)   last(x[,get(eval('F1 score'))]))

mRf<-modls[which.max(errrrr)][[1]]
save(mRf, file=paste0(SAVE,'trainModRF.RData'))
print(paste0('the best model is save in:', SAVE))

tconfu<-salida[which.max(errrrr)][[1]]
print(knitr::kable(tconfu), caption='test/predict')

fwrite(tconfu, file=paste0(SAVE,'MetricsPerformsRF.csv'), sep=',')
print(paste("Performance metrics are saved in:", SAVE))


mRf
})

#######################################################################
#######################################################################
#				_______________
# 				SVM  polynomial
#				---------------
#######################################################################
#######################################################################
FunctionPoly<-cmpfun(function(...){
source(paste0(FuncionesLocal, 'ConfuTable.r'))
#################

fPesos<-function(x) {
  classesPesos <- table(x[,get(eval(name.CLASES))])
  classesPesos <- 1 / classesPesos
  classesPesos <- classesPesos / sum(classesPesos)  # Normalizar
  classesPesos 
}
m<-as.formula(paste(paste(name.CLASES[1], '~'), paste(names(itrain[[1]])[-c(1:2)], collapse='+')))

m.svm.Poly<-cmpfun(function(x) {
  classesPesos<-fPesos(x)
  classesPesos <- setNames(as.list(classesPesos), names(classesPesos))
  
e1071::svm(m, x, scale=FALSE, type='C-classification',tolerance=0.1, kernel="polynomial", 
           degree=3, coef0=0, cost = 10, class.weights =classesPesos)
})

print("Starting the training...")


plan(multicore, workers = n.core) 
options(future.rng.onMisuse="ignore")
modls <- future_lapply(seq_along(itrain), function(i) {
    print(paste("Training model", i, "of", length(itrain))) 

    m.svm.Poly(itrain[[i]])
    
  })
print("All models have been trained.")


Data<-DAT.test
Var.class<-Data[,get(eval(name.CLASES))]
salida<-f.confusionAll(Data, modls, Var.class)
errrrr<-sapply(salida, function(x)   last(x[,get(eval('F1 score'))]))

msvmPoly<-modls[which.max(errrrr)][[1]]
save(msvmPoly, file=paste0(SAVE,'trainModsvmPoly.RData'))
print(paste0('the best model is save in:', SAVE))

tconfu<-salida[which.max(errrrr)][[1]]


print(knitr::kable(tconfu), caption='test/predict')
fwrite(tconfu, file=paste0(SAVE,'MetricsPerformsSVMPolyn.csv'), sep=',')
print(paste("Performance metrics are saved in:", SAVE))

msvmPoly
})

#######################################################################
#######################################################################
#				__________
# 				SVM Radial
#				----------
#######################################################################
#######################################################################
FunctionRad<-cmpfun(function(...){
source(paste0(FuncionesLocal, 'ConfuTable.r'))

fPesos<-function(x) {
  classesPesos <- table(x[,get(eval(name.CLASES))])
  classesPesos <- 1 / classesPesos
  classesPesos <- classesPesos / sum(classesPesos)  # Normalizar
  classesPesos 
}

m<-as.formula(paste(paste(name.CLASES[1], '~'), paste(names(itrain[[1]])[-c(1:2)], collapse='+')))

m.svm.radial<- cmpfun(function(x) {
  classesPesos<-fPesos(x)
  classesPesos <- setNames(as.list(classesPesos), names(classesPesos))
  
 e1071::svm(m, x, scale=FALSE, type='C-classification',tolerance=0.1,  kernel = "radial", cost = 5,class.weights =classesPesos )
})

plan(multicore, workers = n.core) 
options(future.rng.onMisuse="ignore")
modls <- future_lapply(seq_along(itrain), function(i) {
    print(paste("Training model", i, "of", length(itrain))) 

    m.svm.radial(itrain[[i]])
    
  })
print("All models have been trained.")


Data<-DAT.test
Var.class<-Data[,get(eval(name.CLASES))]
salida<-f.confusionAll(Data, modls, Var.class)
errrrr<-sapply(salida, function(x)   last(x[,get(eval('F1 score'))]))

msvmRad<-modls[which.max(errrrr)][[1]]
save(msvmRad, file=paste0(SAVE,'trainModsvmRad.RData'))
print(paste0('the best model is save in:', SAVE))

tconfu<-salida[which.max(errrrr)][[1]]

print(knitr::kable(tconfu), caption='test/predict')
fwrite(tconfu, file=paste0(SAVE,'MetricsPerformsSVMRadial.csv'), sep=',')
print(paste("Performance metrics are saved in:", SAVE))
msvmRad
})

#######################################################################
#######################################################################
#				____
# 				nnet
#				----
#######################################################################
#######################################################################
Functionnnet<-cmpfun(function(...){
source(paste0(FuncionesLocal, 'ConfuTable.r'))
m<-as.formula(paste(paste(name.CLASES[1], '~'), paste(names(itrain[[1]])[-c(1:2)], collapse='+')))
n.class<<-nlevels(itrain[[1]][,get(eval(name.CLASES))])
m.nnet<-cmpfun(function(x) nnet(m, x,size=n.class*10, decay=5e-4, maxit=400, trace = FALSE))



predNNETfun <- cmpfun(function(m, d, ...) factor(predict(m, newdata=d,type ='class',...)))

f.bestmodel<-cmpfun(function(Data, modls,Var.class,...) {
errrrr<-unlist(lapply(modls, function(m)   last(f.confusiontablemetrics(f.confusionnnet(Data, m, Var.class))) [,'F1 score']))
modls[which.max(errrrr)][[1]]
})


plan(multisession, workers = n.core) 
options(future.rng.onMisuse="ignore")
modls<-future_lapply(itrain,FUN=m.nnet)

Data<-DAT.test
Var.class<-Data[,get(eval(name.CLASES))]
mnnet<-f.bestmodel(DAT.test,modls, Var.class)
tconfu<-f.confusiontablemetrics(f.confusionnnet(Data, mnnet, Var.class))


RAST <- readRDS(paste0(SAVE,'RAST.rds'))

CLASI<-predict(RAST, mnnet, fun=predNNETfun, cores = n.core,cpkgs="nnet")
	 writeRaster(CLASI, filename=paste0(SAVE, 'Nnet', '.tif'), overwrite=TRUE)

print(knitr::kable(tconfu), caption='test/predict')

fwrite(tconfu, file=paste0(SAVE,'MetricsPerformsnnet.csv'), sep=',')
save(mnnet, file=paste0(SAVE,'trainModnnet.RData'))
print(paste0('Model is save in:', SAVE))
CLASI
})

#######################################################################
#######################################################################
#				_______
# 				XGboots
#				-------
#######################################################################
#######################################################################
FunctionXGB<-cmpfun(function(...){
source(paste0(FuncionesLocal, 'ConfuTable.r'))

itrain.XGB<-data.table(rbindlist(itrain))

LaBel<-as.numeric(as.factor(itrain.XGB[,get(eval(name.CLASES))]))-1
OR.labels<-data.table(Asigna=factor(unique(sort(LaBel))),
                      ORIG=factor(levels(as.factor(itrain.XGB[,get(eval(name.CLASES))]))))

fPesos<-function(x) {
  classesPesos <- table(x)
  classesPesos <- 1 / classesPesos
  classesPesos <- classesPesos / sum(classesPesos)  # Normalizar
  classesPesos 
}

Pesos <- fPesos(LaBel)[as.character(LaBel)]


param <- list("objective" = "multi:softmax",
              "num_parallel_tree"=100,
              "num_class" = n.class,
              "eval_metric" = "merror",
              "colsample_bytree" = .7,
              'subsample' = .7,
              'gamma'=2,
              "eta" = .1, "max.depth" = 16)

mXGboots<-xgboost(data = as.matrix(itrain.XGB[,-c(1:2)]),
                  label =LaBel,
                  weight = Pesos, 
                  param, 
                  nrounds = 1000)
                  
save(mXGboots, file=paste0(SAVE,'trainModXGboots.RData'))

Data<-DAT.test
Var.class<-Data[,get(eval(name.CLASES))]
tconfu<-f.confusiontablemetrics(f.DcontinXgbts(Data, mXGboots, Var.class, OR.labels))
print(knitr::kable(tconfu), caption='test/predict')

write(tconfu, file=paste0(SAVE,'MetricsPerformsnnet.csv'))
mXGboots
})


#######################################################################
#######################################################################
#				_______
# 				deepNN Torch
#				-------
#######################################################################
#######################################################################

nnNet.fun <- cmpfun(function(itrain, name.CLASES, SAVE, DAT.test, OPEN, file.img, num_epochs, batch_size, FuncionesLocal, ...) {
  
  source(paste0(FuncionesLocal, 'ConfuTable.r'))

  #device1 <- torch::torch_device(if (torch::cuda_is_available()) "cuda" else "cpu")
  device1 <-device1 <- torch::torch_device("cpu")
  # Definiendo la red
  Net <- nn_module(
    "Net",
    
    initialize = function(input_size, hidden_sizes, output_size, dropout_probs) {
      self$layers <- nn_sequential(
        nn_linear(input_size, hidden_sizes[1]),
        nn_relu(),
        nn_dropout(dropout_probs[1]),
        nn_linear(hidden_sizes[1], hidden_sizes[2]),
        nn_relu(),
        nn_dropout(dropout_probs[2]),
        nn_linear(hidden_sizes[2], hidden_sizes[3]),
        nn_relu(),
        nn_dropout(dropout_probs[3]),
        nn_linear(hidden_sizes[3], output_size),
        nn_softmax(dim = 1) 
      )
    },
    
    forward = function(x) {
      self$layers(x)
    }
  )
  
  # Preparar los datos
  dX <- rbindlist(itrain)
  Yresp <- dX[, get(eval(name.CLASES))]
  
  #x_batch <- torch_tensor(x_batch, dtype = torch_float())$to(device1)
  x <- torch_tensor(as.matrix(dX[, -c(1, 2)]), dtype = torch_float())
  y <- torch_tensor(as.numeric(factor(unlist(Yresp))), dtype = torch_long())  # Convertir a índices
  
  ################## Configurar el entrenamiento #####################
  
  input_size <- ncol(dX[, -c(1, 2)])  # Número de características de entrada
  hidden_sizes <- c(40, 100, 40)  # Tamaños de las capas ocultas
  output_size <- length(unique(Yresp))  # Número de clases
  
  # Probabilidades de dropout
  dropout_probs <- c(0.1, 0.15, 0.15)  # Probabilidades de dropout para cada capa
  
  # Crear la red
  model <- Net(input_size, hidden_sizes, output_size, dropout_probs)
  
  # Definir el optimizador y la función de pérdida
  optimizer <- optim_adam(model$parameters, lr = 0.0001)
  loss_fn <- nn_cross_entropy_loss()
  

  # Entrenar
  num_epochs <- num_epochs  # Ajusta el número de épocas según sea necesario
  batch_size <- batch_size  # Tamaño del lote
  
  #device1 <- torch::torch_device(if (torch::cuda_is_available()) "cuda" else "cpu")
  
  model <- model$to(device=device1)  # Mover el modelo al dispositivo
  
  for (epoch in 1:num_epochs) {
    
    model$train()  # Poner el modelo en modo entrenamiento
    
    # Inicializar vector para pérdidas
    train_losses <- vector()  # Inicializar un vector vacío
    
    # Dividir en lotes
    for (i in seq(1, nrow(dX), by = batch_size)) {
      x_batch <- as.matrix(dX[i:min(i + batch_size - 1, nrow(dX)), -c(1, 2)])
      y_batch <- y[i:min(i + batch_size - 1, nrow(dX))]
      
      # Convertir a tensores y mover al dispositivo
      x_batch <- torch_tensor(x_batch, dtype = torch_float())$to(device=device1)
      y_batch <- y_batch$to(device=device1)
      
      # Hacer la pasada hacia adelante
      optimizer$zero_grad()
      y_pred <- model(x_batch)
      loss <- loss_fn(y_pred, y_batch)
      
      # Hacer la pasada hacia atrás
      loss$backward()
      optimizer$step()
      
      # Almacenar la pérdida
      train_losses <- c(train_losses, loss$item())
    }
    
    # Imprimir la pérdida media cada época
    cat(sprintf("Loss at epoch %d: training: %3f\n", epoch, mean(train_losses)))
  }
  
  ##########################Evaluación del modelo #####################
  torch_save(model, paste0(SAVE, "modeloRN.pt"))
  
  ## Asegúrate de que eval.dt esté preparado correctamente
  xtest <- torch_tensor(as.matrix(DAT.test[,-c(1:2)]), dtype = torch_float())
  xtest <-xtest$to(device = device1)
  
  ytest<-torch_tensor(as.numeric(factor(unlist(DAT.test[,get(eval(name.CLASES))]))) , dtype = torch_long())  
  ytest<-ytest$to(device = device1)
  # Poner el modelo en modo evaluación y hacer predicciones
  
  model$eval()
  
  # Hacer las predicciones
  with_no_grad({
    predictions <- model(xtest)
  })
  
  # Comprobar el tamaño de predictions
  predictions<-predictions$to(device = "cpu")
  max_result <- data.table(as.matrix(predictions))
  colnames(max_result)<-levels(Yresp)
  vPred<-c()
  for(i in 1:nrow(max_result)) vPred[i]<-levels(Yresp)[which.max(max_result[i,])]
  
  # Obtener las clases predichas
  tconfu<-f.confusiontablemetrics(table(DAT.test[,get(eval(name.CLASES))], vPred)) 
  
  print(knitr::kable(tconfu), caption='test/predict')
  
  fwrite(tconfu, file=paste0(SAVE,'MetricsPerformsnnet.csv'))
  #save(model, file=paste0(SAVE,'trainModNNtorch.RData'))
  print(paste0('Model is save in:', SAVE))
  
  
  model
})

#######################################################################
#######################################################################
#				_______
# 				deepNN
#				-------
#######################################################################
#######################################################################
FunctiondeepNN<-cmpfun(function(itrain, name.CLASES, SAVE,DAT.test, OPEN, file.img, FuncionesLocal,...){
source(paste0(FuncionesLocal, 'ConfuTable.r'))

source(paste0(FuncionesLocal, 'predictNNMisc.r'))


d<-rbindlist(itrain)
v<-d[,get(eval(name.CLASES))]
d1<-f(v)

plan(multisession, workers = n.core) 
options(future.rng.onMisuse="ignore")
truth <- future_lapply(1:nrow(d1),FUN=function(i) unlist(d1[i,]))

plan(multisession, workers = n.core) 
options(future.rng.onMisuse="ignore")
dat <- future_lapply(1:nrow(d),FUN=function(i) unlist(d[i,-c(1:2)]))


net <- network( dims = c(length(names(d)[-c(1:2)]),10, 4,10, length(truth[[1]])),
                activ=list(ReLU(),sigmoid(), ReLU(),ReLU(),softmax()))


netwts <- train(dat=dat,
                truth=truth,		#lista de clases
                net=net,
                eps=0.01,
                tol=100,            # run for 100 iterations
                batchsize=900,       # note this is not enough
                loss=multinomial(), # for convergence
                stopping="maxit",
		plot=FALSE)
LABel<-as.numeric(as.factor(d[,get(eval(name.CLASES))]))-1
OR.labels<-data.table(Asigna=factor(unique(sort(LABel))),
                      ORIG=factor(levels(as.factor(d[,get(eval(name.CLASES))]))))


Data<-DAT.test
Var.class<-Data[,get(eval(name.CLASES))]
tconfu<-f.confusiontablemetrics(f.DcontinDeepNN(Data, netwts, Var.class, OR.labels))

RAST <- readRDS(paste0(SAVE,'RAST.rds'))

#system.time(vsub[,PREDIC:=predRast(RAST)])


CLASI<-predRast(RAST,...)
writeRaster(CLASI, filename=paste0(SAVE, 'DeepNN', '.tif'), overwrite=TRUE)


print(knitr::kable(tconfu), caption='test/predict')

fwrite(tconfu, file=paste0(SAVE, 'MetricsPerformDeepNN.csv'), sep=',')
save(netwts, file=paste0(SAVE, 'trainModDeepNN.RData'))
print(paste0('Model is save in:',SAVE))
CLASI
})


########################################################################################################
########################################################################################################
##################### 		SVM radial o polinomico     ############################################
########################################################################################################
########################################################################################################
FunctionRadIncrement<-cmpfun(function(...){
source(paste0(FuncionesLocal, 'ConfuTable.r'))

fPesos<-function(x) {
  classesPesos <- table(x[,get(eval(name.CLASES))])
  classesPesos <- 1 / classesPesos
  classesPesos <- classesPesos / sum(classesPesos)  # Normalizar
  classesPesos 
}

m<-as.formula(paste(paste(name.CLASES[1], '~'), paste(names(itrain[[1]])[-c(1:2)], collapse='+')))



m.svm.radialTune<- cmpfun(function(x) {
  classesPesos<-fPesos(x)
  classesPesos <- setNames(as.list(classesPesos), names(classesPesos))
  e1071::tune.svm(m, x, scale=FALSE, type='C-classification',
                gamma = 2^(-1:1),
                cost = 2^(2:4),
                kernel = "polynomial",
                degree = 3,   
                coef0 = 1,
                class.weights =classesPesos)
})

m.svm.radialInitPara<-m.svm.radial(itrain[[1]])


modeloInicial <- svm(m, x, scale=FALSE, type='C-classification',
                     kernel = "polynomial", 
                     degree = 3,
                     coef0 = 1,
                     gamma = m.svm.radialInitPara$best.parameters$gamma,
                     cost = m.svm.radialInitPara$best.parameters$cost)

print("Incremental training has started")
for(i in 2:lenght(itrain)) updated_m<- update(modeloInicial, data = itrain[[i]])

print("Incremental training has concluded")


Data<-DAT.test
Var.class<-Data[,get(eval(name.CLASES))]


save(updated_m, file=paste0(SAVE,'trainModsvmRad.RData'))
print(paste0('the best model is save in:', SAVE))

tconfu<-confusionAll(Data, updated_m, Var.class) 

print(knitr::kable(tconfu, caption='test/predict'))

fwrite(tconfu, file=paste0(SAVE, 'MetricsPerformSVMRadial.csv'), sep=',')
print(paste0("Performance metrics are saved in: ", SAVE))

updated_m
})





  
  
########################################################################################################
########################################################################################################  
  
########################################################################################################
########################################################################################################  
  





#######################################################################################################
						######################################################
########################################
			#################################################################
			#                Reentrenar Torch
			
######################################################################################################
nnNet.fun.reentrenar <- cmpfun(function(itrain, name.CLASES, SAVE, DAT.test, OPEN, num_epochs, batch_size, FuncionesLocal, ...) {
  Net <- nn_module(
    "Net",
    
    initialize = function(input_size, hidden_sizes, output_size, dropout_probs) {
      self$layers <- nn_sequential(
        nn_linear(input_size, hidden_sizes[1]),
        nn_relu(),
        nn_dropout(dropout_probs[1]),
        nn_linear(hidden_sizes[1], hidden_sizes[2]),
        nn_relu(),
        nn_dropout(dropout_probs[2]),
        nn_linear(hidden_sizes[2], hidden_sizes[3]),
        nn_relu(),
        nn_dropout(dropout_probs[3]),
        nn_linear(hidden_sizes[3], output_size),
        nn_softmax(dim = 1) 
      )
    },
    
    forward = function(x) {
      self$layers(x)
    }
  )
  device1 <-device1 <- torch::torch_device("cpu")
  model <- torch_load(paste0(SAVE, "modeloRN.pt"))
  #load(paste0(SAVE, 'trainModNNtorch.RData'))
  model <- model$to(device=device1)  # Mover el modelo al dispositivo
  source(paste0(FuncionesLocal, 'ConfuTable.r'))
# Preparar los nuevos datos
dX_nuevos <- rbindlist(itrain)  # Suponiendo que tienes un nuevo conjunto de entrenamiento
Yresp_nuevos <- dX_nuevos[, get(eval(name.CLASES))]

x_nuevos <- torch_tensor(as.matrix(dX_nuevos[, -c(1, 2)]), dtype = torch_float())
y_nuevos <- torch_tensor(as.numeric(factor(unlist(Yresp_nuevos))), dtype = torch_long())  # Convertir a índices

device1 <-device1 <- torch::torch_device("cpu")
# Definir el optimizador y la función de pérdida
optimizer <- optim_adam(model$parameters, lr = 0.0001)  # Ajusta la tasa de aprendizaje si es necesario
loss_fn <- nn_cross_entropy_loss()

num_epochs <- num_epochs # Ajusta según sea necesario
batch_size <- batch_size  # Tamaño del lote



for (epoch in 1:num_epochs) {
  model$train()  # Poner el modelo en modo entrenamiento
  
  train_losses <- vector()  # Inicializar un vector vacío
  
  # Dividir en lotes
  for (i in seq(1, nrow(dX_nuevos), by = batch_size)) {
    x_batch <- as.matrix(dX_nuevos[i:min(i + batch_size - 1, nrow(dX_nuevos)), -c(1, 2)])
    y_batch <- y_nuevos[i:min(i + batch_size - 1, nrow(dX_nuevos))]
    
    # Convertir a tensores y mover al dispositivo
    x_batch <- torch_tensor(x_batch, dtype = torch_float())$to(device=device1)
    y_batch <- y_batch$to(device=device1)
    
    # Hacer la pasada hacia adelante
    optimizer$zero_grad()
    y_pred <- model(x_batch)
    loss <- loss_fn(y_pred, y_batch)
    
    # Hacer la pasada hacia atrás
    loss$backward()
    optimizer$step()
    
    # Almacenar la pérdida
    train_losses <- c(train_losses, loss$item())
  }
  
  # Imprimir la pérdida media cada época
  cat(sprintf("Loss at epoch %d: training: %3f\n", epoch, mean(train_losses)))
}

torch_save(model, paste0(SAVE, "modelo_actualizado.pt"))

xtest <- torch_tensor(as.matrix(DAT.test[,-c(1:2)]), dtype = torch_float())
xtest <-xtest$to(device = device1)
ytest<-DAT.test[,get(eval(name.CLASES))]
#ytest<-torch_tensor(as.numeric(factor(unlist(DAT.test[,get(eval(name.CLASES))]))) , dtype = torch_long())  
#ytest<-ytest$to(device = device1)
# Poner el modelo en modo evaluación y hacer predicciones
#save(model, file = paste0(SAVE, "modelo_actualizado.RData"))

predictions <- model(xtest)
#model$eval()

# Hacer las predicciones
#with_no_grad({
#  predictions <- model(xtest)
#})

# Comprobar el tamaño de predictions
predictions<-predictions$to(device = "cpu")
max_result <- data.table(as.matrix(predictions))
colnames(max_result)<-levels(ytest)
vPred<-c()
for(i in 1:nrow(max_result)) vPred[i]<-levels(Yresp_nuevos)[which.max(max_result[i,])]

# Obtener las clases predichas
tconfu<-f.confusiontablemetrics(table(DAT.test[,get(eval(name.CLASES))], vPred)) 

print(knitr::kable(tconfu), caption='test/predict')
})

