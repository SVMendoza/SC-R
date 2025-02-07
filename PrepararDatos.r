
######### normalizar imagen #############################
extract.dat<-cmpfun(function(file.img, name.shape, name.CLASES,OPEN,SAVE, n.core, exact=NULL, Normalize=FALSE, envir=.GlobalEnv) {

print("Preparing the data for training...")
  if(is.null(n.core)) n.core<-detectCores()-1 #
if(Normalize){ 
RAST <- Rastnorm01(rast(paste0(OPEN, file.img)))
names(RAST)<-paste0('Lyrs', 1:nlyr(RAST))
 #saveRDS(RAST, file=paste0(SAVE,'RAST.rds')) 
 writeRaster(RAST, file=paste0(SAVE,'RAST.tif'), overwrite=TRUE)
}  else {
RAST <- rast(paste0(OPEN, file.img)) 
names(RAST)<-paste0('Lyrs', 1:nlyr(RAST))
}

   name.band<<-names(RAST)
   
CLA <- vect(paste0(OPEN, name.shape))
	DAT<-data.table(extract(RAST,CLA, na.rm=FALSE, method='simple', df=TRUE))
CLA <-data.frame(as.data.frame(CLA)[,name.CLASES[1]], 1:nrow(CLA))
names(CLA)<-c( name.CLASES[1], 'ID')
      
	DAT.todos <- merge(data.table(CLA), DAT, by='ID', all=TRUE) 
      nLevels<-nlevels(as.factor(data.frame(CLA)[,name.CLASES[1]]))
  DAT.todos<-na.omit(DAT.todos)
  DAT.todos[,name.CLASES[1]:=as.factor(get(eval(name.CLASES[1])))]
  return(DAT.todos)
})


#####################################################################
#####################################################################
		## Separa los datos en train and validate
#####################################################################
#####################################################################
d.val<-cmpfun(function(propVal, DAT.todos) {

sumVal<-DAT.todos[,unique(ID), by=eval(name.CLASES)]
mVal<-sumVal[,.N, by=eval(name.CLASES)]
mVal[, polVal:=round(N*propVal)]
V.pol<-unlist(lapply(1:nrow(mVal), function(i) sample(sumVal[mVal[i ,get(eval(name.CLASES))]==get(eval(name.CLASES)), ]$V1,  mVal[i ,polVal])))

VAL.d<-DAT.todos[, ifelse(ID%in%V.pol,1,0)]
return(list(d.train=DAT.todos[VAL.d==0,], d.Val=DAT.todos[VAL.d==1,]))
})
#################################################
#################################################
#################################################
	##Set datos ###
#################################################
#################################################
#################################################
f.train<-cmpfun(function(data, name.CLASES, sel.n, ndt, ...) {
  if(is.null(n.core)) n.core<-detectCores()-1
  DAT.todos<-data.table(data)
un<-dim(DAT.todos)[1]==length(unique(DAT.todos$ID))
if(isTRUE(un)) {f.random.dat<- cmpfun( function(i)  sub.dt<-DAT.todos[,.SD[sample(.N,min(.N,sel.n))],by = get(eval(name.CLASES))][,-c('get')] ) } else{
f.random.dat<- cmpfun( function(i)  sub.dt<-DAT.todos[,.SD[sample(.N,min(.N,sel.n))],by = 'ID'])
}

cl<-makeCluster(n.core)
clusterEvalQ(cl, library('data.table'))
clusterExport(cl=cl,list('DAT.todos', 'name.CLASES', 'sel.n'), envir=environment())
setDat.parar<-parLapply(cl,c(1:ndt),fun=f.random.dat)
stopCluster(cl)
return(setDat.parar)
})


#################################################
	##Set datos por tamaño ###
#################################################

f.trainBalance<-cmpfun(function(i) {
  L<-list()
  uniqueClases<-levels(data[[name.CLASES]])
  for(clases in uniqueClases) {
    subdata<-data[get(eval(name.CLASES))==clases, ]
    idvalue<-unique(subdata$ID)
    if(length(idvalue)>nsize) {
      L[[clases]]<-subdata[ID%in%sample(idvalue, nsize), .SD[sample(.N, 1)], by = ID]
    } else {
      numSamp<-nsize-length(idvalue)
      L[[clases]]<-rbind(subdata[, .SD[sample(.N, 1)], by = ID], subdata[, .SD[sample(.N, numSamp, replace=TRUE)]])
    }
    
  }
  rbindlist(L)
})

f.trainBal <- cmpfun(function(data, name.CLASES, ndt, nsize, 
                           n.core, ...) {
  if (is.null(n.core)) n.core <- detectCores() - 1  # Detectar núcleos por defecto
  plan(multicore, workers = n.core) 
  options(future.rng.onMisuse="ignore")
 future_lapply(1:ndt,FUN=f.trainBalance)
})
###############################################
###############################################
###############################################
###############################################
	# Balancear datos
###############################################
###############################################

BalancearDts <- function(datos, name.CLASES) {
  
Maxsize<-max(datos[,.N, by=name.CLASES]$N)
  
 DtList <- split(datos, by = name.CLASES)
    oversampList <- lapply(DtList  , function(x) {
    n <- nrow(x)
    if (n < Maxsize) {
      necesarios.n <- size - n
      sampleX <- x[sample(.N, necesarios.n, replace = TRUE)]
      return(rbind(x, sampleX))
    }
    x
  })
  rbindlist(DtList)
}

###############################################
###############################################
###############################################
###############################################
	# Dividir datos para predecir
###############################################
###############################################
#dT tabla y segmentSize = tamaño de la tabla dividida
splitDT <- function(dt, segmentSize) {
  lista <- list()
  n <- nrow(dt)
  
 for (i in seq(1, n, by = segmentSize)) {
   lista[[length(lista) + 1]] <- dt[i:min(i + segmentSize - 1, n), ]
  }
   lista
}
###############################################################
###############################################
###############################################

