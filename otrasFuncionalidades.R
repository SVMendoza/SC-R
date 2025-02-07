
###############################################
###############################################
###############################################
	# Estandarizar datos de las imagenes
###############################################
###############################################

# Normalizar entre [0,1]
Rastnorm01 <- cmpfun(function (xx) {
MaRast<-global(xx, fun="max", na.rm=TRUE)
MiRast<-global(xx, fun="min",na.rm=TRUE)
(xx-MiRast[,1])/(MaRast[,1]-MiRast[,1])
})
  
#######################################################################
##########################  Mosaico  #######################
########################################################################

MosaicImg<-cmpfun(function(L.Img){
Resraster <- sprc(L.Img)
mosaic(Resraster, fun='median')
})


########################################################################
##########################  Principal Component  #######################
########################################################################

rastPC<-cmpfun(function(Img, n.core){
   dt<-data.table(spatSample(Img, 1200000, method="regular", as.df=TRUE, na.rm=FALSE, 
                            as.raster=FALSE))  
    PC<- princomp(na.omit(dt), cor = TRUE)  
	cum_variance  <- cumsum((PC$sdev^2)/sum(PC$sdev^2))
	threshold <- 0.97

num_PC<- which(cum_variance >= threshold)[1]    
                   
 predict(Img, PC, cores=n.core, cpkgs="stats",index=1:num_PC)
  
})

########################################################################
########################## Cortar imagenes  ############################
########################################################################
# En batch

cutImage<-cmpfun(function(Img, block_size, SAVE) {
  L<-list()
  for (i in seq(1, nrow(Img), by = block_size - overlap)) {
    for (j in seq(1, ncol(Img), by = block_size - overlap)) {
      print(paste0('segmento', i, '_',j)) 
      block <- Img[i:min(i + block_size - 1, nrow(Img)), 
                 j:min(j + block_size - 1, ncol(Img)), drop = FALSE]
      
      block_filename <- paste0(SAVE, 'Img', i, "_", j, ".tif")
      writeRaster(block, block_filename, overwrite = TRUE)
     # L[ block_filename ]<-block
    }
  }
  L
})

cutImageConv <- cmpfun(function(Img, block_size, SAVE, out_channels, device, kernel_size, stride, padding, n.core, overlap = 24) {
#  L<-list()
 	#Img<-Rastnorm01(Img)  
  	#Img<-rastPC(Img, n.core)
  	
  	#Img<-Rastnorm01(Img)  
  	
  for (i in seq(1, nrow(Img), by = block_size - overlap)) {
    for (j in seq(1, ncol(Img), by = block_size - overlap)) {
      print(paste0('segmento', i, '_',j)) 
      block <- Img[i:min(i + block_size - 1, nrow(Img)), 
                 j:min(j + block_size - 1, ncol(Img)), drop = FALSE]
      
      block_filename <- paste0(SAVE, 'Img', i, "_", j, ".tif")
      block<-convolImg(block, out_channels, device='cpu', kernel_size, stride, padding, n.core)
  if(!is.null(block)) {
        writeRaster(block, block_filename, overwrite = TRUE)
        }
     # L[ block_filename ]<-block
    }
  }
 # L
})
###############################################
###############################################
###############################################
###############################################
## Convolution image
###############################################
###############################################

# Img imagen a convolucionar 
# device= c"cuda", "cpu")
# kernel_size = tamaño del kernel, ventana de convolución (filtro)
# out_channels número de filtros o canales para convolucionar pueden ser 16, 32, 64

########################################################################################
## Prueba ##
convolImg <- function(file.imaa, out_channels, device, kernel_size, stride, padding, n.core, block_size = 256) {
  
  # Ajustar los núcleos en la CPU si es necesario
  if(device == 'cpu') Sys.setenv(OMP_NUM_THREADS = n.core)
  
  # Configurar el dispositivo de trabajo
  device <- torch::torch_device(device)
  
  # Verificar si la imagen tiene valores válidos
  if (all(is.na(values(file.imaa)))) {
    return(NULL)  # Ignorar este bloque si no tiene datos válidos
  } else {
    
    # Definir el número de canales de entrada según la imagen
    in_channels <- nlyr(file.imaa)
    
    # Obtener las dimensiones de la imagen
    nrows <- nrow(file.imaa)
    ncols <- ncol(file.imaa)
    
    # Crear un raster vacío de salida
    resultadoRaster <- rast(nrows = nrows, ncols = ncols, nlyrs = out_channels, ext = ext(file.imaa), crs = crs(file.imaa))
    
    # Definir la capa de convolución
    conv_layer <- nn_conv2d(in_channels = in_channels, 
                            out_channels = out_channels, 
                            kernel_size = kernel_size, 
                            stride = stride, 
                            padding = padding)$to(device=device)
    
    # Dividir la imagen en bloques (sin cargar la imagen completa en memoria)
    for (start_row in seq(1, nrows, by = block_size - stride)) {
      for (start_col in seq(1, ncols, by = block_size - stride)) {
        
        # Definir las coordenadas del bloque
        end_row <- min(start_row + block_size - 1, nrows)
        end_col <- min(start_col + block_size - 1, ncols)
        
        # Cortar el bloque actual de la imagen
        block <- crop(file.imaa, ext(start_col, end_col, start_row, end_row))
        
        # Convertir el bloque a un tensor
        block_array <- as.array(block)
        block_tensor <- torch_tensor(block_array)$to(device=device)
        
        # Ajustar las dimensiones del tensor para la convolución
        block_tensor <- block_tensor$unsqueeze(1)$permute(c(1, 4, 2, 3))  # Cambiar el orden de las dimensiones
        
        # Aplicar la convolución
        feature_maps <- conv_layer(block_tensor)
        
        # Convertir el resultado de vuelta a un array
        resultadoArray <- as.array(feature_maps)
        
        # Colocar los resultados de la convolución en el raster final
        for (i in 1:out_channels) {
          values(resultadoRaster[[i]])[start_row:end_row, start_col:end_col] <- resultadoArray[, i, , ]
        }
      }
    }
    
    # Retornar el raster con la convolución aplicada
    return(resultadoRaster)
  }
}




############## FUNCIONA PERO PARA PEQUEÑAS IMAGENES #####################################

convolImg <- function(file.imaa, out_channels, device, kernel_size, stride, padding, n.core) {
  
  if(device=='cpu') Sys.setenv(OMP_NUM_THREADS = n.core)
  
  device <- torch::torch_device(device)
  
  
  #Img<-rast(paste0(SAVE, file.imaa))
  
  if (all(is.na(values(file.imaa)))) {
    return(NULL)  # Ignorar este bloque
  } else {
  in_channels <- nlyr(file.imaa)
  block_array <- as.array(file.imaa)
  block_tensor <- torch_tensor(block_array)$to(device=device)
  block_tensor <- block_tensor$unsqueeze(1)$permute(c(1, 4, 2, 3))
  
  
  conv_layer <- nn_conv2d(in_channels = in_channels, 
                          out_channels = out_channels, 
                          kernel_size = kernel_size, stride = stride, padding = padding)$to(device=device)
  
  
  feature_maps <- conv_layer(block_tensor)
  resultadoArray <- as.array(feature_maps)
  
  
  resultadoRaster <- rast(nrows = dim(resultadoArray)[3],
                           ncols = dim(resultadoArray)[4],
                           nlyrs = out_channels, ext = ext(file.imaa), crs = crs(file.imaa))
  
  for (i in 1:out_channels) {
    values(resultadoRaster[[i]]) <- resultadoArray[, i, , ]
  }
  #writeRaster(resultadoRaster, paste0(SAVE, 'conv_',file.imaa), overwrite = TRUE)
 return(resultadoRaster)
}

}

#######################################################################################
########################################################################################

###############################################
###############################################
# add XYs
###############################################
###############################################
###############################################
###############################################
Rast.addXY<-cmpfun(function(x) {
  nam<-names(x)
  d<-data.table(as.data.frame(x, xy=TRUE))
  d[,x1:=fifelse(!is.na(get(eval(nam[1]))), x,0)]
  d[,y1:=fifelse(!is.na(get(eval(nam[1]))), y,0)]
  d[,x2:=fifelse(!is.na(get(eval(nam[1]))), x^2,0)]
  d[,x3:=fifelse(!is.na(get(eval(nam[1]))), x^3,0)]
  d[,y2:=fifelse(!is.na(get(eval(nam[1]))), y^2,0)]
  d[,y3:=fifelse(!is.na(get(eval(nam[1]))), y^3,0)]
  r<-RAST[[1]]
  selnam<-c(names(d)[!names(d)%in%nam])
 d<-data.frame(d[,..selnam])
  x1 <- rast(d, type="xyz")
  crs(x1) <- crs(x)
  Rastnorm01(x1)
})


###############################################################
###############################################
###############################################
###############################################
###############################################
# Pooling
###############################################
###############################################
f.poolingIma<-cmpfun(function(r, V, metrics=c('median','mean', 'min', 'max'), ...)
{
  f<- function(metric) aggregate(r, fun=metric,fact=V, cores=cores)
  n<-nlyr(r)
  if(metrics=='min') { r.polling<-f(metric='min')
  names(r.polling)<-paste0(rep('b', n), 1:n,'min')
  }
  else if(metrics=='median') { r.polling<-f(metric='median')
  names(r.polling)<-paste0(rep('b', n), 1:n,'median')
  }
  else if(metrics=='mean') { r.polling<-f(metric='mean')
  names(r.polling)<-paste0(rep('b', n), 1:n,'mean')
  }
  else if (metrics=='max') {r.polling<-f(metric='max')
  names(r.polling)<-paste0(rep('b', n),1:n,'max')
  }
  else if (metrics=='sd') {
    r.polling<-f(metric='sd') 
    names(r.polling)<-paste0(rep('b', n),1:n,'sd')
  }
  else {
    m<-c( 'median','mean','min', 'max','sd')
    polling<-list()
    for(i in 1:5) {polling[[i]]<-f(metric=m[i])
    names(polling[[i]])<-paste0(rep('b', n), 1:n,m[i]) 
    
    }
    r.polling<-rast(polling)    
  }
  
  Rastnorm01(r.polling) 
})


###############################################################
###############################################
###############################################
###############################################
###############################################
## Generar métricas
###############################################

GenNewDescript<-cmpfun(function(x, Convolution=FALSE, XY=FALSE, All=FALSE, Pooling=TRUE, PoolingW=3, metricPool=c('median','mean', 'min', 'max')) {

if(Convolution) { r<-fConvo(x) }
else if(XY){ r <-Rast.addXY(x)}
else if(All) { r<-c(fConvo(x), Rast.addXY(x))}
else if(Pooling) { r<-f.poolingIma(x, V=PoolingW, metrics=metricPool) }
else{ stop('Check the arguments')}
n<-length(names(r))
names(r) <- paste0('B', 1:n)
r

})


