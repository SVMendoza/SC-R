f<-cmpfun(function(x) {
  valorUnico<-levels(x)
  mtrust <- matrix(0, nrow=length(x), ncol=length(levels(x)))
  
  for (i in 1:nrow(mtrust)) {
    mtrust[i, which(valorUnico == x[i])] <- 1
  }
  mtrust
})
  

#fcrea<-cmpfun(function(x) unlist(x[i,]))
########################PREDICT
fRast<-cmpfun(function(i) unlist(vsub1[i,])) 

f.pred<-cmpfun(function(i) { 
 print(i)
  OR.labels$ORIG[NNpredict(net=net,
                 param=netwts$opt,
                 newdata=valor[i],
                 newtruth=list(rep(1, length(OR.labels$ORIG))),
                 record=TRUE,
                 plot=FALSE)$pred]

})


predRast <- cmpfun(function(RAST,...) {
  v<-as.data.frame(RAST, na.rm=FALSE)
v<-data.table(v)
nam<-names(RAST)[1]
  v[, naD:=fifelse(is.na(get(eval(nam[1]))), 1,0)]
  v[, ContabID:=seq(1,.N)]

vsub<-data.table(subset(v,naD!=1))[,-c("naD", "ContabID")]

fcrea<-cmpfun(function(i) unlist(vsub[i,]))
plan(multisession, workers = n.core) 
  options(future.rng.onMisuse="ignore")
  valor<-future_lapply(1:nrow(vsub),FUN=fcrea)

vsub<-data.table(subset(v,naD!=1))
vsub$pred<-map(valor, f.pred)
 
v<-merge(v, vsub[,c('ContabID', 'pred')], by='ContabID')

RAST<-RAST[[1]]
values(RAST)<-unlist(setorder(v, cols = "ContabID")$pred)
RAST
})




################################


f.pred<-cmpfun(function(x) { 
 #print(x)
  OR.labels$ORIG[which.max(NNpredict1(net=net,
                 param=netwts$opt,
                 newdata=valor[1],
                ))]

})






NNpredict1 <- function(net,param,newdata,freq=1000){

        testdat <- newdata[[i]]
        w <- weights2list(param[1:nnetpar(net)],net$dims)
        b <- bias2list(param[(nnetpar(net)+1):length(param)],net$dims)
        cls <- net$forward_pass(testdat,
                                weights=w,
                                bias=b,
                                dims=net$dims,
                                nlayers=net$nlayers,
                                activ=net$activ,
                                back=FALSE,
                                regulariser=net$regulariser)

       
         return(cls$output)

}



weights2list <- function(weights,dims){
    dims <- c(dims,0)
    wlist <- list()
    start <- 1
    for(i in 1:(length(dims)-2)){
        end <- start + dims[i]*dims[i+1] - 1
        wlist[[i]] <- matrix(weights[start:end],dims[i+1],dims[i])
        start <- start + dims[i]*dims[i+1]
    }
    return(wlist)
}

bias2list <- function(bias,dims){
    blist <- list()
    start <- 1
    end <- dims[2]
    blist[[1]] <- bias[start:end]
    for(i in 2:(length(dims)-1)){
        start <- end + 1
        end <- start + dims[i+1] -1
        blist[[i]] <- bias[start:end]
    }
    return(blist)
}
