##################################################################
##################################################################
##################################################################
##################################################################
##################################################################
##################################################################
		#Tabla contigencia#
##################################################################
##################################################################
f.confusiontablemetrics<-cmpfun(function(T.contin,...) {

total<-apply(T.contin,1,sum, na.rm=TRUE)
VP<-diag(T.contin)
FP<-apply(T.contin,2,sum, na.rm=TRUE)- diag(T.contin)
FN<-apply(T.contin,1,sum, na.rm=TRUE) - diag(T.contin)
Tpositivos <- VP + FN
Tnegativos<-sum(total, na.rm=TRUE)-Tpositivos
VN<-Tnegativos-FP


preciS<- VP/(VP+FP)#Cuantos son positivos reales. (Positive Predictive Value). cuantos de los calisificados fueron poditivamente relevantes
recall<-  VP/(VP+FN) # recall probabiidad de que un pixel sea un VN (Sensitivity). Cuantos el modelo predice como positivos (cuantos pixeles son positivos
specificity<-VN/(VN+FP) # True negative rate. cuantos obtuvieron resultados negativos. que tan buena es la clasificación para evitar falsas alarmas.
F1score<- 2*((preciS*recall)/(preciS+recall)) #is the harmonic mean of precision and sensitivity/Media harmonica. qué tan buena es la clasificacion para detectar los positivos.
Prec.recallRatio<-preciS/recall
BalancedAccurasyScore <-mean (recall)
preciS.average<-mean (preciS)
F1score.average<- BalancedAccurasyScore*preciS.average/(BalancedAccurasyScore+preciS.average)

#MCC <- ((VP * VN) - (FP * FN)) / sqrt((VP + FP) * (VP + FN) * (VN + FP) * (VN + FN))#MCC: Es una medida que se utiliza para evaluar la calidad de clasificaciones, teniendo en cuenta #verdaderos positivos (VP), falsos positivos (FP), verdaderos negativos (VN) y falsos negativos (FN).

MCC.f <- function() {
for(i in 1:length(VN)) {
 if(is.na(VP[i]) || is.na(VN[i]) || is.na(FP[i]) || is.na(FN[i])){
   MCC[[i]]<-NA
 } else {
   denominator <- sqrt((VP[i] + FP[i]) * (VP[i] + FN[i]) * (VN[i] + FP[i]) * (VN[i] + FN[i]))
   if (denominator == 0) {MCC[[i]]<-NA} else { MCC[[i]]<- ((VP[i] * VN[i]) - (FP[i] * FN[i])) / denominator}
 } 
}
MCC
}

#MCC<-MCC.f()
FNR<- FN/(FN+VP)   #miss rate or false negative rate (FNR)
FPR<- FP/(FP+VN) #miss rate or false positive rate

###Diversidad index
EquitabilityH<-function(x) (-sum(x*log(x)))/log(length(x))

median1<-function(x) {
n<-length(x)
sum(x, na.rm=TRUE)/n
}

metrics<-cbind(preciS, recall,specificity,  F1score)
summa<-data.table(as.data.frame.matrix(as.matrix(T.contin)), metrics)
dsumm<-data.table(t(c(rep(NA, ncol(T.contin)), as.vector(apply(metrics, 2, median1))))    )
colnames(dsumm)<-colnames(summa)
metricaEnd<-rbind(summa, dsumm)
colnames(metricaEnd)<- c(colnames(summa)[-c((ncol(summa)-3):ncol(summa))], 'Presicion', 'Recall',  "specificity", 'F1 score')


return(data.table(Class=c(rownames(T.contin), NA), metricaEnd))
})

##################################################################
##################################################################
##################################################################

		#Tabla contigencia para RF, SVM#

##################################################################
##################################################################
##################################################################



 f.confusionAll<-cmpfun(function(Data, mod, Var.class,...) { 
                                          
                                          n <-nrow(Data) 
                                          TamSegment <- ceiling(n / 50)
                                          
                                          splitData <- split(Data, rep(1:TamSegment, each = TamSegment, length.out = n))
                                          #print('dividio')
                                          plan(multicore, workers = n.core) 
                                          options(future.rng.onMisuse="ignore")
                                          
                                          predictions <- future_lapply(mod, function(mod) {
                                            #print('dentro')
                                           table(Var.class, 
                                                  do.call(c, lapply(splitData, 
                                                                    function(x) predict(mod, newdata = x))))
                                            
                                                                     })                                         
                       
                                          lapply(predictions, function(x)   f.confusiontablemetrics(x) )   
                                            })

##################################################################
##################################################################
##################################################################

		#Tabla contigencia solo XGboots#

####################################################################
##################################################################
##################################################################


f.DcontinXgbts<-cmpfun(function(Data, mod,Var.class,OR.labels,... ) {

  	p<-data.table(pred=as.factor(round(predict(mod, as.matrix(Data[,-c(1:2)]), iterationrange = c(1, 1)))))

p[, fac:=factor(pred, levels =OR.labels$Asigna, labels=OR.labels$ORIG)]
#f.cat<- cmpfun(function(x) if(sum(OR.labels$V1%in%x)>0) return(OR.labels[OR.labels$V1%in%x, ]$ORIG) else return(x))
#p[, VALOR:=Map(f.cat, p$p.val)]

T.contin<-table(Var.class, p$fac)

T.contin<-T.contin[,apply(T.contin, 2, sum)>0]
T.continEnd<-T.contin[, colnames(T.contin)%in% OR.labels$ORIG]
T.continEnd
}
)

##################################################################
##################################################################
##################################################################

		#Tabla contigencia solo DeepNN#

####################################################################
##################################################################
##################################################################
f.DcontinDeepNN<-cmpfun(function(Data, mod,Var.class,... ) {

f<-cmpfun(function(i) unlist(Data[i,-c(1:2)]))
plan(multisession, workers = n.core) 
options(future.rng.onMisuse="ignore")
d<-future_lapply(1:nrow(Data),FUN=f)

pred <- lapply(1:length(d),function(i) {NNpredict(net=net,
                    param=netwts$opt,
                    newdata=d[i],
                    newtruth=list(rep(1, length(OR.labels$ORIG))),
                    record=TRUE,
                    plot=FALSE)})

sal<-factor(sapply(1:length(pred), function(i) OR.labels$ORIG[pred[[i]]$pred_MC]))

T.contin<-table(Var.class,sal)
T.contin<-T.contin[,levels(Var.class)]
})
