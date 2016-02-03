#Función para obtener la clasificación de un dataset comprimido mediante diferentes clasificadores y modelos de compresión
#data -> Dataset seleccionado.
#porcLayers -> Vector con el porcentaje de las diferentes capas a aplicar, si no es válido se aplica sin compresión.
#modelo -> Modelo seleccionado para realizar la compresión, si no es válido se aplica sin compresión.
#clasif -> Clasificador que se aplicará, si no es válido no se obtiene resultados.
deepLearning_classification <- function(data,porcLayers,modelo,clasif){
  timTot<-Sys.time()
  porcLayersC<-paste(porcLayers,collapse="_")
  contents<-paste("Ejecución para el dataset",data,"con el modelo",modelo,"y el clasificador",clasif,":\n",sep=" ")
  contents<-paste(contents,"Porcentajes de las capas: ",porcLayersC,"\n",sep="")
  accTot<-0.0
  timComTot<-0.0
  timClasTot<-0.0
  porcLayers<-sort(porcLayers,decreasing = TRUE)
  if(porcLayers[length(porcLayers)]<=0 || !is.numeric(porcLayers)){
    modelo<-"NO_APLICA"
  }
  for(j in 1:2){ 
    for(i in 1:5){
      ej<-paste(j,"-",i,sep="")
      contents<-paste(contents,"----Ejecución",ej,":----\n",sep=" ")
      nombreTest<-paste(data,j,"_","5-",i,".txt",sep="")
      test<-read.table(nombreTest,sep='\t')
      train<-NULL
      layers<-as.integer(porcLayers*(ncol(test)-1))
      for(k in 1:5){
        if(k!=i){
          nombreTrain<-paste(data,j,"_","5-",k,".txt",sep="")
          trainR<-read.table(nombreTrain, sep='\t')
          if(is.null(test)){
            train<-trainR
          }else{
            train<-rbind(train,trainR)
          }
        }
      }
      switch(modelo,
             H2O_v1={
               timCom<-Sys.time()
               data_comp<-compress_data_h2ov1(train,test,layers)
               timCom<-as.numeric(Sys.time())-as.numeric(timCom)
               timComTot<-timComTot+timCom
               contents<-paste(contents,'Tiempo en comprimir dataset:',timCom,"seg.\n",sep=" ")
             },
             H2O_v2={
               timCom<-Sys.time()
               data_comp<-compress_data_h2ov2(train,test,layers)
               timCom<-as.numeric(Sys.time())-as.numeric(timCom)
               timComTot<-timComTot+timCom
               contents<-paste(contents,'Tiempo en comprimir dataset:',timCom,"seg.\n",sep=" ")
             },
             autoencoders={
               timCom<-Sys.time()
               data_comp<-compress_data_autoencoder(train,test,layers)
               timCom<-as.numeric(Sys.time())-as.numeric(timCom)
               timComTot<-timComTot+timCom
               contents<-paste(contents,'Tiempo en comprimir dataset:',timCom,"seg.\n",sep=" ")
             },
             {
                contents<-paste(contents,'Modelo no válido. No se aplica compresión.',"\n",sep=" ")
                y <- colnames(train)[ncol(train)]
                train[,y] <- as.factor(train[,y])
                test[,y] <- as.factor(test[,y])
                train_res <- as.h2o(train)
                test_res <- as.h2o(test)
                data_comp<-c(train_res,test_res)
             })
      switch(clasif,
             RandomForest={
               timCla<-Sys.time()
               result<-clasif_randomForest(data_comp)
               timCla<-as.numeric(Sys.time())-as.numeric(timCla)
               timClasTot<-timClasTot+timCla
               contents<-paste(contents,'Tiempo en clasificar dataset:',timCla,"seg.\n",sep=" ")
               tableResult<-table(result$Predicted, result$Real)
               if(nrow(tableResult)!= ncol(tableResult)){
                 dif<-setdiff(colnames(tableResult),rownames(tableResult))
                 for(z in 1:length(dif)){
                   name<-dif[z]
                   newRow<-rep(0,ncol(tableResult))
                   tableResult<-rbind(tableResult,newRow)
                   rownames(tableResult)[ncol(tableResult)]<-name
                 }
                 tableResult<-as.table(tableResult)
               }
               confMatrix<-confusionMatrix(tableResult)
               accuracy<-as.numeric(confMatrix[[3]][1])
               contents<-paste(contents,'Precisión para la ejecución:',accuracy,"\n",sep=" ")
               accTot<-accTot+accuracy
               #print(accuracy)
             },
             KNN={
               timCla<-Sys.time()
               result<-clasif_KNN(data_comp)
               timCla<-as.numeric(Sys.time())-as.numeric(timCla)
               timClasTot<-timClasTot+timCla
               contents<-paste(contents,'Tiempo en clasificar dataset:',timCla,"seg.\n",sep=" ")
               tableResult<-table(result$Predicted, result$Real)
               if(nrow(tableResult)!= ncol(tableResult)){
                 dif<-setdiff(colnames(tableResult),rownames(tableResult))
                 for(z in 1:length(dif)){
                   name<-dif[z]
                   newRow<-rep(0,ncol(tableResult))
                   tableResult<-rbind(tableResult,newRow)
                   rownames(tableResult)[ncol(tableResult)]<-name
                 }
                 tableResult<-as.table(tableResult)
               }
               confMatrix<-confusionMatrix(tableResult)
               accuracy<-as.numeric(confMatrix[[3]][1])
               contents<-paste(contents,'Precisión para la ejecución:',accuracy,"\n",sep=" ")
               accTot<-accTot+accuracy
               #print(accuracy)
             },
             C4.5={
               timCla<-Sys.time()
               result<-clasif_C45(data_comp)
               timCla<-as.numeric(Sys.time())-as.numeric(timCla)
               contents<-paste(contents,'Tiempo en clasificar dataset:',timCla,"seg.\n",sep=" ")
               timClasTot<-timClasTot+timCla
               tableResult<-table(result$Predicted, result$Real)
               if(nrow(tableResult)!= ncol(tableResult)){
                 dif<-setdiff(colnames(tableResult),rownames(tableResult))
                 for(z in 1:length(dif)){
                   name<-dif[z]
                   newRow<-rep(0,ncol(tableResult))
                   tableResult<-rbind(tableResult,newRow)
                   rownames(tableResult)[ncol(tableResult)]<-name
                 }
                 tableResult<-as.table(tableResult)
               }
               confMatrix<-confusionMatrix(tableResult)
               accuracy<-as.numeric(confMatrix[[3]][1])
               contents<-paste(contents,'Precisión para la ejecución:',accuracy,"\n",sep=" ")
               accTot<-accTot+accuracy
               #print(accuracy)
             },
             {
               contents<-paste(contents,'Clasificador no válido.',"\n",sep=" ")
             })
    }
  }
  accTot<-accTot/10
  timClasTot<-timClasTot/10
  timComTot<-timComTot/10
  timTot<-as.numeric(Sys.time())-as.numeric(timTot)
  contents<-paste(contents,'----Resultados Finales----\n',sep=" ")
  contents<-paste(contents,'Tiempo medio en la compresión de datos:',timComTot,"seg.\n",sep=" ")
  contents<-paste(contents,'Tiempo medio en la clasificación:',timClasTot,"seg.\n",sep=" ")
  contents<-paste(contents,'Precisión media todas las ejecuciones:',accTot,"\n",sep=" ")
  contents<-paste(contents,'Tiempo total:',timTot,"seg.\n",sep=" ")
  nomArch<-paste(data,"_",porcLayersC,"_",modelo,"_",clasif,sep="")
  nomArch<-gsub('\\.','',nomArch)
  nomArch<-paste(nomArch,".txt",sep="")
  write(contents,nomArch)
  #cat(contents)
}