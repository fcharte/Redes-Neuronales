#Función para realizar las particiones necesarias para el proceso 2-5CV
#Mediante un fichero de entrenamiento, test, un fichero con las clases de test y el nombre final
do_partitions <- function(train,test,testC,name){  
  train<-read.table(train)
  test<-read.table(test)
  testR<-read.table(testC)
  
  y<-colnames(train)[ncol(train)]
  
  names(testR)[1]<-y
  test<-cbind(test,testR)
  
  train<-rbind(train,test)
  
  for(j in 1:2){
    indicesTotal<-1:nrow(train)
    nTraining <- as.integer(nrow(train)*.20)
    for(i in 1:5){
      nombreArc<-paste(name,j,"_","5-",i,".txt",sep="")
      if(i!=5){
        indices <- sample(indicesTotal,nTraining)
      }else{
        indices <- indicesTotal
      }
      indicesTotal <- indicesTotal[-match(indices,indicesTotal)]
      
      data<-train[indices,]
      
      write.table(data, nombreArc, sep='\t')
    }
  }
  
}

#Función para realizar las particiones necesarias para el proceso 2-5CV
#Mediante un fichero de entrenamiento, test, el nombre final y si tiene o no cabecera (ficheros .csv)
do_partitions_csv <- function(train,test,name,head){  
  train<-read.csv(train,header=head)
  test<-read.csv(test,header=head)
  
  train<-rbind(train,test)
  
  for(j in 1:2){
    indicesTotal<-1:nrow(train)
    nTraining <- as.integer(nrow(train)*.20)
    for(i in 1:5){
      nombreArc<-paste(name,j,"_","5-",i,".txt",sep="")
      if(i!=5){
        indices <- sample(indicesTotal,nTraining)
      }else{
        indices <- indicesTotal
      }
      indicesTotal <- indicesTotal[-match(indices,indicesTotal)]
      
      data<-train[indices,]
      
      write.table(data, nombreArc, sep='\t')
    }
  }
  
}

#Función para realizar las particiones necesarias para el proceso 2-5CV
#Mediante un fichero de datos, el nombre final y el separador utilizado.
do_partitions1 <- function(train,name,separ){  
  train<-read.table(train,sep=separ)
  
  for(j in 1:2){
    indicesTotal<-1:nrow(train)
    nTraining <- as.integer(nrow(train)*.20)
    for(i in 1:5){
      nombreArc<-paste(name,j,"_","5-",i,".txt",sep="")
      if(i!=5){
        indices <- sample(indicesTotal,nTraining)
      }else{
        indices <- indicesTotal
      }
      indicesTotal <- indicesTotal[-match(indices,indicesTotal)]
      
      data<-train[indices,]
      
      write.table(data, nombreArc, sep='\t')
    }
  }
  
}


#Función para realizar las particiones necesarias para el proceso 2-5CV
#Mediante un fichero de datos, un fichero con las clases correspondientes y el nombre final.
do_partitions2 <- function(train,trainC,name){  
  train<-read.table(train)
  trainR<-read.table(trainC)
  
  names(trainR)[1]<-"class"
  
  train<-cbind(train,trainR)
  
  for(j in 1:2){
    indicesTotal<-1:nrow(train)
    nTraining <- as.integer(nrow(train)*.20)
    for(i in 1:5){
      nombreArc<-paste(name,j,"_","5-",i,".txt",sep="")
      if(i!=5){
        indices <- sample(indicesTotal,nTraining)
      }else{
        indices <- indicesTotal
      }
      indicesTotal <- indicesTotal[-match(indices,indicesTotal)]
      
      data<-train[indices,]
      
      write.table(data, nombreArc, sep='\t')
    }
  }
  
}
