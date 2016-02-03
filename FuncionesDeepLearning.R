#Función para comprobar que un paquete esté instalado:
is.installed <- function(paquete) is.element(
  paquete, installed.packages())

# Función que construye un vector de modelos de autoencoder (1 por capa)
  get_stacked_ae_array <- function(training_data,layers,args){  
    vector <- c()
    index = 0
    for(i in 1:length(layers)){    
      index = index + 1
      ae_model <- do.call(h2o.deeplearning, 
                          modifyList(list(x=names(training_data),
                                          training_frame=training_data,
                                          autoencoder=T,
                                          hidden=layers[i]),
                                     args))
      training_data = h2o.deepfeatures(ae_model,training_data,layer=1)
      
      names(training_data) <- gsub("DF", paste0("L",index,sep=""), names(training_data)) 
      vector <- c(vector, ae_model)    
    }
    vector
  }
  
  # Función que devuelve el dataset comprimido según el modelo dado.
  apply_stacked_ae_array <- function(data,ae){
    index = 0
    for(i in 1:length(ae)){
      index = index + 1
      data = h2o.deepfeatures(ae[[i]],data,layer=1)
      names(data) <- gsub("DF", paste0("L",index,sep=""), names(data)) 
    }
    data
  }
  # Función que obtiene el dataset comprimido mediante la función deepLearning del paquete H2O 
  compress_data_h2ov1 <- function(train,test,layers){  
    train<-as.h2o(train)
    test<-as.h2o(test)
    y <- colnames(train)[ncol(train)]
    x <- setdiff(names(train), y)

    train[,y] <- as.factor(train[,y])
    test[,y] <- as.factor(test[,y])
    
    model <- h2o.deeplearning(x = x, training_frame=train, validation_frame=test,autoencoder=TRUE ,hidden =layers,input_dropout_ratio = 0.2, sparse = TRUE)
    
    layerR<-length(layers)
    
    training_data <- h2o.deepfeatures(model,train,layer=layerR)
    training_data$classLabel <- train[,y]
    test_data <- h2o.deepfeatures(model,test,layer=layerR)
    test_data$classLabel <- test[,y]
    
    result<-c(training_data,test_data)
    
    result
    
  }
  
  #Función que devuelve el dataset comprimido mediante diferentes funciones que utilizan internamente el paquete H2O.
  compress_data_h2ov2<- function(train_hex,test_hex,layers){  
    train_hex<-as.h2o(train_hex)
    test_hex<-as.h2o(test_hex)
    response<-ncol(train_hex)
    
    train <- train_hex[,-response]
    test  <- test_hex [,-response]
    train_hex[,response] <- as.factor(train_hex[,response])
    test_hex[,response] <- as.factor(test_hex [,response])
    
    args <- list(activation="Tanh", epochs=1, l1=1e-5)
    ae <- get_stacked_ae_array(train, layers, args)
    
    ## Now compress the training/testing data with this 3-stage set of AE models
    train_compressed <- apply_stacked_ae_array(train, ae)
    test_compressed <- apply_stacked_ae_array(test, ae)
    
    ## Build a simple model using these new features (compressed training data) and evaluate it on the compressed test set.
    train_w_resp <- h2o.cbind(train_compressed, train_hex[,response])
    test_w_resp <- h2o.cbind(test_compressed, test_hex[,response])
    
    result<-c(train_w_resp,test_w_resp)
    
    result
  }
  
  #Función que obtiene el modelo comprimido mediante el paquete autoencoder.
  compress_data_autoencoder<- function(train,test,layers){
    y <- colnames(train)[ncol(train)]
    x <- setdiff(names(train), y)
    
    train[,y] <- as.factor(train[,y])
    test[,y] <- as.factor(test[,y])
    
    train_aut <- train[,x]
    test_aut <- test[,x]
    
    train_aut <- data.matrix(train_aut)
    test_aut <- data.matrix(test_aut)
    nl1=length(layers)+2
    
    model1<-autoencode(X.train=train_aut,nl=nl1,N.hidden=layers, unit.type="tanh", lambda= 1e-05,beta = 1, rho = 0.99, epsilon = 1e-08, max.iteration=100, rescale.flag=TRUE,rescaling.offset=0.001)
    
    train_comp<-predict(model1,X.input=train_aut,hidden.output=TRUE)
    train_comp<-train_comp$X.output
    train_comp <- data.frame(train_comp)
    
    test_comp<-predict(model1,X.input=test_aut,hidden.output=TRUE)
    test_comp<-test_comp$X.output
    test_comp<-data.frame(test_comp)
    
    train_comp$classLabel <- train[,y]
    test_comp$classLabel <- test[,y]
    
    train_res <- as.h2o(train_comp)
    test_res <- as.h2o(test_comp)
    
    result<-c(train_res,test_res)
    
    result
    
  }
  #Función que aplica el clasificador RandomForest y devuelve la matriz de confusión.
  clasif_randomForest<- function(data_comp){
    train<-data_comp[[1]]
    test<-data_comp[[2]]
    
    colnames(train)[ncol(train)]<-"classLabel"
    colnames(test)[ncol(test)]<-"classLabel"
    y1 <-"classLabel"
    x1 <- setdiff(names(train), "classLabel")
    
    model<-h2o.randomForest(x=x1,y=y1,training_frame=train, validation_frame=test)
    
    predictions<-h2o.predict(object= model, newdata= test)
    
    train<-as.data.frame(train)
    test<-as.data.frame(test)
    
    predict<-factor(as.vector(predictions[1]))
    
    result <- data.frame(
      Real = test$classLabel, 
      Predicted = levels(train$classLabel)[predict])
    
    result
    
  }
  
  #Función que aplica el clasificador C4.5 y devuelve la matriz de confusión.
  clasif_C45<- function(data_comp){
    train<-data_comp[[1]]
    test<-data_comp[[2]]
    
    train<-as.data.frame(train)
    test<-as.data.frame(test)
    
    colnames(train)[ncol(train)]<-"classLabel"
    colnames(test)[ncol(test)]<-"classLabel"
    x1 <- setdiff(names(train), "classLabel")
    
    model<-J48(classLabel~.,data=train)
    
    predictions <- predict(model, test[,x1])
    
    result <- data.frame(
      Real = test$classLabel, 
      Predicted = levels(train$classLabel)[predictions])
    
    result
    
  }
  
  #Función que aplica el clasificador KNN y devuelve la matriz de confusión.
  clasif_KNN<- function(data_comp){
    train<-data_comp[[1]]
    test<-data_comp[[2]]
    
    train<-as.data.frame(train)
    test<-as.data.frame(test)
    
    colnames(train)[ncol(train)]<-"classLabel"
    colnames(test)[ncol(test)]<-"classLabel"
    x1 <- setdiff(names(train), "classLabel")
    
    model<-kknn(classLabel~.,train,test)
    
    result <- data.frame(
      Real = test$classLabel, 
      Predicted = levels(train$classLabel)[model$fitted.values])
    
    result
    
  }
  
  
  
  