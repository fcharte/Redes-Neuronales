#Ejecuciones programadas
#Carga de funciones utilizadas:
source("~/Documents/RepositoriosGit/Redes-Neuronales/FuncionesDeepLearning.R")
source("~/Documents/RepositoriosGit/Redes-Neuronales/DeepLearning_Classification.R")

#1- Carga de los paquetes necesarios:
if(!is.installed('autoencoder'))
  install.packages('autoencoder')
library('autoencoder')

if(!is.installed('h2o'))
  install.packages('h2o')
library('h2o')
h2o.init (nthreads = -1)

if(!is.installed('RWeka'))
  install.packages('RWeka')
library(RWeka)

if(!is.installed('caret'))
  install.packages('caret')
library(caret)

if(!is.installed('kknn'))
  install.packages('kknn')
library(kknn)

#2- Establecer working directory.
setwd("~/Documents/RepositoriosGit/Redes-Neuronales/Dataset")

#3- Ejecuciones 
dataset<-c("coil2000","letter","MNIST","madelon","gisette","arcene")
layers<-list(c(1.5,0.1),c(1.5,0.15),c(1.5,0.2),c(1.5,0.5,0.1),c(1.5,0.5,0.15),c(1.5,0.5,0.2),c(1.5,0.5,0.3,0.1),c(1.5,0.5,0.3,0.15),c(1.5,0.5,0.3,0.2))
model<-c("H2O_v1","H2O_v2","autoencoders")
clasif<-c("RandomForest","KNN","C4.5")

for(i in 1:length(dataset)){
  for(j in 1:length(layers)){
    for(k in 1:length(model)){
      for(z in 1:length(clasif)){
        deepLearning_classification(dataset[i],layers[[j]],model[k],clasif[z])
      }
    }
  }
}

