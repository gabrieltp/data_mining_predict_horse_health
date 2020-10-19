rm(list = ls())   #limpa o workspace
cat("\014")       #limpa o console

#install.packages("caret")
#install.packages("gtools")
#install.packages("rlist")
#install.packages("remotes")
#install.packages("forcats")
#install.packages("reshape2")
#install.packages("tree")
#install.packages("randomForest")
#install.packages("e1071")
#remotes::install_github("vqv/ggbiplot")
library(caret)
library(gtools)
library(rlist)
library(ggbiplot)
library(forcats)
library(reshape2)
library(rpart)
library(tree)
library(randomForest)
library(e1071)
library(class)

#setar o diretório de trabalho
setwd("C:\\Users\\GabrielTaranto\\Desktop\\Trabalho_DM")

#carregar a base
db_horses_train <- file.path("C:\\Users\\GabrielTaranto\\Desktop\\Trabalho_DM\\horse.csv")
db_horses_test <- file.path("C:\\Users\\GabrielTaranto\\Desktop\\Trabalho_DM\\horseTest.csv")
horses_train = read.table(db_horses_train, header = TRUE, sep=",")
horses_test = read.table(db_horses_test, header = TRUE, sep=",")

summary(horses_train)
summary(horses_test)

####Análise exploratória, missing values e atributos desnecessários####
#boxplot para verificar outliers
keeps <- c("hospital_number", "rectal_temp", "pulse", "respiratory_rate", 
           "packed_cell_volume", "total_protein", "abdomo_protein", 
           "lesion_1", "lesion_2", "lesion_3", "outcome")
horses_train_box = horses_train[keeps]

summary(horses_train_box)

horses_train_box.m <- melt(horses_train_box, na.rm = TRUE, id.var = "outcome")
p <- ggplot(data = horses_train_box.m, aes(x=variable, y=value)) + 
  geom_boxplot(aes(fill=outcome))
p + facet_wrap( ~ variable, scales="free")

#remover outliers
outliers <- boxplot(horses_train$hospital_number, plot=FALSE)$out
outliers2 <- boxplot(horses_train$rectal_temp, plot=FALSE)$out
outliers3 <- boxplot(horses_train$respiratory_rate, plot=FALSE)$out
outliers4 <- boxplot(horses_train$abdomo_protein, plot=FALSE)$out
outliers5 <- boxplot(horses_train$lesion_1, plot=FALSE)$out


horses_train_box[which(horses_train_box$hospital_number %in% outliers),]
horses_train_box[which(horses_train_box$rectal_temp %in% outliers2),]
horses_train_box[which(horses_train_box$respiratory_rate %in% outliers3),]
horses_train_box[which(horses_train_box$abdomo_protein %in% outliers4),]
horses_train_box[which(horses_train_box$lesion_1 %in% outliers5),]

horses_train_box <- horses_train_box[-which(horses_train_box$hospital_number %in% outliers),]
horses_train_box <- horses_train_box[-which(horses_train_box$rectal_temp %in% outliers2),]
horses_train_box <- horses_train_box[-which(horses_train_box$respiratory_rate %in% outliers3),]
horses_train_box <- horses_train_box[-which(horses_train_box$abdomo_protein %in% outliers4),]
horses_train_box <- horses_train_box[-which(horses_train_box$lesion_1 %in% outliers5),]

boxplot(horses_train_box$hospital_number)
boxplot(horses_train_box$rectal_temp)
boxplot(horses_train_box$respiratory_rate)
boxplot(horses_train_box$abdomo_protein)
boxplot(horses_train_box$lesion_1)

horses_train_box.m <- melt(horses_train_box, na.rm = TRUE, id.var = "outcome")
p <- ggplot(data = horses_train_box.m, aes(x=variable, y=value)) + 
  geom_boxplot(aes(fill=outcome))
p + facet_wrap( ~ variable, scales="free")

#aplicar para toda a base de treino
horses_train <- horses_train[-which(horses_train$hospital_number %in% outliers),]
horses_train <- horses_train[-which(horses_train$rectal_temp %in% outliers2),]
horses_train <- horses_train[-which(horses_train$respiratory_rate %in% outliers3),]
horses_train <- horses_train[-which(horses_train$abdomo_protein %in% outliers4),]
horses_train <- horses_train[-which(horses_train$lesion_1 %in% outliers5),]

boxplot(horses_train$hospital_number)
boxplot(horses_train$rectal_temp)
boxplot(horses_train$respiratory_rate)
boxplot(horses_train$abdomo_protein)
boxplot(horses_train$lesion_1)


#retornar o nome das colunas com missing
list_na <- colnames(horses_train)[ apply(horses_train, 2, anyNA) ]
list_na

#subsitituir os missing values dos atributos numéricos
missingModel <- preProcess(horses_train, "medianImpute")
horses_train <- predict(missingModel, horses_train)
summary(horses_train)

horses_test <- predict(missingModel, horses_test)
summary(horses_test)

list_na <- colnames(horses_train)[ apply(horses_train, 2, anyNA) ]
list_na


#substituir o mais frequente nos atributos categóricos
for (i in list_na){
  horses_train[i] = na.replace(horses_train[i],names(which.max(table(horses_train[i]))))
}

for (j in list_na){
  horses_test[j] = na.replace(horses_test[j],names(which.max(table(horses_train[j]))))
}

summary(horses_train)
summary(horses_test)

#deletar colunas não importantes
drop <- c("nasogastric_reflux_ph", "cp_data", "lesion_3")
horses_train = horses_train[,!(names(horses_train) %in% drop)]
summary(horses_train)

horses_test = horses_test[,!(names(horses_test) %in% drop)]
summary(horses_test)



####Balanceamento dos Dados e processo para gerar as duas bases tratadas####
#substituir euthanized por died
old_value <- "euthanized"
new_value <- "died"

for (k in seq_along(horses_train$outcome)) {
  horses_train$outcome[[k]][horses_train$outcome[[k]] %in% old_value] <- new_value
}

for (l in seq_along(horses_test$outcome)) {
  horses_test$outcome[[l]][horses_test$outcome[[l]] %in% old_value] <- new_value
}

#deleta o level euthanized que não está sendo usado
levels(droplevels(horses_train$outcome))
horses_train$outcome <- factor(horses_train$outcome)

levels(droplevels(horses_test$outcome))
horses_test$outcome <- factor(horses_test$outcome)

summary(horses_train)
summary(horses_test)

#normalizar os dados
normalizedTrain <- preProcess(horses_train, method = "range")
horses_train_normal <- predict(normalizedTrain, horses_train)

horses_test_normal <- predict(normalizedTrain, horses_test)

summary(horses_train_normal)
summary(horses_test_normal)

####Qui Quadrado e PCA####
#testar quais atributos numéricos são importantes para o outcome
keeps <- c("hospital_number", "rectal_temp", "pulse", "respiratory_rate", 
           "packed_cell_volume", "total_protein", "abdomo_protein", 
           "lesion_1", "lesion_2", "outcome")
horses_train_chi = horses_train_normal[keeps]

list_values_tot <- list()
for (m in seq_along(horses_train_chi)){
  list_values_tot = list.append(list_values_tot, 
                                chisq.test(horses_train_chi[m], 
                                           horses_train_chi$outcome, correct=FALSE))
  if (list_values_tot[[m]]["p.value"]<0.05){
    cat("Influencia na saída!\n")
    
  }
  if (list_values_tot[[m]]["p.value"]>0.05){
    cat("Não influencia na saída! Verificar o atributo ", m, "\n")
  }
  if (all(list_values_tot[[m]]["p.value"])==0){
    cat("O atributo contem somente zeros.\n")
  } 
}


#PCA para rankear os atributos
keeps_pca <- c("hospital_number", "rectal_temp", "pulse", "respiratory_rate", 
               "packed_cell_volume", "total_protein", "abdomo_protein", 
               "lesion_1", "lesion_2")
horse_train_num_normal = horses_train_normal[keeps_pca]
horses_train_num_normal.pca <- prcomp(horse_train_num_normal, center = TRUE, scale. = TRUE)
summary(horses_train_num_normal.pca)

ggbiplot(horses_train_num_normal.pca)

#remover atributos que não influenciam na saída
drop <- c("lesion_1", "lesion_2")
horses_train_normal = horses_train_normal[,!(names(horses_train_normal) %in% drop)]
horses_test_normal = horses_test_normal[,!(names(horses_test_normal) %in% drop)]

summary(horses_train_normal)
summary(horses_test_normal)

####Testes de diferentes modelos####

#Decision Tree
system.time(tree_model <- tree(outcome ~., horses_train_normal))
predictionsDTree <- predict(tree_model, horses_test_normal, type="class")
table(predictionsDTree, horses_test_normal$outcome)
accuracy = 1 - mean(predictionsDTree != horses_test_normal$outcome)
accuracy

summary(tree_model)
plot(tree_model)
text(tree_model)

ggplot(data=horses_train_normal, aes(x = capillary_refill_time, fill = outcome)) + geom_bar()

#SVM
system.time(svm_model <- svm(outcome ~., horses_train_normal, probability = T))
predictionsSVM <- predict(svm_model, horses_test_normal, probability = T)
table(predictionsSVM, horses_test_normal$outcome)
accuracy = 1 - mean(predictionsSVM != horses_test_normal$outcome)
accuracy

summary(svm_model)

probabilidades = attr(predictionsSVM, "probabilities")
predictionsAndProbabilities = cbind(horses_test_normal$outcome, predictionsSVM, 
                                    probabilidades)
View(predictionsAndProbabilities)

#Random Forest
system.time(forest_model <- randomForest(outcome ~., data = horses_train_normal,
                                         mtry = 8, importance = TRUE, 
                                         do.trace = 100))
predictionsForest = predict(forest_model, horses_test_normal)
table(predictionsForest, horses_test_normal$outcome)
accuracy = 1 - mean(predictionsForest != horses_test_normal$outcome)
accuracy

plot(forest_model)

forest_model
varImpPlot(forest_model)

#KNN
drop <- c("outcome")
horses_train_knn = horses_train_normal[,!(names(horses_train_normal) %in% drop)]
horses_test_knn = horses_test_normal[,!(names(horses_test_normal) %in% drop)]

#transformando os atributos categóricos em numéricos
classesTrain = lapply(horses_train_knn, class)
for (n in seq_along(classesTrain)){
  if (classesTrain[[n]] == "factor"){
    horses_train_knn[n] =  as.numeric(unlist(horses_train_knn[n])) 
  }
}

classesTest = lapply(horses_test_knn, class)
for (n in seq_along(classesTest)){
  if (classesTest[[n]] == "factor"){
    horses_test_knn[n] =  as.numeric(unlist(horses_test_knn[n])) 
  }
}


system.time(knn_model <- knn(horses_train_knn, 
                             horses_test_knn,
                 cl = horses_train_normal$outcome, k = 5))

table(knn_model, horses_test_normal$outcome)
accuracy = 1 - mean(knn_model != horses_test_normal$outcome)
accuracy
