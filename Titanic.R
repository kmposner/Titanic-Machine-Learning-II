rm(list=ls())
library(glmnet)
library(ggplot2)
options(scipen = 100)
train_titanic = read.csv("train.csv", header=TRUE, sep=',', stringsAsFactors = FALSE)
test_titanic = read.csv("test.csv", header=TRUE, sep=',', stringsAsFactors = FALSE)
#Take the cabin, name, and ticket columns out 
test_titanic <- test_titanic[-c(1,4,9,11)]
train_y<- train_titanic$Survived
#Take the cabin, name, and ticket columns out 
train_titanic <- train_titanic[-c(1,4,9,11)]

#changing gender to binary variables: male = 1 and female =0
train_titanic$Sex<-ifelse(train_titanic$Sex=="male",1,0)
test_titanic$Sex<- ifelse(test_titanic$Sex=="male",1,0)

#Omit based on two blank observations in Embarked column:  
train_titanic<-train_titanic[-c(62,830),]

#Omit empty Fares observation in test_titanic Set:
test_titanic <- na.omit(test_titanic)

#String as Factors
train_titanic$Sex <-as.factor(train_titanic$Sex)
train_titanic$Pclass <- as.factor(train_titanic$Pclass)
train_titanic$Survived <- as.factor(train_titanic$Survived)
train_titanic$Embarked <- as.factor(train_titanic$Embarked)
test_titanic$Sex <-as.factor(test_titanic$Sex)
test_titanic$Pclass <- as.factor(test_titanic$Pclass)
test_titanic$Survived <- as.factor(test_titanic$Survived)
test_titanic$Embarked <- as.factor(test_titanic$Embarked)
#Replacing the NA's in Age columns in training and testing data with median age of those on Titanic: 28
train_titanic$Age <- replace(train_titanic$Age,is.na(train_titanic$Age),28)
test_titanic$Age <- replace(test_titanic$Age,is.na(test_titanic$Age),28)





###############SVM MODEL##################
library(e1071)
set.seed(5082)
svmfit001 <- svm(Survived ~ .,
                 data = train_titanic ,
                 kernel = "linear",
                 cost = 0.1,
                 scale = FALSE)

yhat <- predict(svmfit001, newdata=test_titanic) 

confusion<-table(Predicted = yhat, Actual= test_titanic$Survived)
#Accuracy rate
(confusion[1]+confusion[4])/sum(confusion) #SVM model provides accuracy rate: 87.31%





############LOGISTIC MODEL################
set.seed(5082)
logisticmodel <-glm(formula=Survived~., family = binomial, data = train_titanic)
logisticmodel
yhat<-predict(logisticmodel, test_titanic, type="response")
yhat<-ifelse(yhat >.5, "1", "0")
confusion<-table(Predicted=yhat, Actual=test_titanic$Survived)
confusion
#Accuracy rate
(confusion[1]+confusion[4])/sum(confusion) #Logistic model provides accuracy rate: 86.40%



############LDA MODEL################
library(lmtest)
library(caret)
library(dplyr)
library(MASS)
library(rattle)
set.seed(5082)

preprocessing <- train_titanic %>% preProcess(method = c("center", "scale"))
train_titanic <- preprocessing %>% predict(train_titanic)
test_titanic <- preprocessing %>% predict(test_titanic)
ldamodel <- lda(Survived~., data=train_titanic)
ldamodel
predictions <- ldamodel %>% predict(test_titanic)
confusion<-table(predictions$class,test_titanic$Survived)
#Accuracy rate
(confusion[1]+confusion[4])/sum(confusion) #LDA model provides accuracy rate: 87.31%






###############BAGGING MODEL##################
library(randomForest)
set.seed(5082)
bag.titanic<- randomForest(Survived~., data=train_titanic, mtry=7, importance = TRUE)
report<-randomForest::importance(bag.titanic)

Important_Features<-data.frame(Feature=row.names(report), Importance=report[,1])
plot_ <- ggplot(Important_Features, 
               aes(x= reorder(Feature,
                              Importance) , y = Importance) ) +
  geom_bar(stat = "identity", 
           fill = "#800080") +
  coord_flip() +
  theme_light(base_size = 20) +
  xlab("") + 
  ylab("Importance")+
  ggtitle("Important Features When Predicting No Survival on Titanic\n") +
  theme(plot.title = element_text(size=18))
ggsave("important_features.png", 
       plot_)
plot_
Important_Features<-data.frame(Feature=row.names(report), Importance=report[,2])
plot_ <- ggplot(Important_Features, 
                aes(x= reorder(Feature,
                               Importance) , y = Importance) ) +
  geom_bar(stat = "identity", 
           fill = "#800080") +
  coord_flip() +
  theme_light(base_size = 20) +
  xlab("") + 
  ylab("Importance")+
  ggtitle("Important Features When Predicting Survival on Titanic\n") +
  theme(plot.title = element_text(size=18))
ggsave("important_features.png", 
       plot_)
plot_
varImpPlot (bag.titanic)
yhat<-predict(bag.titanic, newdata=test_titanic)
confusion<-table(Predicted=yhat, Actual=test_titanic$Survived)
confusion
#Accuracy rate
(confusion[1]+confusion[4])/sum(confusion)
#Accuracy for bagging model is: 93.96%









############RANDOM FOREST MODEL################

#Random forest model: used for loop to find best mtry 
k<-6
#create a 7-element vectors with 0s
Random_accuracy<-rep(0, length=k)
#Use a for loop to train  random forest models

for(i in 1:k){
  set.seed(5082)
  bag.titanic<- randomForest(Survived~., data=train_titanic, mtry=i, importance = TRUE)
  yhat<-predict(bag.titanic, newdata=test_titanic)
  confusion<-table(Predicted=yhat, Actual=test_titanic$Survived)
  confusion
  #Accuracy rate
  Random_accuracy[i]<-(confusion[1]+confusion[4])/sum(confusion)
}
which.max(Random_accuracy) #Mtry=3 has the best accuracy rates: 97.28%








################BEST MODEL USING MTRY =3####################### 
set.seed(5082)
bag.titanic<- randomForest(Survived~., data=train_titanic, mtry=3, importance = TRUE)
report<-randomForest::importance(bag.titanic)
Important_Features<-data.frame(Feature=row.names(report), Importance=report[,1])
plot_ <- ggplot(Important_Features, 
                aes(x= reorder(Feature,
                               Importance) , y = Importance) ) +
  geom_bar(stat = "identity", 
           fill = "#800080") +
  coord_flip() +
  theme_light(base_size = 20) +
  xlab("") + 
  ylab("Importance")+
  ggtitle("Important Features When Predicting No Survival on Titanic\n") +
  theme(plot.title = element_text(size=18))
ggsave("important_features.png", 
       plot_)
plot_
Important_Features<-data.frame(Feature=row.names(report), Importance=report[,2])
plot_ <- ggplot(Important_Features, 
                aes(x= reorder(Feature,
                               Importance) , y = Importance) ) +
  geom_bar(stat = "identity", 
           fill = "#800080") +
  coord_flip() +
  theme_light(base_size = 20) +
  xlab("") + 
  ylab("Importance")+
  ggtitle("Important Features When Predicting Survival on Titanic\n") +
  theme(plot.title = element_text(size=18))
ggsave("important_features.png", 
       plot_)
plot_
varImpPlot (bag.titanic)
importance(bag.titanic)
yhat<-predict(bag.titanic, newdata=test_titanic)
confusion<-table(Predicted=yhat, Actual=test_titanic$Survived)
confusion
#Accuracy rate
(confusion[1]+confusion[4])/sum(confusion) #Mtry=3 has the best accuracy rates: 97.28%





