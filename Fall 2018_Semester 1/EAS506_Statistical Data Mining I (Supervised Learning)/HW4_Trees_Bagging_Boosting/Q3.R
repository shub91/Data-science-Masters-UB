###########################################################
##
## This code is the solution for Question 3 of Homework 4
## 
## 
##
## Created: November 24, 2018
## Name: Shubham Sharma
###########################################################

#######################################
# Loading libraries and attaching data 
#######################################

rm(list = ls())

library(MASS)
library(gbm)
library(randomForest)
library(ISLR)
library(class)
library(dplyr)
library(randomForest)
library(adabag)

#######################################
# Loading data 
#######################################

data(Boston)

set.seed(12345)

B_fl=rbind((filter(Boston,Boston$crim>median(Boston$crim))) %>% mutate(flag='1'),
          (filter(Boston,Boston$crim>median(Boston$crim))) %>% mutate(flag='0'))
B_fl=B_fl[,-1]
set.seed(12345)
train=sample(seq(1,nrow(B_fl)),0.7*nrow(B_fl))
btrain = B_fl[train,]
test = B_fl[-train,]
model_control=rpart.control(minsplit=2,xval=10,cp=0.01)
b_fit=rpart(flag~.,data=B_fl,method='class',control = model_control)

#######################################
# Pruning Tree
#######################################

min_cp=which.min(b_fit$cptable[,4])
pruned_fit=prune(b_fit,cp=min_cp)

#######################################
# Random Forest
#######################################

boston.rf.fit=randomForest(as.factor(flag)~.,data=B_fl[train,],n.tree=1000)
pred=predict(boston.rf.fit,B_fl[-train,])
te.error=(1/length(B_fl[-train,14]))*sum((as.numeric(pred)-as.numeric(B_fl[-train,14]))^2) # 1.8816

#######################################
# Boosting
#######################################

b1 = gbm (flag~., data =B_fl[train,], n.trees = 5000, shrinkage = .1, interaction.depth = 3, distribution = "adaboost")
b2 = gbm (flag~., data = B_fl[train,], n.trees = 5000, shrinkage = .7, interaction.depth = 3, distribution = "adaboost")

quartz()
par(mfrow = c(1,2))
summary(b1)
title('Summary with shrinkage = 0.1')
summary(b2)
title('Summary with shrinkage = 0.7')

## Error for shrinkage = 0.1
y_hat1 = predict(b1, newdata =B_fl[-train,], n.trees = 1000, type="response")
err1 = sum(abs(as.numeric(y_hat1) - as.numeric(B_fl[-train,14])))/length(B_fl[-train,14])
err1 #0.789

## Error for shrinkage = 0.7
y_hat2 = predict(b2, newdata = B_fl[-train,], n.trees = 1000, type="response")
error2 = sum(abs(as.numeric(y_hat2) - as.numeric(B_fl[-train,14])))/length(B_fl[-train,14])
error2 #0.828

#######################################
# Bagging
#######################################

rf.fit = randomForest(as.factor(flag)~., data = B_fl[train,], n.tree = 10000,mtry = 13)
quartz()
varImpPlot(rf.fit, main = "Gini decrease VS variables")
importance(rf.fit)

y_hat = predict(rf.fit, newdata = B_fl[-train,], type = 'response')
y_hat = as.numeric(y_hat) - 1
misclass1 = sum(abs(as.numeric(B_fl[-train,14]) - y_hat))/length(y_hat)
cat("Misclassification error rate for test set using Bagging :" ,misclass1) #0.855

set.seed(12345)

Boston$crim = ifelse(Boston$crim < median(Boston$crim), 0, 1)
Boston$crim = as.numeric(Boston$crim)

train =  sample(1:nrow(Boston), .70*nrow(Boston))
boston_train = Boston[train,]
boston_test  = Boston[-train,]

#########################################################
## Applying Logistic Regression
#########################################################

modellog = glm(crim~ . , data = boston_train, family = "binomial")
predlog  = predict(modellog, newdata = boston_test, type = "response")

errorlog = length(which(round(predlog) != boston_test[,1])) / length(predlog) 

#########################################################
## Applying kNN
#########################################################

modelknn <- knn(boston_train,boston_test,boston_train$crim, k=5)
summary(modelknn)

table(modelknn,boston_test$crim)
mean(modelknn == boston_test$crim) 
error_knn <- mean(modelknn!=boston_test$crim)
 
