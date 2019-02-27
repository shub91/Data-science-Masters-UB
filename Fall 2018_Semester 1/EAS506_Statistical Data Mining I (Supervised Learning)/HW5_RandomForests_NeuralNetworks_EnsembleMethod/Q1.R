###########################################################
##
## This code is the solution for Question 1 of Homework 5
## 
## 
##
## Created: December 05, 2018
## Name: Shubham Sharma
###########################################################

#######################################
# Loading libraries and attaching data 
#######################################

rm(list = ls())
graphics.off()
library(ElemStatLearn)
library(class)
library(glmnet)
library(pls)
library(leaps)
library(randomForest)
library(ElemStatLearn)
library(Metrics)
data(spam)

#######################################
# Understanding the data
#######################################

str(spam)
summary(spam)
spam = spam[sample(nrow(spam)),];

#######################################################
# Splitting into Training and test set (ratio = 2:1 )
#######################################################

set.seed(12345)
tr = sample(1:nrow(spam), nrow(spam)*2/3)
trd = spam[tr, ]
tsd = spam[-tr, ]

#######################################################
# Bagging
#######################################################

rand_bag = randomForest(trd$spam~., data = trd, mtry = 57, importance = TRUE)
pred_spam = predict(rand_bag, newdata = tsd)
summary(pred_spam)
summary(tsd$spam)

#######################################################
# Random Forest
#######################################################

m_val = 1:57
OOB_err = numeric(length = length(m_val))
test_err = numeric(length = length(m_val))
for (i in seq_along(m_val)) {
  pred_rand = randomForest(trd$spam~., data = trd , mtry = i , importance = TRUE)
  pred_rand
  pred_RF = predict(pred_rand, newdata = tsd)
  
  OOB_err[i] = mean(pred_rand$err.rate);
  test_err[i] = rmse(summary(tsd$spam), summary(pred_RF));
}

#######################################################
# Plot of OOB error and test error vs m values
#######################################################
quartz()
par(mfrow=c(1,2))
plot(m_val,OOB_err, ann = TRUE, type = "l", xlab = "Number of randomly selected inputs for each tree", ylab = "OOB Error")
points(which.min(OOB_err), OOB_err[which.min(OOB_err)], col = 'green', pch = 20, type = 'o')
plot(m_val,test_err, ann = TRUE, type = "l", ylab= "Test Error",xlab = "Number of randomly selected inputs for each tree")
points(which.min(test_err), test_err[which.min(test_err)], col = 'green', pch = 20, type = 'o')

## Minimum Test set error
test_err[which.min(test_err)] # 26

## m value for min test set error
which.min(test_err) # 52

## Minimum OOB error
OOB_err[which.min(OOB_err)] # 0.0553

## m value for min OOB set error
which.min(OOB_err) # 6
