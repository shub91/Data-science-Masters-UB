###########################################################
##
## This code is the solution for Question 4 of Homework 5
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
library(ISLR)
library(e1071)
data(OJ)

#######################################
# Understanding the data
#######################################

str(OJ)
summary(OJ)

#######################################################
# Splitting into Training and test set (ratio = 7:3 )
#######################################################

set.seed(345)
train = sample(seq(1:nrow(OJ)),0.7*nrow(OJ))
tro = OJ[train,]
tso = OJ[-train,]

#######################################################
# Support Vector Machine
#######################################################

costs=0.01:10

train.error.store.lin = c(cost=character(0),err=character(0))
test.error.store.lin = c(cost=character(0),err=character(0))

for (cost in costs){
  svm.linear = svm(Purchase ~ ., kernel = "linear", data = tro, cost = cost)
  pred = (predict(svm.linear,tso[,-1]))
  
  te.error=(1/length(tso[,'Purchase']))*
    sum((as.numeric(pred)-as.numeric(tso[,'Purchase']))^2)
  
  tr.error=(1/length(tro[,'Purchase']))*
    sum((as.numeric(predict(svm.linear,tro[,-1]))
         -as.numeric(tro[,'Purchase']))^2)
  
  
  train.error.store.lin=rbind(train.error.store.lin,cbind(cost,tr.error))
  test.error.store.lin=rbind(test.error.store.lin,cbind(cost,te.error))
}

#######################################################
# Plot for test error
#######################################################

quartz()
par(mfrow=c(1,2))
plot(test.error.store.lin, xlab="m values", ylab="Test Error", type = 'l')
points(which.min(test.error.store.lin), test.error.store.lin[which.min(test.error.store.lin)], col = 'green', pch = 20)
plot(train.error.store.lin, xlab="m values", ylab="Train Error", type = 'l')
points(which.min(train.error.store.lin), train.error.store.lin[which.min(train.error.store.lin)], col = 'green', pch = 20)

print(min(test.error.store.lin[,2])) # 0.1776
print(min(train.error.store.lin[,2])) #0.1669
which.min(train.error.store.lin[,2]) # 2 - 1.01
which.min(test.error.store.lin[,2]) # 1 - 0.01

#######################################################
# Radial and Polynomial Kernel
#######################################################

# Radial Kernel

train.error.store.rad=c(cost=character(0),err=character(0))
test.error.store.rad=c(cost=character(0),err=character(0))

for (cost in costs){
  svm.rad = svm(Purchase ~ ., kernel = "radial", data = tro, cost = cost)
  pred<-(predict(svm.rad,tso[,-1]))
  
  te.error=(1/length(tso[,'Purchase']))*
    sum((as.numeric(pred)-as.numeric(tso[,'Purchase']))^2)
  
  tr.error=(1/length(tro[,'Purchase']))*
    sum((as.numeric(predict(svm.rad,tro[,-1]))
         -as.numeric(tro[,'Purchase']))^2)
  
  
  train.error.store.rad=rbind(train.error.store.rad,cbind(cost,tr.error))
  test.error.store.rad=rbind(test.error.store.rad,cbind(cost,te.error))
}

quartz()
par(mfrow=c(1,2))
plot(test.error.store.rad, xlab="m values", ylab="Test Error", type = 'l')
points(which.min(test.error.store.rad), test.error.store.rad[which.min(test.error.store.rad)], col = 'green', pch = 20)
plot(train.error.store.rad, xlab="m values", ylab="Train Error", type = 'l')
points(which.min(train.error.store.rad), train.error.store.rad[which.min(train.error.store.rad)], col = 'green', pch = 20)
min(train.error.store.rad[,2]) #0.1335
min(test.error.store.rad[,2]) #0.1776
which.min(train.error.store.rad[,2]) # 8 - 7.01
which.min(test.error.store.rad[,2]) # 5 - 4.01


# Polynomial Kernel

train.error.store.quad = c(cost=character(0),err=character(0))
test.error.store.quad = c(cost=character(0),err=character(0))

for (cost in costs){
  svm.quad = svm(Purchase ~ ., kernel = "poly", degree=2, data = tro, cost = cost)
  pred = (predict(svm.quad,tso[,-1]))
  
  te.error = (1/length(tso[,'Purchase']))*
    sum((as.numeric(pred)-as.numeric(tso[,'Purchase']))^2)
  
  tr.error = (1/length(tro[,'Purchase']))*
    sum((as.numeric(predict(svm.quad,tro[,-1]))
         -as.numeric(tro[,'Purchase']))^2)
  
  train.error.store.quad = rbind(train.error.store.quad,cbind(cost,tr.error))
  test.error.store.quad = rbind(test.error.store.quad,cbind(cost,te.error))
}

quartz()
par(mfrow=c(1,2))
plot(test.error.store.quad, xlab="m values", ylab="test error", type = 'l')
points(which.min(test.error.store.quad), test.error.store.quad[which.min(test.error.store.quad)], col = 'red', pch = 20)
plot(train.error.store.quad, xlab="m values", ylab="train error", type = 'l')
points(which.min(train.error.store.quad), train.error.store.quad[which.min(train.error.store.quad)], col = 'red', pch = 20)
min(train.error.store.quad[,2]) #0.1388
min(test.error.store.quad[,2]) #0.1900
which.min(train.error.store.quad[,2]) # 10 - 9.01
which.min(test.error.store.quad[,2]) # 10 - 9.01



