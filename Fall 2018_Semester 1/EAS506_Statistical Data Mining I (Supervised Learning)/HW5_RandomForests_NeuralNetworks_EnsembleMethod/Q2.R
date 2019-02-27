###########################################################
##
## This code is the solution for Question 2 of Homework 5
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

library(neuralnet)
# ls("package:neuralnet")

library(ElemStatLearn)
data(spam)
d = spam
#######################################
# Understanding the data
#######################################

str(spam)
summary(spam)

#######################################################
# Splitting into Training and test set (ratio = 2:1 )
#######################################################

spam$spam = ifelse(spam$spam == "spam",1,0)
set.seed(12345)
tr = sample(1:nrow(spam), nrow(spam)*2/3)
trd = spam[tr, ]
tsd = spam[-tr, ]
nam = names(trd)
f = as.formula(paste("spam ~", paste(nam[!nam %in% "spam"], collapse = " + ")))

#######################################################
# Fitting the neural network model
#######################################################

nn_err = c() # to store errors for different number of neurons
# Cross-validation to determine the number of neurons in the hidden layer
for(i in 1:8){
  nn = neuralnet(f, data = data.matrix(trd), hidden = i, threshold = 0.15, err.fct = 'ce', linear.output = FALSE)
  nn_pred = round(compute(nn,tsd[,1:57])$net.result[,1])
  gen_err = sum(abs(tsd$spam - nn_pred))/length(nn_pred)
  nn_err = c(nn_err,gen_err)
}

###########################################################################
# Plot for generalisation error vs. Number of neurons in the hidden layer 
###########################################################################

quartz()
plot(nn_err, xlab = "Number of neurons in the hidden layer", ylab="Generalisation(Test set) Error", type = 'l')
points(which.min(nn_err), nn_err[which.min(nn_err)], col = 'green', pch = 20, type = 'o')

## Minimum Test set error
nn_err[which.min(nn_err)]

## Number of neurons for minimum test set error 
which.min(nn_err)

nn_4 = neuralnet(f, data = data.matrix(trd), hidden = 4, threshold = 0.15, err.fct = 'ce', linear.output = FALSE)
quartz()
plot(nn_4)
