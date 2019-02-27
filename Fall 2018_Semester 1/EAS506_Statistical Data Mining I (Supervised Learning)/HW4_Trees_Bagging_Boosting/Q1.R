###########################################################
##
## This code is the solution for Question 1 of Homework 4
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

library(ElemStatLearn)
library(glmnet)
library(boot)
data(prostate)
require(leaps)
library(bootstrap)

#######################################
# Understanding the data
#######################################

set.seed(12345)

str(prostate)
summary(prostate)
prosdata = prostate[, -10]

# cor( prostate[,1:8] )
# pairs( prostate[,1:9], col="violet" )
train = subset( prostate, train==TRUE )[,1:9]
test  = subset( prostate, train==FALSE )[,1:9]

#######################################
# Best Subset Selection
#######################################

bss = regsubsets( lpsa ~ . , data=train , nbest=1, nvmax = 9, really.big=FALSE, method = 'exhaustive')
my_sum = summary(bss)
my_sum$outmat

#######################################
# Plotting the Cp and BIC
#######################################

quartz()
par(mfrow = c(1,2))
# plot(my_sum$rss, xlab = "Number of Predictors", ylab = "RSS", type = "b")
# plot(my_sum$adjr2, xlab = "Number of Predictors", ylab = "Adjusted R^2", type = "b")
plot(my_sum$cp, xlab = "Number of Predictors", ylab = "Cp", type = "b")
points(which.min(my_sum$cp), my_sum$cp[which.min(my_sum$cp)], col = 'blue')
plot(my_sum$bic, xlab = "Number of Predictors", ylab = "BIC", type = "b")
points(which.min(my_sum$bic), my_sum$bic[which.min(my_sum$bic)], col = 'blue')

## Minimum value of Cp
min(my_sum$cp)
which.min(my_sum$cp)

## Minimum value of BIC
min(my_sum$bic)
which.min(my_sum$bic)

## Test Set MSE using OLS

cn = colnames(prostate, do.NULL = FALSE)
ci = coef(bss, id = 2)
newd = data.frame((train[, cn %in% names(ci)]))
newd = cbind(data.frame(newd), data.frame(lpsa = train[,9]))
p = lm(lpsa~., data = newd)
summary(p)
prp = predict(p, test, type = "response")
tsmseB = mean((prp-test[,9])^2)

cic = coef(bss, id = 7)
newdc = data.frame((train[, cn %in% names(cic)]))
newdc = cbind(data.frame(newdc), data.frame(lpsa = train[,9]))
pc = lm(lpsa~., data = newdc)
summary(pc)
prpc = predict(pc, test, type = "response")
tsmseC = mean((prpc-test[,9])^2)

# test set MSE for OLS using BIC statistic
tsmseB

# test set MSE for OLS using Cp statistic
tsmseC

#############################
## K-Fold Cross Validation
#############################

bss1 = regsubsets( lpsa ~ . , data=prostate, method = 'exhaustive')
my_sum1 = summary(bss1)

select = my_sum1$outmat
## k = 5

k5_err = c()
for (i in 1:8)
{
  temp = which(select[i,] == '*')
  cv5_dtr = prosdata[,c(temp,9)] 
  glmfit5 = glm(lpsa~., data = cv5_dtr)
  kcv5_err = cv.glm(cv5_dtr, glmfit5,K = 5)$delta[2]
  k5_err = c(k5_err,kcv5_err)
}

## k = 10

k10_err = c()
for (i in 1:8)
{
  temp = which(select[i,] == '*')
  cv10_dtr = prosdata[,c(temp,9)] 
  glmfit10 = glm(lpsa~., data = cv10_dtr)
  kcv10_err = cv.glm(cv10_dtr, glmfit10,K = 10)$delta[2]
  k10_err = c(k10_err,kcv10_err)
}

quartz()
par(mfrow = c(1,2))
plot(k5_err, xlab="Number of Predictors", ylab="Adjusted Cross Validation Error", type = 'b', main = 'K=5')
points(which.min(k5_err), k5_err[which.min(k5_err)], col = 'blue', pch = 20)
plot(k10_err, xlab="Number of Predictors", ylab="Adjusted Cross Validation Error", type = 'b', main = 'K=10')
points(which.min(k10_err), k10_err[which.min(k10_err)], col = 'blue', pch = 20)

which.min(k10_err)
which.min(k5_err)

# Adjusted Cross Validation Errors
k5_err[which.min(k5_err)]
k10_err[which.min(k10_err)]

#######################
## Bootstrap .632
#######################

x = prostate[,c(1:8)]
y = prostate[,9]

t_fit = function(x,y){lsfit(x,y)}
t_predict = function(fit,x){cbind(1,x)%*%fit$coef}

esq = function(y,y_hat){(y - y_hat)^2}

err_bootstrap = c()

for(i in 1:8)
{
  temp = which(select[i,] == '*')
  result = bootpred(x[,temp], y, nboot = 50, t_fit , t_predict, err.meas = esq)
  err_bootstrap = c(err_bootstrap, result[[3]])
}

quartz()
plot(err_bootstrap, xlab="Number of Predictors", ylab="Bootstrap Error", type = 'b')
points(which.min(err_bootstrap), err_bootstrap[which.min(err_bootstrap)], col = 'blue')

# Bootstrap estimate of Prediction error
err_bootstrap[which.min(err_bootstrap)]
