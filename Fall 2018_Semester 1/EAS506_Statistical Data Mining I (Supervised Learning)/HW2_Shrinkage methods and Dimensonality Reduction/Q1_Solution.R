###########################################################
##
## This code is the solution for Question 1 of Homework 2
## dealing with College data set
##
## Created: October 2, 2018
## Name: Shubham Sharma
###########################################################

rm(list = ls())
library(leaps)
library(glmnet)
library(ISLR)
library(plotmo)
library(pls)
############################################
# Splitting data into test set and training
# set (50:50)
############################################

str(College)
col = na.omit(College)
set.seed(12345)
trn = sample(1:nrow(col), round(0.5*nrow(col)))
trcol = na.omit(col[trn,])
tscol = na.omit(col[-trn,])

############################################
# Fitting linear model uing OLS on training
# set using response variable Apps
############################################

ls = lm(Apps~., data = trcol)
summary(ls)
prcolls = predict(ls, tscol, type = "response")
tsmse = mean((round(prcolls) - tscol[,"Apps"])^2)
# test set MSE
tsmse

############################################
# Ridge Regression
# 
############################################

trcolm = model.matrix(Apps~., data = trcol)
tscolm = model.matrix(Apps~., data = tscol)

# ridge regression model
rtrcol = glmnet(trcolm,trcol[, "Apps"],alpha = 0)
names(rtrcol)
coef(rtrcol)
dim(coef(rtrcol))

## Look at different lambdas
# 10th lamda value
rtrcol$lambda[10]
coef(rtrcol)[,10]
sqrt(sum(coef(rtrcol)[3:19, 10]^2)) #l2_norm

# 50th lamda value
rtrcol$lambda[50]
coef(rtrcol)[,50]
sqrt(sum(coef(rtrcol)[3:19, 50]^2)) #l2_norm

# 100th lamda value
rtrcol$lambda[100]
coef(rtrcol)[,100]
sqrt(sum(coef(rtrcol)[3:19, 100]^2)) #l2_norm


# Predict the model for a "new value" of lambda
predict(rtrcol, s = 100, type = "coefficient")
predict(rtrcol, s = 0.0005, type = "coefficient")

############################################
#
# Model Selection
############################################

cvtr = cv.glmnet(trcolm,trcol[, "Apps"],alpha = 0)
quartz()
plot(cvtr)
names(cvtr)
bltr = cvtr$lambda.min
bltr
rtrp = predict(rtrcol, s=bltr, type = "coefficients")
rtrp
rtrp2 = predict(rtrcol, s=bltr, newx = tscolm, type = "response")
rtrp2

y_hatr = round(rtrp2)
y_truer= tscol[,"Apps"]
tsmser = mean((y_hatr - y_truer)^2)
# test set MSE for RIDGE model
tsmser


############################################
#
# LASSO
############################################

ltrcol = glmnet(trcolm,trcol[, "Apps"], alpha = 1)
names(ltrcol)
quartz()
plot_glmnet(ltrcol, xvar = "lambda", label = 6)

## Coefficients

ltrcol$lambda[70]
coef(ltrcol)[,70]

ltrcol$lambda[50]
coef(ltrcol)[,50]

ltrcol$lambda[20]
coef(ltrcol)[,20]

## Best Lambda

lcolcv.out = cv.glmnet(trcolm,trcol[, "Apps"], alpha = 1)
lcolbestlam = lcolcv.out$lambda.min
lcolbestlam

ltrp = predict(ltrcol, s = lcolbestlam, type = "coefficients")
ltrp
ltrp2 = predict(ltrcol, s = lcolbestlam, newx = tscolm, type = "response")
ltrp2
y_hatl = round(ltrp2)
y_truel = tscol[,"Apps"]

# test set MSE for LASSO
tsmsel = mean((y_hatl-y_truel)^2)
tsmsel

############################################
#
# Principal Component Regression
############################################

setseed(2)
pcrcol = pcr(Apps~., data = trcol, scale = TRUE, validation = "CV")
summary(pcrcol)
quartz()
validationplot(pcrcol, val.type = "MSEP")

# Computing Test error

pcrtrerr = c()
pcrtserr = c()
for (i in 1:17){
  pcrtrp = round(predict(pcrcol, trcol, ncomp = i))
  pcrtsp = round(predict(pcrcol, tscol, ncomp = i))
  trerr = mean((pcrtrp-trcol[,"Apps"])^2)
  tserr = mean((pcrtsp-tscol[,"Apps"])^2)
  pcrtrerr = c(pcrtrerr, trerr)
  pcrtserr = c(pcrtserr, tserr)
}

quartz()
plot(pcrtserr, xlab = "No. of Components", ylab = "Test set MSE")
pcrtserr
############################################
#
# Partial Least Squares
############################################

plscol = plsr(Apps~., data = trcol, scale = TRUE, validation = "CV")
summary(plscol)
quartz()
validationplot(plscol, val.type = "MSEP")

# Computing Test error

plstrerr = c()
plstserr = c()
for (i in 1:17){
  plstrp = round(predict(plscol, trcol, ncomp = i))
  plstsp = round(predict(plscol, tscol, ncomp = i))
  trerr = mean((plstrp-trcol[,"Apps"])^2)
  tserr = mean((plstsp-tscol[,"Apps"])^2)
  plstrerr = c(plstrerr, trerr)
  plstserr = c(plstserr, tserr)
}

quartz()
plot(plstserr, xlab = "No. of Components", ylab = "Test set MSE", main = "PLS")
plstserr

# Computing R square

tsav = mean(tscol[,"Apps"])
ltsr2 = 1-tsmse/mean((tscol[,"Apps"]-tsav)^2) 
rtsr2 = 1-tsmser/mean((tscol[,"Apps"]-tsav)^2)
lstsr2 = 1-tsmsel/mean((tscol[,"Apps"]-tsav)^2)
pcrtsr2 = 1-pcrtserr[10]/mean((tscol[,"Apps"]-tsav)^2)
plstsr2 = 1-plstserr[6]/mean((tscol[,"Apps"]-tsav)^2)
quartz()
barplot(c(ltsr2, rtsr2, lstsr2, pcrtsr2, plstsr2), col = "blue", names.arg = c("OLS", "Ridge", "Lasso", "PCR", "PLS"), main="Test R-squared")

