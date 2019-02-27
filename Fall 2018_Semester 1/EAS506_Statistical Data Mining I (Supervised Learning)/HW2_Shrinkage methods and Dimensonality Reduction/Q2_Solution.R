###########################################################
##
## This code is the solution for Question 2 of Homework 2
## dealing with insurance company benchmark dataset
##
## Created: October 2, 2018
## Name: Shubham Sharma
###########################################################

rm(list = ls())
library(leaps)
library(glmnet)

###########################################################
## Converting text file into data frame as well as 
## matrix
###########################################################

tr = na.omit(read.delim("ticdata2000.txt", header = FALSE, sep = "\t"))
ts = na.omit(read.delim("ticeval2000.txt", header = FALSE, sep = "\t"))
tg = na.omit(read.delim("tictgts2000.txt", header = FALSE, sep = "\t"))
trmx = as.matrix(tr[,1:85])
trmy = as.matrix(tr[,86])
tsmx = as.matrix(ts[,1:85])
tgmx = as.matrix(tg[,1])

p = lm(V86~., data=tr)
summary(p)
prp = round(predict(p, ts, type = "response"))
tsmse = mean((prp-tg[,1])^2)
# test set MSE for OLS
tsmse

sum(tr$V86) # No of customers who purchased the policy



###########################################################
## Checking with different variables about the 
## likelihood of purchase of Caravan policy
###########################################################

bp = table(tr$V82[tr$V86==1]) # boat policy
bp
ssip = table(tr$V85[tr$V86==1]) # social security insurance policy
ssip
ccp = table(tr$V47[tr$V86==1]) # Contribution car policy
ccp
cb = table(tr$V1[tr$V86==1]) # Customer Subtype
cb
quartz()
barplot(cb, xlab = "Customer Subtype", ylab = "Number of Customers")
aa = table(tr$V4[tr$V86==1]) # Average Age
aa
ai = table(tr$V42[tr$V86==1]) # Average Income
ai
li = table(tr$V55[tr$V86==1]) # Contribution to life insurance
li
fai = table(tr$V57[tr$V86==1]) # Contribution to family accident insurance
fai

###########################################################
## Forward and backward subset selection
## 
###########################################################


regfit.fwd <- regsubsets(V86~., data = tr, nvmax = 85, method = "forward")
my_sumf = summary(regfit.fwd)
par(mfrow = c(2,2))
plot(my_sumf$rss, xlab = "Number of Variables", ylab = "RSS", type = "l")
plot(my_sumf$adjr2, xlab = "Number of Variables", ylab = "Adjusted R^2", type = "l")
plot(my_sumf$cp, xlab = "Number of Variables", ylab = "Cp", type = "l")
plot(my_sumf$bic, xlab = "Number of Variables", ylab = "BIC", type = "l")


#quartz()
#par(mfrow = c(2,2))
#plot(regfit.fwd, scale = "r2")
#plot(regfit.fwd, scale = "adjr2")
#plot(regfit.fwd, scale = "Cp")
#plot(regfit.fwd, scale = "bic")

regfit.bwd <- regsubsets(V86~., data = tr, nvmax = 85, method = "backward")
my_sumb = summary(regfit.bwd)
quartz()
par(mfrow = c(2,2))
plot(my_sumb$rss, xlab = "Number of Variables", ylab = "RSS", type = "l")
plot(my_sumb$adjr2, xlab = "Number of Variables", ylab = "Adjusted R^2", type = "l")
plot(my_sumb$cp, xlab = "Number of Variables", ylab = "Cp", type = "l")
plot(my_sumb$bic, xlab = "Number of Variables", ylab = "BIC", type = "l")

which.min(my_sumf$cp) # Min Cp for Forward
which.min(my_sumb$cp) # Min Cp for Forward

which.min(my_sumf$bic) # Min BIC for Forward
which.min(my_sumb$bic) # Min BIC for Forward


coef(regfit.bwd, id = 8)
coef(regfit.fwd, id = 8)

###########################################################
## Fit a ridge regression model
## 
###########################################################

rtr = glmnet(trmx,trmy,alpha = 0)
names(rtr)
dim(coef(rtr))

## Look at different lambdas
rtr$lambda[100]
coef(rtr)[,100]
l2_norm = sqrt(sum(coef(rtr)[2:86, 100]^2))
l2_norm

# Predict the model for a "new value" of lambda
predict(rtr, s = 0.0005, type = "coefficient")

## Model Selection

cv.out = cv.glmnet(trmx, trmy, alpha = 0)
quartz()
plot(cv.out)
names(cv.out)
bestlam = cv.out$lambda.min
bestlam
ridge.pred = predict(rtr, s=bestlam, type = "coefficients")
ridge.pred
ridge.pred2 = round(predict(rtr, s=bestlam, newx = tsmx, type = "response"))
ridge.pred2

y_hat = ridge.pred2
y_true = tgmx
test_error = sum((y_hat - y_true)^2)
test_error # test set MSE

############################################
#
# LASSO
############################################

ltr = glmnet(trmx, trmy, alpha = 1)
names(ltr)
#quartz()
#plot(ltr)

## Coefficients

ltr$lambda[50]
coef(ltr)[,50]

rtr$lambda[50]
coef(rtr)[,50]

## Best Lambda

lcv.out = cv.glmnet(trmx,trmy, alpha = 1)
lbestlam = lcv.out$lambda.min

lasso.pred = predict(ltr, s = lbestlam, type = "coefficients")
lasso.pred

lasso.pred2 = round(predict(ltr, s = lbestlam, newx = tsmx, type = "response"))

ly_hat = lasso.pred2
y_true = tgmx

ltest_error = sum((ly_hat-y_true)^2) # Test set MSE
ltest_error

# Coefficients
p$coefficients # OLS
coef(regfit.bwd, id = 29)
coef(regfit.fwd, id = 23)


