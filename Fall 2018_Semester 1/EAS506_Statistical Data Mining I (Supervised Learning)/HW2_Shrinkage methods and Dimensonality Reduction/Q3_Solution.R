###########################################################
##
## This code is the solution for Question 3 of Homework 2
## 
##
## Created: October 2, 2018
## Name: Shubham Sharma
###########################################################

library(leaps)

############################################
# Generating data as a matrix
# 
############################################

set.seed(12345)
d = matrix(rnorm(20000), 1000, 20, byrow = TRUE)
b = rnorm(20)
e = rnorm(20)
b[19] = b[16] = b[15] = b[11] = b[6] = b[1] = 0 
y = d%*%b + e  ## Matrix-vector product

############################################
# Splitting the data into 100 training 
# observations and 900 test observations
############################################

set.seed(1234)
tr = sample(1:1000, 100) 
trd = d[tr,]
tsd = d[-tr,]
try = y[tr,]
tsy = y[-tr,]

############################################
# Best Subset Selection
# 
############################################

bss <- regsubsets(try~., data = data.frame(trd, try), nbest = 1, nvmax = 20, method = "exhaustive")

## Plotting the Training and Test MSE

trerrd = c()
tserrd = c()
cn = colnames(d, do.NULL = FALSE, prefix = "X")
for (i in 1:20){
  ci = coef(bss, id = i)
  bssptr = as.matrix(trd[, cn %in% names(ci)]) %*% ci[names(ci) %in% cn]
  bsspts = as.matrix(tsd[, cn %in% names(ci)]) %*% ci[names(ci) %in% cn]
  trerr = mean((try - bssptr)^2)
  trerrd = c(trerrd, trerr)
  tserr = mean((tsy - bsspts)^2)
  tserrd = c(tserrd, tserr)
}
quartz()
par(mfrow = c(1,2))
plot(trerrd, xlab = "No. of variables in the best model", ylab = "Training MSE", type = "b")
plot(tserrd, xlab = "No. of variables in the best model", ylab = "Test MSE", type = "b")

## Smallest test MSE
which.min(tserrd)

## Coefficients for the best model
coef(bss, id = 13)
