###########################################################
##
## This code is the solution for Question 1 of Homework 3
## 
## 
##
## Created: October 28, 2018
## Name: Shubham Sharma
###########################################################

#######################################
# Loading libraries and attaching data 
#######################################

rm(list = ls())
require(MASS)
attach(Boston)
library(klaR)
library(caret)
library(class)
summary(Boston)
?Boston
str(Boston)

# New Qualitative Response Variable for Crime Rate
Boston$rv = ifelse(Boston$crim > median(Boston$crim), 1, 0) # More than median is 1
#Boston$rv = factor(Boston$rv)
my_Boston = Boston[,-1]
cor(my_Boston[,1:13])

###################################
# Create Test and training dataset
###################################

set.seed(12345)
train = sample(1:nrow(my_Boston), nrow(my_Boston)*0.75)
btr = my_Boston[train, ]
bts = my_Boston[-train, ]

###################################
# Linear Discriminant Analysis
###################################

lda.fit = lda(rv~., data = btr)
lda.ptr = predict(lda.fit, newdata = btr)
rv_hat_tr = as.numeric(lda.ptr$class)-1
lda.pts = predict(lda.fit, newdata = bts)
rv_hat_ts = as.numeric(lda.pts$class)-1

mean(lda.ptr$class == btr$rv) # Accuracy

# Compute the test and training error

lda_trerr = sum(abs(btr$rv - rv_hat_tr))/length(btr$rv) # 0.133
lda_tserr = sum(abs(bts$rv - rv_hat_ts))/length(bts$rv) # 0.165

conf = confusionMatrix(as.factor(rv_hat_ts), as.factor(bts$rv))
conf$table
conf

###################################
# Quadratic Discriminant Analysis
###################################

qda.fit = qda(rv~., data = btr)
qda.ptr = predict(qda.fit, newdata = btr)
rv_hat_tr = as.numeric(qda.ptr$class)-1
qda.pts = predict(qda.fit, newdata = bts)
rv_hat_ts = as.numeric(qda.pts$class)-1

# Compute the test and training error

qda_trerr = sum(abs(btr$rv - rv_hat_tr))/length(btr$rv) # 0.0897
qda_tserr = sum(abs(bts$rv - rv_hat_ts))/length(bts$rv) # 0.126

table(lda.pts$class, bts$rv) # Confusion Matrix for QDA

###################################
# Logistic Regression
###################################

glm.fit = glm(rv~., data = btr, family = "binomial")
summary(glm.fit)

# Prediction

glm.prtr = predict(glm.fit, type = "response", newdata = btr)
rv_hat_tr = round(glm.prtr) # Converting Posterior probabilities to 0 and 1
glm.prts = predict(glm.fit, type = "response", newdata = bts)
rv_hat_ts = round(glm.prts)

# Compute the test and training error

lr_trerr = sum(abs(btr$rv - rv_hat_tr))/length(btr$rv) # 0.0844
lr_tserr = sum(abs(bts$rv - rv_hat_ts))/length(bts$rv) # 0.0945
lr_trerr
lr_tserr

###################################
# Confusion Matrix
###################################

conf = confusionMatrix(as.factor(rv_hat_ts), as.factor(bts$rv))
conf$table
conf

###################################
# K-Nearest Neighbors
###################################

set.seed(12345)
knn.pr = knn(btr[,1:13],bts[,1:13],btr[,14],k=1)

knn1_tserr = sum(abs(bts$rv - (as.numeric(knn.pr)-1)))/length(bts$rv) # 0.0945
knn1_tserr

conf = confusionMatrix(knn.pr, as.factor(bts$rv))
conf$table
conf

set.seed(12345)
knn.pr = knn(btr[,1:13],bts[,1:13],btr[,14],k=5)

knn5_tserr = sum(abs(bts$rv - (as.numeric(knn.pr)-1)))/length(bts$rv) # 0.1024
knn5_tserr

conf = confusionMatrix(knn.pr, as.factor(bts$rv))
conf$table
conf

###################################
# Selecting the predictors
###################################

summary(glm.fit)

vars = c("zn", "nox", "age", "dis", "rad", "ptratio", "black", "medv", "rv")

###########################################
# Fitting models for new set of predictors
#
###########################################

btr1 = my_Boston[train, vars]
bts1 = my_Boston[-train, vars]

###################################
# Linear Discriminant Analysis
###################################

lda.fit = lda(rv~., data = btr1)
lda.ptr = predict(lda.fit, newdata = btr1)
rv_hat_tr = as.numeric(lda.ptr$class)-1
lda.pts = predict(lda.fit, newdata = bts1)
rv_hat_ts = as.numeric(lda.pts$class)-1

# Compute the test and training error

lda_trerr = sum(abs(btr1$rv - rv_hat_tr))/length(btr1$rv) # 0.119
lda_tserr = sum(abs(bts1$rv - rv_hat_ts))/length(bts1$rv) # 0.157
lda_trerr
lda_tserr

conf = confusionMatrix(as.factor(rv_hat_ts), as.factor(bts1$rv))
conf$table
conf

###################################
# Logistic Regression
###################################

glm.fit = glm(rv~., data = btr1, family = "binomial")
summary(glm.fit)

# Prediction

glm.prtr = predict(glm.fit, type = "response", newdata = btr1)
rv_hat_tr = round(glm.prtr) # Converting Posterior probabilities to 0 and 1
glm.prts = predict(glm.fit, type = "response", newdata = bts1)
rv_hat_ts = round(glm.prts)

# Compute the test and training error

lr_trerr = sum(abs(btr1$rv - rv_hat_tr))/length(btr1$rv) # 0.1003
lr_tserr = sum(abs(bts1$rv - rv_hat_ts))/length(bts1$rv) # 0.1339
lr_trerr
lr_tserr

###################################
# Confusion Matrix
###################################

conf = confusionMatrix(as.factor(rv_hat_ts), as.factor(bts1$rv))
conf$table
conf

###################################
# K-Nearest Neighbors
###################################

set.seed(12345)
knn.pr = knn(btr1[,1:8],bts1[,1:8],btr1[,9],k=1)

knn1_tserr = sum(abs(bts1$rv - (as.numeric(knn.pr)-1)))/length(bts1$rv) # 0.0945
knn1_tserr

conf = confusionMatrix(knn.pr, as.factor(bts1$rv))
conf$table
conf

# K=5
set.seed(12345)
knn.pr = knn(btr1[,1:8],bts1[,1:8],btr1[,9],k=5)

knn5_tserr = sum(abs(bts1$rv - (as.numeric(knn.pr)-1)))/length(bts1$rv) # 0.1654
knn5_tserr

conf = confusionMatrix(knn.pr, as.factor(bts1$rv))
conf$table
conf

###################################
# Selecting the predictors - Set 2
###################################

summary(glm.fit)

vars = c("zn", "nox", "age", "dis", "rad", "medv", "rv")

###########################################
# Fitting models for new set of predictors
#
###########################################

btr1 = my_Boston[train, vars]
bts1 = my_Boston[-train, vars]

###################################
# Linear Discriminant Analysis
###################################

lda.fit = lda(rv~., data = btr1)
lda.ptr = predict(lda.fit, newdata = btr1)
rv_hat_tr = as.numeric(lda.ptr$class)-1
lda.pts = predict(lda.fit, newdata = bts1)
rv_hat_ts = as.numeric(lda.pts$class)-1

# Compute the test and training error

lda_trerr = sum(abs(btr1$rv - rv_hat_tr))/length(btr1$rv) # 0.119
lda_tserr = sum(abs(bts1$rv - rv_hat_ts))/length(bts1$rv) # 0.157
lda_trerr
lda_tserr

conf = confusionMatrix(as.factor(rv_hat_ts), as.factor(bts1$rv))
conf$table
conf

###################################
# Logistic Regression
###################################

glm.fit = glm(rv~., data = btr1, family = "binomial")
summary(glm.fit)

# Prediction

glm.prtr = predict(glm.fit, type = "response", newdata = btr1)
rv_hat_tr = round(glm.prtr) # Converting Posterior probabilities to 0 and 1
glm.prts = predict(glm.fit, type = "response", newdata = bts1)
rv_hat_ts = round(glm.prts)

# Compute the test and training error

lr_trerr = sum(abs(btr1$rv - rv_hat_tr))/length(btr1$rv) # 0.1003
lr_tserr = sum(abs(bts1$rv - rv_hat_ts))/length(bts1$rv) # 0.1339
lr_trerr
lr_tserr

###################################
# Confusion Matrix
###################################

conf = confusionMatrix(as.factor(rv_hat_ts), as.factor(bts1$rv))
conf$table
conf

###################################
# K-Nearest Neighbors
###################################

set.seed(12345)
knn.pr = knn(btr1[,1:6],bts1[,1:6],btr1[,7],k=1)

knn1_tserr = sum(abs(bts1$rv - (as.numeric(knn.pr)-1)))/length(bts1$rv) # 0.0945
knn1_tserr

conf = confusionMatrix(knn.pr, as.factor(bts1$rv))
conf$table
conf

# K=5
set.seed(12345)
knn.pr = knn(btr1[,1:6],bts1[,1:6],btr1[,7],k=5)

knn5_tserr = sum(abs(bts1$rv - (as.numeric(knn.pr)-1)))/length(bts1$rv) # 0.1654
knn5_tserr

conf = confusionMatrix(knn.pr, as.factor(bts1$rv))
conf$table
conf

