###########################################################
##
## This code is the solution for Question 2 of Homework 3
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

library(klaR)
library(caret)
library(class)

#######################################
# Reading data from the text file
#######################################

cn = c("1","2","3","OBSNO","G.AREA","I.AREA","SSPG","RWT","FPG","CLASS")
d=read.table("DiabetesAndrews36_1.txt",sep="",header=FALSE, col.names = cn)

############################
# Pairwise Scatter Plot
############################

quartz()
pairs(~ G.AREA + I.AREA + SSPG + RWT + FPG, d, col = d$CLASS) # 1-Black,2-Red,3-Green

###################################
# Create Test and training dataset
###################################

set.seed(1234)
vars = c("G.AREA","I.AREA","SSPG","RWT","FPG","CLASS")
train = sample(1:nrow(d), nrow(d)*0.75)
dtr = d[train, vars]
dts = d[-train, vars]

###################################
# Linear Discriminant Analysis
###################################

lda.fit = lda(CLASS~., data = dtr)
lda.ptr = predict(lda.fit, newdata = dtr)
lda_trerr = 1 - mean(lda.ptr$class == dtr$CLASS) 
lda.pts = predict(lda.fit, newdata = dts)
lda_tserr = 1 - mean(lda.pts$class == dts$CLASS) 
lda_trerr
lda_tserr

###################################
# Quadratic Discriminant Analysis
###################################

qda.fit = qda(CLASS~., data = dtr)
qda.ptr = predict(qda.fit, newdata = dtr)
qda_trerr = 1 - mean(qda.ptr$class == dtr$CLASS) 
qda.pts = predict(qda.fit, newdata = dts)
qda_tserr = 1 - mean(qda.pts$class == dts$CLASS) 
qda_trerr
qda_tserr

###################################
# C Part
###################################

df = d[0,vars]
df[1,] = c(0.98, 122, 544, 186, 184, NA)
lda.pts1 = predict(lda.fit, newdata = df)
qda.pts1 = predict(qda.fit, newdata = df)

lda.pts1$class #LDA
qda.pts1$class #QDA
