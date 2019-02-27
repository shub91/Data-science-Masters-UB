###########################################################
##
## This code is the solution for Question 4 of Homework 1
## dealing with Exploratory data analysis on Student
## Performance data
##
## Created: September 13, 2018
## Name: Shubham Sharma
###########################################################

rm(list = ls())
setwd("~/Documents/SDM/Q4")
library(ElemStatLearn)
?ElemStatLearn
library(ggplot2)
ls("package:ElemStatLearn")
?zip.test
?zip.train
tr = as.data.frame(zip.train)
dim(tr)
ts = as.data.frame(zip.test)
dim(ts)
trs = subset(tr, V1 == 2 | V1 == 3)
tss = subset(ts, V1 == 2 | V1 == 3)

lm.4 = lm(V1~., data=trs)
summary(lm.4)

## KNN for test data

g <- ggplot(trs, aes(V3,V2)) + geom_point(aes(colour = as.factor(V1))) + theme(legend.position = "none")
quartz()
plot(g)
ggsave(filename = "orig.png", plot = g, height = 5, width = 5)
