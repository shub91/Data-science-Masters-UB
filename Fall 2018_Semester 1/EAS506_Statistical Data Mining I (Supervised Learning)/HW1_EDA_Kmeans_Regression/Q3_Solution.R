###########################################################
##
## This code is the solution for Question 3 of Homework 1
## dealing with Exploratory data analysis on Student
## Performance data
##
## Created: September 13, 2018
## Name: Shubham Sharma
###########################################################

rm(list = ls())
setwd("~/Documents/SDM/Q3")
library(MASS)
data(Boston)

############################
# Pairwise Scatter Plot
############################
quartz()
pairs(~ crim + nox + lstat + black + medv + chas, Boston)
quartz()
pairs(~ crim + dis + rad + indus + tax + ptratio, Boston)
quartz()
pairs(~ nox + age + zn, Boston)

############################
# Box plots
############################

quartz()
par(mfrow = c(1,3))
boxplot(Boston$crim, horizontal = FALSE, xlab = "Per capita crime rate")
boxplot(Boston$tax, horizontal = FALSE, xlab = "full-value property-tax rate per 10,000")
boxplot(Boston$ptratio, horizontal = FALSE, xlab = "pupil-teacher ratio by town")
max(Boston$crim)-min(Boston$crim)
max(Boston$tax)-min(Boston$tax)
max(Boston$ptratio)-min(Boston$ptratio)

rm8 = subset(Boston, rm>8)
rm7 = subset(Boston, rm>7)
dim(rm7)
dim(rm8)
mean(rm8$medv)
mean(rm8$crim)
mean(rm8$lstat)
mean(rm8$tax)
mean(Boston$medv)
mean(Boston$tax)
mean(Boston$crim)
mean(Boston$lstat)

quartz()
par(mfrow = c(2,4))
boxplot(rm8$medv, horizontal = FALSE, xlab = "medv")
boxplot(Boston$medv, horizontal = FALSE, xlab = "Boston medv")
boxplot(rm8$tax, horizontal = FALSE, xlab = "tax")
boxplot(Boston$tax, horizontal = FALSE, xlab = "Boston tax")
boxplot(rm8$lstat, horizontal = FALSE, xlab = "lstat")
boxplot(Boston$lstat, horizontal = FALSE, xlab = "Boston lstat")
boxplot(rm8$crim, horizontal = FALSE, xlab = "crim")
boxplot(Boston$crim, horizontal = FALSE, xlab = "Boston crim")

savehistory()
q()
