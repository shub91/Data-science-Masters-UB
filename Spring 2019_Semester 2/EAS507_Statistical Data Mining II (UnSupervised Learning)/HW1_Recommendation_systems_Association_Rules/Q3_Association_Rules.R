###########################################################
##
## This code is the solution for Question 3 of Homework 1
## dealing with Boston Housing Data in the ElemStatLearn 
## Package
## Created: February 21, 2018
## Name: Shubham Sharma
###########################################################

###########################################################
## Attaching Libraries and Data
## 
###########################################################

rm(list = ls())
library(MASS)
library(arules)

attach(Boston)
summary(Boston)
str(Boston)

###########################################################
## Plotting histigrams and making grouping categories for 
## various attributes
###########################################################

d = Boston

quartz()
par(mfrow = c(2,2))
hist(Boston$crim,breaks = "Sturges", probability = T, xlab = "CRIM", main = "Histogram for per capita crime rate by town")
hist(Boston$zn,breaks = "Sturges", probability = T, xlab = "zn", main = "Histogram for proportion of residential land zoned for lots over 25,000 sq.ft")
hist(Boston$indus,breaks = "Sturges", probability = T, xlab = "indus", main = "Histogram for proportion of non-retail business acres per town")
hist(Boston$chas,breaks = "Sturges", probability = T, xlab = "chas", main = "Histogram for Charles River dummy variable")

d[["crim"]] = ordered(cut(d[["crim"]], c(0, 10, 20, 90)), labels = c("low", "med", "high"))
d[["zn"]] = ordered(cut(d[["zn"]], c(-1, 20, 101)), labels = c("low", "high"))
d[["indus"]] = ordered(cut(d[["indus"]], c(0, 18, 28)), labels = c("low", "high"))
d[["chas"]] = ordered(cut(d[["chas"]], c(-1, 0.5, 1.1)), labels = c("Bounded", "Unbounded"))

quartz()
par(mfrow = c(2,2))
hist(Boston$nox,breaks = "Sturges", probability = T, xlab = "nox", main = "Histogram for nitrogen oxides concentration (parts per 10 million)")
hist(Boston$rm,breaks = "Sturges", probability = T, xlab = "rm", main = "Histogram for average number of rooms per dwelling")
hist(Boston$age,breaks = "Sturges", probability = T, xlab = "age", main = "Histogram for proportion of owner-occupied units built prior to 1940")
hist(Boston$dis,breaks = "Sturges", probability = T, xlab = "dis", main = "Histogram for weighted mean of distances to five Boston employment centres")

d[["nox"]] = ordered(cut(d[["nox"]], c(0, 0.55, 0.75, 0.9)), labels = c("low", "med", "high"))
d[["rm"]] = ordered(cut(d[["rm"]], c(0, 5.5,7,9)), labels = c("low","med", "high"))
d[["age"]] = ordered(cut(d[["age"]], c(0,30,70,100)), labels = c("low","med", "high"))
d[["dis"]] = ordered(cut(d[["dis"]], c(0,3,8,13)), labels = c("low","med","high"))

quartz()
par(mfrow = c(2,2))
hist(Boston$rad,breaks = "Sturges", probability = T, xlab = "rad", main = "Histogram for index of accessibility to radial highways")
hist(Boston$tax,breaks = "Sturges", probability = T, xlab = "tax", main = "Histogram for full-value property-tax rate per $10,000")
hist(Boston$ptratio,breaks = "Sturges", probability = T, xlab = "ptratio", main = "Histogram for pupil-teacher ratio by town")
hist(Boston$black,breaks = "Sturges", probability = T, xlab = "black", main = "Histogram for 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town")

d[["rad"]] = ordered(cut(d[["rad"]], c(0, 10, 25)), labels = c("low","high"))
d[["tax"]] = ordered(cut(d[["tax"]], c(185, 300, 500 ,715)), labels = c("low","med", "high"))
d[["ptratio"]] = ordered(cut(d[["ptratio"]], c(12,16,19,23)), labels = c("low","med", "high"))
d[["black"]] = ordered(cut(d[["black"]], c(0,350,400)), labels = c("low","high"))

quartz()
par(mfrow = c(1,2))
hist(Boston$lstat,breaks = "Sturges", probability = T, xlab = "lstat", main = "Histogram for lower status of the population (percent)")
hist(Boston$medv,breaks = "Sturges", probability = T, xlab = "medv", main = "Histogram for median value of owner-occupied homes in $1000s")

d[["lstat"]] = ordered(cut(d[["lstat"]], c(0, 15,60)), labels = c("low", "high"))
d[["medv"]] = ordered(cut(d[["medv"]], c(0, 25,70)), labels = c("low", "high"))

# Convertion to binary incidence matrix

d_bim = as(d, "transactions")
summary(d_bim)

# Item Frequency Plot
quartz()
itemFrequencyPlot(d_bim, support = 0.05, cex.names = 0.8, type = "relative", horiz = TRUE, xlab = "Item Frequency (relative)", main = "Item Frequency Plot" )

# Apriori Algorithm
rules = apriori(d_bim, parameter = list(support = 0.01, confidence = 0.7, maxlen = 14))
summary(rules)

# Part C

rules_1 = subset(rules, subset = lhs %ain% "crim=low" & rhs %ain% "dis=low" & lift>1)
rules_2 = subset(rules, subset = lhs %ain% c("crim=low", "dis=low") & lift>1)
inspect(head(sort(rules_1, by = "lift", decreasing = TRUE), n=10 ), ruleSep = "--->", itemSep = "+")
inspect(head(sort(rules_2, by = "lift", decreasing = TRUE), n=10 ), ruleSep = "--->", itemSep = "+")

# Part D

rules_3 = subset(rules, subset = rhs %in% "ptratio=low" & lift>1)
inspect(head(sort(rules_3, by = "lift", decreasing = TRUE), n=10 ), ruleSep = "--->", itemSep = "+")

# Part E

lm.fit1 = lm(ptratio~., data=Boston)
summary(lm.fit1)
