###########################################################
##
## This code is the solution for Question 2 of Homework 1
## dealing with MovieLense data in the recommenderlab 
## package
## Created: February 21, 2018
## Name: Shubham Sharma
###########################################################

###########################################################
## Attaching Libraries
###########################################################

rm(list = ls())
library(recommenderlab)

VM = matrix(c(5,4,NA,7,1,6,NA,3,4,NA,7,3,4,3,3,4,NA,1,6,2,3,5,1,NA,2,NA,4,NA,4,5), nrow = 5)

###########################################################
## Create rating matrix
###########################################################


matrix = as(VM, "realRatingMatrix")

###########################################################
## UBCF (Pearson Coefficient)
###########################################################




model_ubcf = Recommender(matrix, method = 'UBCF', parameter = list(normalize = "center", method = "pearson"))

UR = predict(model_ubcf, matrix, type = "ratings")

getRatingMatrix(UR)

model_ibcf = Recommender(matrix, method = "IBCF", parameter = list(method = "cosine"))

IR = predict(model_ibcf, matrix, type = 'ratings')

getRatingMatrix(IR)