###########################################################
##
## This code is the solution for Question 1 of Homework 1
## dealing with MovieLense data in the recommenderlab 
## package
## Created: February 21, 2018
## Name: Shubham Sharma
###########################################################

###########################################################
## Attaching Libraries and Data
## 
###########################################################

rm(list = ls())
library(recommenderlab)
data(MovieLense)

summary(MovieLense)
str(MovieLense)

###########################################################
## Loading Data and converting it into real rating Matrix
###########################################################

rm(list = ls())
data("MovieLense")

d = MovieLense
dim(d) ### 943 users and 1664 movies

d = as(d,'realRatingMatrix')
class(d)

## Creating the recommender model using user based collaboartive filtering

rp = Recommender(d, method = "UBCF", param=list(method="Cosine",nn=50))
ur = predict(rp, d, type = "ratings")

getRatingMatrix(ur)
pval = as(ur,'matrix')
write.csv(pval, file = "q1ur.csv")

###########################
## Cross validation
###########################

set.seed(12345)
S = evaluationScheme(MovieLense, method="cross-validation", k = 5, given = -5, goodRating = 4)
S
A = list("random items" = list(name = "RANDOM", param = NULL), "popular items" = list(name = "POPULAR", param = NULL),
         "user-based CF"= list(name = "UBCF", param = list(nn = 50)), "item-based CF"= list(name = "IBCF", param = list(k = 50)),
          "SVD approximation" = list(name = "SVD", param = list(k = 50)))

R = evaluate(S, A, type = "topNList", n = c(1,3,5,10,15,20))
Ru = evaluate(S, A, type = "ratings", n = c(1,3,5,10,15,20))
avg(Ru)
names(R)
quartz()
plot(R, annotate=c(1,3), legend="topleft")
quartz()
plot(R, "prec/rec", annotate = TRUE)
