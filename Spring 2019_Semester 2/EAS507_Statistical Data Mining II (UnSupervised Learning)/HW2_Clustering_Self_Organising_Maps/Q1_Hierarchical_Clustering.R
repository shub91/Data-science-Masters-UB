###########################################################
##
## This code is the solution for Question 1 of Homework 2
## dealing with USArrests dataset in the base R package. 
## 
## Created: March 28, 2018
## Name: Shubham Sharma
###########################################################

###########################################################
## Attaching Libraries and Data
## 
###########################################################

rm(list = ls())

# source("http://bioconductor.org/biocLite.R")
# biocLite()
# biocLite("multtest")
# biocLite("cluster")
# install.packages("fpc")
# install.packages("bootcluster")

library("ISLR")
library("cluster")

attach(USArrests)
summary(USArrests)
str(USArrests)

apply(USArrests, 2, mean)
apply(USArrests, 2, var)

###########################################################
## Hierarchical Clustering
###########################################################

d_1 = USArrests

# Part a

d = dist(d_1, method = 'euclidean')
dim(as.matrix(d)) # 50 x 50
hc_1 = hclust(d, method = "complete")
quartz()
plot(hc_1, hang = -1, labels = row.names(d_1), cex = 0.5, main = "Cluster Dendogram without scaling")

# Part b

ct_1 = cutree(hc_1, h = 110)
ct_1
si_1 = silhouette(ct_1, dist = d)
quartz()
plot(si_1)

summary(si_1)

# Part c

ds_1 = scale(d_1)
d_s = dist(ds_1, method = 'euclidean')
hc_com = hclust(d_s, method = "complete")
quartz()
plot(hc_com, hang = -1, labels = row.names(ds_1), cex = 0.5, main = "Cluster Dendogram with scaling")

# Part d

ct_1_s = cutree(hc_com, k = 3)
ct_1_s
si_1_s = silhouette(ct_1_s, dist = d_s)
quartz()
plot(si_1_s)
summary(si_1_s)
table(ct_1, ct_1_s)

rand.index(ct_1,ct_1_s) # 0.69
adj.rand.index(ct_1,ct_1_s) # 0.36
