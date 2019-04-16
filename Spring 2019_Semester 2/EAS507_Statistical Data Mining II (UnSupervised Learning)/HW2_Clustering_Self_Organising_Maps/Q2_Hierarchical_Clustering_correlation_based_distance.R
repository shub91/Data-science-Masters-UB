###########################################################
##
## This code is the solution for Question 2 of Homework 2
## dealing with gene expression dataset. 
## 
## Created: March 28, 2018
## Name: Shubham Sharma
###########################################################

###########################################################
## Attaching Libraries and Data
###########################################################

rm(list = ls())

library("ISLR")
library("cluster")

d_2 = read.csv('Ch10Ex11.csv',header = F)
summary(d_2)
str(d_2) # 1000 obs. of  40 variables

apply(d_2, 2, mean)
apply(d_2, 2, var)

###########################################################
## Hierarchical Clustering
###########################################################

hc_2_c = hclust(as.dist(1 - cor(d_2)), method = "complete")
quartz()
plot(hc_2_c, cex = 0.5, main = "Cluster Dendogram using Complete Linkage")

hc_2_s = hclust(as.dist(1 - cor(d_2)), method = "single")
quartz()
plot(hc_2_s, cex = 0.5, main = "Cluster Dendogram using Single Linkage")

hc_2_a = hclust(as.dist(1 - cor(d_2)), method = "average")
quartz()
plot(hc_2_a, cex = 0.5, main = "Cluster Dendogram using Average Linkage")

# Part c: Principal Component Analysis

pc = prcomp(t(d_2), center = TRUE) # transposing the matrix
head(pc$rotation)

quartz()
plot(pc)

quartz()
biplot(pc, scale = 0)

totload = apply(pc$rotation, 1, sum)
i = order(abs(totload), decreasing = TRUE)
i[1:10]



