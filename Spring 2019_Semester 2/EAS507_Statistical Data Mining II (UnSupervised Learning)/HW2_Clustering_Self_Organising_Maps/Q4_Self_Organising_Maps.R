###########################################################
##
## This code is the solution for Question 4 of Homework 2
## dealing with Wisconsin Breast-Cancer data. 
## 
## Created: March 28, 2018
## Name: Shubham Sharma
###########################################################

###########################################################
## Attaching Libraries and Data
###########################################################

rm(list = ls())

library(kohonen)
library(TH.data)

# d_4 = read.table("breast-cancer-wisconsin.data.txt", header = FALSE, sep = ",")
d_4 = read.csv("wdbc.data", sep = ",", header = FALSE)
summary(d_4)
str(d_4) # 569 obs. of  32 variables

d_4 = d_4[,-1]
summary(d_4)
str(d_4) # 569 obs. of  31 variables
Classes = d_4[,1]
d_4 = d_4[,-1]
d_4 = as.matrix(d_4)

###########################################################
## Self Organising Maps
###########################################################

som_4 = somgrid(xdim = 3, ydim = 3, topo = "rectangular")
d_som = som(d_4, grid = som_4, rlen = 2000)
C = d_som$codes[[1]]

quartz()
par(mfrow = c(2,2))
plot(d_som, main = "WDBC Data")
plot(d_som, type = "changes", main = "WDBC Data")
plot(d_som, type = "count")
plot(d_som, type = "mapping")
pal_name = function(n, alpha = 1){rainbow(n, end=4/6, alpha = alpha)[n:1]}

quartz()
plot(d_som, type = "dist.neighbours", palette.name = pal_name)

# Component plane plots

for (i in 1:30){
  plot(d_som, type = "property", property=C[,i], main = colnames(C)[i])
}

d_c = dist(C)
hc = hclust(d_c)

quartz()
plot(hc)

som_cluster = cutree(hc, k = 2)

# SOM with the clusters obtained

pal = c("red", "blue")
bhcol = pal[som_cluster]

quartz()
plot(d_som, type = "mapping", col = "black", bgcol = bhcol)
add.cluster.boundaries(d_som, som_cluster)


