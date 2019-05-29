#####################################################################
## This code is the solution for Question 5 of Homework 4 which is a                
## function for Complete-linkage agglomerative hierarchical clustering
## Shubham Sharma                                                   
## Created: April 29, 2019                                          
#####################################################################

rm(list = ls())
graphics.off()

##################################################
## Attaching the data
##################################################

library(datasets)
data(iris)
d = iris[c(1:4)]

# Options for distance metric: "euclidean", "manhattan","binary" ,"maximum", "canberra", "minkowski".

dist_d = dist(d, method = 'euclidean')
dist_d_mat = as.matrix(dist_d)

D = data.frame(dist_d_mat)
rownum = nrow(D)
M = matrix(0, rownum-1, 2)
H = vector(length = rownum-1)
diag(D) = Inf
rownames(D) = -(1:rownum)
colnames(D) = -(1:rownum)

for (i in 1:(rownum-1)) { 
  name_col <- colnames(D)
  d = which(D == min(D), arr.ind = TRUE)[1,,drop=FALSE]
  H[i] = min(D) # Min distance pair
  M[i,] = as.numeric(name_col[d])
  clus = c(d, which(name_col %in% name_col[d[1, name_col[d] > 0]]))
  
  colnames(D)[clus] = i
  r = apply(D[d,], 2, max)
  D[min(d),] = r
  D[,min(d)] = r
  D[min(d), min(d)] = Inf
  D[max(d),] = Inf
  D[,max(d)] = Inf
}

clust_d = hclust(dist_d, method = 'single')
h = list() 
h$merge = M 
h$order = clust_d$order 
h$height = H  
h$labels = iris$Species
class(h) = 'hclust' 

quartz()
plot(h, main = "Agglomerative Hierarchical Clustering using Complete Linkage", cex = 0.5)


