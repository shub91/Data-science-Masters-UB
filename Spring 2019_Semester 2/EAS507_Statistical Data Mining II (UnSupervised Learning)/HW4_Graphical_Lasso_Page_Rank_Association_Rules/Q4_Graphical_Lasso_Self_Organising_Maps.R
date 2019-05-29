#####################################################################
## This code is the solution for Question 4 of Homework 4                  
##        
## Shubham Sharma                                                   
## Created: April 29, 2019                                          
#####################################################################

rm(list = ls())
graphics.off()

##################################################
## Loading the libraries
##################################################

library(gRbase)
library(gRim)
library(gRain)
library(glasso)
library(Rgraphviz)
library(qgraph)

##################################################
## Attaching the data
##################################################

d = state.x77
summary(d)
quartz()
pairs(d)

# quartz()
# plot(d[,4], d[,5])

d = as.data.frame(d)

# Look at partial correlation
S.d = cov.wt(d, method = "ML")
PC.d = cov2pcor(S.d$cov)
quartz()
heatmap(PC.d, main = "Heat map for Partial Correlation")

# Graphical Lasso package to learn GGMs (Gausian Graphical Models)

ls("package:glasso")
S = S.d$cov # covariance matrix
library(graph)

# Estimate a single graph
m0.lasso = glasso(S, rho = 1)
names(m0.lasso)
my.edges = m0.lasso$wi != 0
diag(my.edges) = FALSE
g.lasso = as(my.edges,  "graphNEL") # convert for plotting
nodes(g.lasso) = names(d)
glasso.net = cmod(g.lasso, data = d)

quartz()
plot(glasso.net, main = "For rho = 1")

# Estimate over a range of rho's
rhos = c(2, 4, 6, 10, 15)
m0.lasso = glassopath(S, rho = rhos)
graphics.off()
for (i in 1:length(rhos)){
  my.edges = m0.lasso$wi[, , i] != 0 # turning to adjacency matrix
  diag(my.edges) = FALSE
  g.lasso = as(my.edges, "graphNEL") # convert for plotting
  nodes(g.lasso) = names(d)
  glasso.net = cmod(g.lasso, data = d)
  
  quartz()
  plot(glasso.net, main = "Graphical Lasso Plot for rho =")
}

##################################################
## Self-Organising Maps
##################################################

library(kohonen)

d.scaled <- scale(d)

# fit an SOM
set.seed(123)
som_grid = somgrid(xdim = 5, ydim = 5, topo = "hexagonal")
wine.som = som(d.scaled, grid = som_grid, rlen = 3000) #3000 iterations through the dataset

codes = wine.som$codes[[1]]
?plot.kohonen

quartz()
plot(wine.som, main = "5 x 5 Hexagonal grid SOM Plot")

quartz()
plot(wine.som, type = "changes", main = "")

quartz()
plot(wine.som, type = "count", main = "No. of items mapped to different units")

quartz()
plot(wine.som, type = "mapping", main = "No. of items mapped to different units")

coolBlueHotRed = function(n, alpha = 1){rainbow(n, end=4/6, alpha = alpha)[n:1]} # creating color pallete

quartz()
plot(wine.som, type = "dist.neighbours", palette.name = coolBlueHotRed)

# component plane plots
# for (i in 1:13){
#  quartz()
#  plot(wine.som, type = "property", property=codes[,i], main = colnames(codes)[i])
#}

d <- dist(codes)
hc <- hclust(d)

quartz()
plot(hc)

som_cluster <- cutree(hc, h = 7)

# plot the SOM with the found clusters

my_pal <- c("red", "blue", "yellow")
my_bhcol <- my_pal[som_cluster]

quartz()
plot(wine.som, type = "mapping", col = "black", bgcol = my_bhcol)
add.cluster.boundaries(wine.som, som_cluster)












