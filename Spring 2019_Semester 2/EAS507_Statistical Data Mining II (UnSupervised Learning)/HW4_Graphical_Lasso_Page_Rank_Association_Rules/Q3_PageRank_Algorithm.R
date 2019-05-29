######################################################################
## This code is the solution for Question 3 of Homework 4 which deals                
## with Page Rank algorithm
## Shubham Sharma                                                   
## Created: April 29, 2019                                          
######################################################################

rm(list = ls())
graphics.off()

##################################################
## Loading the libraries
##################################################

library(igraph)
library(Rgraphviz)

##################################################
## Attaching the data
##################################################

# Webgraph A
n_1 = data.frame(names = c("A", "B", "C", "D", "E", "F"))
edges_1 = data.frame(from = c("D","D","E","B","B","F","C"), to =   c("B","E","D","E","C","C","A"))
g_1 = graph.data.frame(edges_1, directed = TRUE, vertices = n_1)
quartz()
plot(g_1, main = "Webgraph A")
damping_prob = c(0.05, 0.25, 0.50, 0.75, 0.95)
for (i in damping_prob) {
  rank_1 = page.rank(g_1, damping = i)
  print(rank_1$vector)}

# Webgraph B
n_2 = data.frame(names = c("A", "B", "C", "D", "E", "F", "G", "H"))
edges_2 = data.frame(from = c("F", "G", "H", "D", "E", "B", "C"), to =   c("C", "C", "C", "B", "B", "A", "A"))
g_2 = graph.data.frame(edges_2, directed = TRUE, vertices = n_2)
quartz()
plot(g_2, main = "Webgraph B")
rank_2 = page.rank(g_2, damping = 0.15)
rank_2$vector
