###########################################################
##
## This code is the solution for Question 3 of Homework 2
## dealing with primate.scapulae dataset. 
## 
## Created: March 28, 2018
## Name: Shubham Sharma
###########################################################

###########################################################
## Attaching Libraries and Data
###########################################################

rm(list = ls())

library("multtest")
library("fpc")
library("cluster")
library("bootcluster")
library("fossil")

load("primate.scapulae.rda")
d = primate.scapulae
sum(is.na(d)) # gamma has 40 NA values
d = d[,-9] # dropping gamma
summary(d)
str(d) # 105 obs. of  11 variables

# Looking at the classes

unique(d$class)
ds = d[,1:8]

###########################################################
## Part (a) Hierarchical Clustering
###########################################################

d1 = dist(ds)
dim(as.matrix(d1)) # 105 x 105

### Single Linkage

hc_s = hclust(d1, method = "single")
quartz()
plot(hc_s, hang = -1, labels = d$class, cex = 0.5, main = "Cluster Dendogram using Single Linkage")

store_s = c()
for (i in 2:6){
  ct_s = cutree(hc_s, k=i)
  si_s = silhouette(ct_s, dist = d1)
  avg_width_s = summary(si_s)$avg.width
  store_s = c(store_s, avg_width_s)
}
store_s # max silhouette width = 0.572 for k = 4

ct_s = cutree(hc_s, k = 4)
si_s = silhouette(ct_s, dist = d1)

quartz()
plot(si_s)

# Missclassifiction rate

mean(as.numeric(d$classdigit)!= ct_s) # 0.723
table(ct_s,as.numeric(d$classdigit))

# Rand Index

rand.index(ct_s,as.numeric(d$classdigit)) # 0.836
adj.rand.index(ct_s,as.numeric(d$classdigit)) # 0.74

### Average Linkage

hc_a = hclust(d1, method = "average")
quartz()
plot(hc_a, hang = -1, labels = d$class, cex = 0.5, main = "Cluster Dendogram using Average Linkage")

store_a = c()
for (i in 2:6){
  ct_a = cutree(hc_a, k=i)
  si_a = silhouette(ct_a, dist = d1)
  avg_width_a = summary(si_a)$avg.width
  store_a = c(store_a, avg_width_a)
}
store_a # max silhouette width = 0.589 for k = 4

ct_a = cutree(hc_a, k = 4)
si_a = silhouette(ct_a, dist = d1)

quartz()
plot(si_a)

# Missclassifiction rate

mean(as.numeric(d$classdigit)!= ct_a) # 0.876
table(ct_a,as.numeric(d$classdigit))

# Rand Index

rand.index(ct_a,as.numeric(d$classdigit)) # 0.81
adj.rand.index(ct_a,as.numeric(d$classdigit)) # 0.66

### Complete Linkage

hc_c = hclust(d1, method = "complete")
quartz()
plot(hc_c, hang = -1, labels = d$class, cex = 0.5, main = "Cluster Dendogram using Complete Linkage")

store_c = c()
for (i in 2:6){
  ct_c = cutree(hc_c, k=i)
  si_c = silhouette(ct_c, dist = d1)
  avg_width_c = summary(si_c)$avg.width
  store_c = c(store_c, avg_width_c)
}
store_c # max silhouette width = 0.580 for k = 4

ct_c = cutree(hc_c, k = 4)
si_c = silhouette(ct_c, dist = d1)

quartz()
plot(si_c)

# Missclassifiction rate

mean(as.numeric(d$classdigit)!= ct_c) # 0.95
table(ct_c,as.numeric(d$classdigit)) 

rand.index(ct_c,as.numeric(d$classdigit)) # 0.81
adj.rand.index(ct_c,as.numeric(d$classdigit)) # 0.63


###########################################################
## Part (b) K-Medoids
###########################################################

kmed = pamk(ds)

# Optimal number of clusters
kmed$nc #3

# Tabulate the results
table(kmed$pamobject$clustering, as.numeric(d$classdigit))

rand.index(as.numeric(d$classdigit),kmed$pamobject$clustering) # 0.8212
adj.rand.index(as.numeric(d$classdigit),kmed$pamobject$clustering) # 0.6020

# plot the results for k = 3
quartz()
layout(matrix(c(1,2), 1, 2))
plot(kmed$pamobject)

# k= 5
kmed5 = pamk(ds, 5)
table(kmed5$pamobject$clustering, as.numeric(d$classdigit))
rand.index(as.numeric(d$classdigit),kmed5$pamobject$clustering) # 0.916
adj.rand.index(as.numeric(d$classdigit),kmed5$pamobject$clustering) # 0.767

kmed_val = kmed5$pamobject$clustering
kmed_missrate = mean(as.numeric(d$classdigit) != kmed_val) # 0.4

quartz()
layout(matrix(c(1,2), 1, 2))
plot(kmed5$pamobject)

# Gap Statistics
gap_kmed = clusGap(ds, pam, K.max = 10, B = 100)
quartz()
plot(gap_kmed, main = 'Gap Statistics for K-medoids') # k = 5 since max gap statistic

