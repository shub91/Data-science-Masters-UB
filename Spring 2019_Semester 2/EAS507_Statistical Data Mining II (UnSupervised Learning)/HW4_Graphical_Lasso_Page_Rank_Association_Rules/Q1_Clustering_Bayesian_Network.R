####################################################################################
## This code is to consider the Parkinsons Telemonitoring Dataset.
## a) Cluster this data over using a sensible subset of variables using two methods
## described in class. For example, you would not want to use variables like
## “subject id”. Also leave out “motor_UPDRS” and “total_UPDRS”. How well do
## the clusters capture “motor_UPDRS” and “total_UPDRS”.
## b) Fit a Bayesian Network using this data. Include “motor_UPDRS” and
## “total_UPDRS”, but not both, and force this variable to be the bottom node of the
## network.
## c) A collaborator asks you to characterize “Jitter” related variables for a new patient
## that has a relatively high UPDRS score (two standard deviations above the
## mean). Use your Bayesian Network to answer this question.
## Pradeep Kumar Joshi
## HomeWork 4 - Question 1
####################################################################################
rm(list = ls())

library(gRain)
library(Rgraphviz)
library(gRbase)
library(ggm)
library(bnlearn)
library(igraph)
library(gRim)
library(tidyverse)
library(magrittr)
options(warn=-1)

# set working directory
setwd("/Users/pradeep/Desktop/Codes/Data_Mining_II/Assignment-4")

# Reading the Parkinsons Telemonitoring Dataset
#parkinsons <- read_file("parkinsons_updrs.data", locale = default_locale())
parkinsons <- read_csv("parkinsons_updrs.data")
dim(parkinsons) #5875   22
summary(parkinsons)
sum(is.na(parkinsons)) #Checking Null Values - 0 Null Values
park_dataFactors <- parkinsons #For Factoring

#Part - (a) - Clustering
#leaving out unwanted variables
data_park <- parkinsons %>% select(-`subject#`,-motor_UPDRS, -total_UPDRS)
hier_parkClust <- hclust(dist(data_park %>% as.matrix)) #Hierarchical Clustering
quartz()
plot(hier_parkClust, xlab="Euclidean Distance", ylab="Height",labels = FALSE, hang = 0, 
     main = "Parkinsons Cluster Dendogram")
cut_parkClust <- cutree(hier_parkClust, k = 10) #Cut the dendogram

#Cutting total and motor UPDRS to check cluster capture
total_UPDRS <- parkinsons$`total_UPDRS` %>% cut(breaks=c(0,10,20,30,40,50,60))
table(cut_parkClust, total_UPDRS)

motor_UPDRS <- parkinsons$`motor_UPDRS` %>% cut(breaks=c(0, 10, 20, 30, 40))
table(cut_parkClust, motor_UPDRS)
#K-means Clustering
parkinsson_kmeans4 <- kmeans(data_park,4)
table(parkinsson_kmeans4$cluster, motor_UPDRS)

parkinsson_kmeans6 <- kmeans(data_park,6)
table(parkinsson_kmeans6$cluster, total_UPDRS)

#Part - (b) - Fit a Bayesian Network
park_dataFactors$age <- cut(parkinsons$age,breaks=c(30,50,70,100))
park_dataFactors$sex <-  as.factor(parkinsons$sex)
park_dataFactors$test_time <- cut(parkinsons$test_time,breaks=c(-5, 50, 100, 150, 220))
park_dataFactors$total_UPDRS <- cut(parkinsons$total_UPDRS,breaks=c(0,10,20,30,40,50,60))
park_dataFactors$motor_UPDRS <-cut(parkinsons$motor_UPDRS,breaks=c(0, 10, 20, 30, 40))
park_dataFactors$`Jitter(%)` <-  cut(parkinsons$`Jitter(%)`,breaks=c(0, 0.01, 1))
park_dataFactors$`Jitter(Abs)` <- cut(parkinsons$`Jitter(Abs)`,breaks=c(0, 0.00008, 1))
park_dataFactors$`Jitter:RAP` <- cut(parkinsons$`Jitter:RAP`,breaks=c(0, 0.006, 1))
park_dataFactors$`Jitter:PPQ5` <- cut(parkinsons$`Jitter:PPQ5`,breaks=c(0, 0.007, 11))
park_dataFactors$`Jitter:DDP` <- cut(parkinsons$`Jitter:DDP`,breaks=c(0,0.02,1))
park_dataFactors$`Shimmer` <- cut(parkinsons$`Shimmer`,breaks=c(0, 0.08, 1))
park_dataFactors$`Shimmer(dB)` <- cut(parkinsons$`Shimmer(dB)`,breaks=c(0, 0.7, 2.2))
park_dataFactors$`Shimmer:APQ3` <- cut(parkinsons$`Shimmer:APQ3`,breaks=c(0,0.035, 1))
park_dataFactors$`Shimmer:APQ5` <- cut(parkinsons$`Shimmer:APQ5`,breaks=c(0,0.04,1))
park_dataFactors$`Shimmer:APQ11` <- cut(parkinsons$`Shimmer:APQ11`,breaks=c(0, 0.05,1))
park_dataFactors$`Shimmer:DDA` <- cut(parkinsons$`Shimmer:DDA`,breaks=c(0, 0.1,1))
park_dataFactors$`NHR` <- cut(parkinsons$`NHR`,breaks=c(0, 0.06, 1))
park_dataFactors$`HNR` <- cut(parkinsons$`HNR`,breaks=c(0, 14, 40))
park_dataFactors$`RPDE` <- cut(parkinsons$`RPDE`,breaks=c(0, 0.55, 1))
park_dataFactors$`DFA` <- cut(parkinsons$`DFA`,breaks=c(0.5,0.6,0.7,0.87))
park_dataFactors$`PPE` <- cut(parkinsons$`PPE`,breaks=c(0, 0.25,1))

#Include “motor_UPDRS” and “total_UPDRS”, but not both
park_dataFactors <- park_dataFactors %>% select(-motor_UPDRS)
park_dataFactors <- park_dataFactors %>% as.data.frame()

network_park <- hc(park_dataFactors)
bayesian_park <- network_park %>% amat %>% as("graphNEL")
plot(bayesian_park,main="Parkinson Bayesian Network")

#Forcing motor_UPDRS to be at the bottom of the node
park_data <- rep(1,20)
park_data[4] <- 2
park_mat <- matrix(0, 20, 20)

park_dataFactors <- park_dataFactors %>% select(-`subject#`)
colnames(park_mat) <- park_dataFactors %>% names
rownames(park_mat) <- park_dataFactors %>% names

park_mat[park_data == 2, park_data < 2] <- 1
edge_list <- data.frame(get.edgelist(as(park_mat, "igraph")))
names(edge_list) <- c("From", "To")
network_parkMat <- hc(park_dataFactors,blacklist=edge_list)
network_parkForced <- as(amat(network_parkMat), "graphNEL")
plot(network_parkForced,main="Parkinson Bayesian Network with Constraint")

#Part - (c) Characterizing “Jitter” related variables
network_parkCPT <- extractCPT(park_dataFactors, network_parkForced, smooth=0.5)
network_parkCPT <- compileCPT(network_parkCPT)
network_parkGrain <- grain(network_parkCPT)

park_grainEvidence <- setFinding(network_parkGrain,nodes = c("total_UPDRS"), states = c("(50,60]"))

# probabilistic query, with and without given evidence and Marginal distribution 
park_withEvidence <- querygrain(park_grainEvidence, nodes = c("Jitter(%)", "Jitter(Abs)", "Jitter:RAP", "Jitter:PPQ5", "Jitter:DDP"), type = "marginal")
park_withEvidence
park_withoutEvidence <- querygrain(network_parkGrain, nodes = c("Jitter(%)", "Jitter(Abs)", "Jitter:RAP", "Jitter:PPQ5", "Jitter:DDP"), type = "marginal")
park_withoutEvidence