#####################################################################
## This code is the solution for Question 2 of Homework 3                  
##        
## Shubham Sharma                                                   
## Created: April 21, 2019                                          
#####################################################################
rm(list = ls())
graphics.off()


library(gRain)
#library(RHugin)
library(Rgraphviz)
library(gRbase)
library(ggm)

##################################################
## Constructing the given DAG
##################################################

g = list(~Burglary, ~Earthquake, ~Nap, ~TV,  ~John_Call|TV:Burglary:Earthquake, ~Mary_Call|Burglary:Earthquake:Nap)
chestdag = dagList(g)
quartz()
plot(chestdag)
