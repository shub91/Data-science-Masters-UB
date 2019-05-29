#####################################################################
## This code is the solution for Question 3 of Homework 3                  
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

g = list(~A, ~B,  ~C|A, ~F|C:A:E, ~D|A:B, ~E|B, ~G|D:E, ~H|F:G)
chestdag = dagList(g)
quartz()
plot(chestdag)

##################################################
## Checking for d-separation
##################################################

dSep(as(chestdag, "matrix"), first = "C", second = "G", cond = NULL) # False
dSep(as(chestdag, "matrix"), first = "C", second = "E", cond = NULL) # True
dSep(as(chestdag, "matrix"), first = "C", second = "E", cond = c("G")) # D-connected F
dSep(as(chestdag, "matrix"), first = "A", second = "G", cond = c("D","E")) # not D-connected T
dSep(as(chestdag, "matrix"), first = "A", second = "G", cond = c("D")) # D-connected F

