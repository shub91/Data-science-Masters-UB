#####################################################################
## This code is the solution for Question 1 of Homework 3 involving                  
## "cad1" dataset in the package gRbase.        
## Shubham Sharma                                                   
## Created: April 21, 2019                                          
#####################################################################
rm(list = ls())
graphics.off()

##################################################

library(gRain)
#library(RHugin)
library(Rgraphviz)
library(gRbase)
library(ggm)
library(bnlearn)
library(gRim)
library(igraph)

data('cad1', package = "gRbase")
d = cad1
summary(d)
str(d)
head(d)

##################################################
## Constructing the given DAG
##################################################

g = list(~Sex, ~Smoker|Sex, ~SuffHeartF, ~Inherit|Smoker, ~Hyperchol|Smoker:SuffHeartF, ~CAD|Inherit:Hyperchol)
chestdag = dagList(g)

##################################################
## Checking for d-separation
##################################################

dSep(as(chestdag, "matrix"), first = "Smoker", second = "SuffHeartF", cond = NULL) #True
dSep(as(chestdag, "matrix"), first = "Inherit", second = "Hyperchol", cond = c("Smoker")) #True - Common Cause
dSep(as(chestdag, "matrix"), first = "Inherit", second = "Hyperchol", cond = c("CAD")) #FALSE - Common Effect - V-structure thus enables ingluence
dSep(as(chestdag, "matrix"), first = "Inherit", second = "Sex", cond = c("Smoker")) #True - Indirect evidential Effect
dSep(as(chestdag, "matrix"), first = "CAD", second = "Smoker", cond = c("Inherit", "Hyperchol")) # True
dSep(as(chestdag, "matrix"), first = "CAD", second = "Smoker", cond = c("Inherit")) # False
dSep(as(chestdag, "matrix"), first = "Sex", second = "Inherit", cond = c("Smoker")) #True - Indirect Causal Effect
dSep(as(chestdag, "matrix"), first = "Sex", second = "SuffHeartF", cond = NULL) #True
dSep(as(chestdag, "matrix"), first = "Smoker", second = "SuffHeartF", cond = NULL) #True
dSep(as(chestdag, "matrix"), first = "Inherit", second = "SuffHeartF", cond = NULL) #True

##################################################
## Conditional Probability Tables
##################################################

all = xtabs(~Sex, data = cad1[,c("CAD","Hyperchol","Sex","Smoker","SuffHeartF","Inherit")])
s.sx = xtabs(~Smoker+Sex, data = cad1[,c("CAD","Hyperchol","Sex","Smoker","SuffHeartF","Inherit")])
i.sm = xtabs(~Inherit+Smoker , data = cad1[,c("CAD","Hyperchol","Sex","Smoker","SuffHeartF","Inherit")])
c.ih = xtabs(~CAD+Inherit+Hyperchol, data = cad1[,c("CAD","Hyperchol","Sex","Smoker","SuffHeartF","Inherit")])
h.sm.su = xtabs(~Hyperchol+Smoker+SuffHeartF, data = cad1[,c("CAD","Hyperchol","Sex","Smoker","SuffHeartF","Inherit")])
su_obj = xtabs(~SuffHeartF, data = cad1[,c("CAD","Hyperchol","Sex","Smoker","SuffHeartF","Inherit")])

##################################################
## Build the network
##################################################

plist = compileCPT(list(all, s.sx, i.sm, c.ih, h.sm.su, su_obj))
grn1 = grain(plist)
summary(grn1)

# Plotting the network
quartz()
plot(grn1)

##################################################
## Compile the network 
## DAG is created, moralized, and triangularized.
##################################################
grn1c = compile(grn1)
summary(grn1c)

# if interested in "haulting" the compilation process
# g <- grn1$dag
# mg <- moralize(g)
# tmg <- triangulate(mg)
# rip(tmg)
# 
# plot the junction tree
# quartz()
# plot(grn1c, type = "jt")
# 
# names(grn1c)

##################################################
## Propagate the the network 
##################################################

grn1c = propagate(grn1c)
summary(grn1c)

##################################################
## Make Queries 
##################################################

# Absorbing the evidence looking at marginal distribution 

grn1c.ev = setFinding(grn1c, nodes = c("Sex", "Hyperchol"), states = c("Female", "Yes"))

# probabilistic query, given No evidence
querygrain(grn1c, nodes = c("SuffHeartF", "CAD"), type = "marginal")
# probabilistic query, given evidence
querygrain(grn1c.ev, nodes = c("SuffHeartF", "CAD"), type = "marginal")

# abs$SuffHeartF
# not_abs$SuffHeartF
# abs$CAD
# not_abs$CAD

getFinding(grn1c.ev)

# Absorbing the evidence looking at joint distribution 

# probabilistic query, given No evidence
querygrain(grn1c, nodes = c("SuffHeartF", "CAD"), type = "joint")

# probabilistic query, given evidence
querygrain(grn1c.ev, nodes = c("SuffHeartF", "CAD"), type = "joint")

# Absorbing the evidence looking at conditional distribution 

# probabilistic query, given No evidence
querygrain(grn1c, nodes = c("SuffHeartF", "CAD"), type = "conditional")

# probabilistic query, given evidence
querygrain(grn1c.ev, nodes = c("SuffHeartF", "CAD"), type = "conditional")

##################################################
## Simulating new dataset with 5 observations
##################################################
set.seed(12345)
d_sim = simulate(grn1c.ev, nsim = 5)
d_sim
p = predict(grn1c, response = c("Smoker","CAD"), newdata = d_sim, predictors = c("Sex", "SuffHeartF", "Hyperchol", "Inherit"), type = "class")

# Probabilities
predict(grn1c, response = c("Smoker","CAD"), newdata = d_sim, predictors = c("Sex", "SuffHeartF", "Hyperchol", "Inherit"), type = "distribution")

names(p)
p

##################################################
## Simulating new dataset with 500 observations
##################################################

d_sim_500 = simulate(grn1c.ev, nsim = 500)
y_hat = predict(grn1c, response = c("Smoker","CAD"), newdata = d_sim_500, predictors = c("Sex","SuffHeartF","Hyperchol","Inherit"), type = "class")

#Probability of Smoker and CAD
predict(grn1c, response = c("Smoker","CAD"), newdata = d_sim_500, predictors = c("Sex","SuffHeartF","Hyperchol","Inherit"), type = "distribution")
rm(g, chestdag, grn1, grn1c, grn1c.ev, plist, d_sim, c.ih, h.sm.su, i.sm, all, su_obj, s.sx)
save(d_sim_500, file="Q1p4.RData")

##################################################
## Misclassification Rate
##################################################

# Smoker - Accuracy
tab_1 = table(d_sim_500$Smoker, y_hat$pred$Smoker)
(1-sum(diag(tab_1))/500)*100

# CAD - Accuracy
tab_2 = table(d_sim_500$CAD, y_hat$pred$CAD)
(1-sum(diag(tab_2))/500)*100

