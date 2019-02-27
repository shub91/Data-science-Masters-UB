###########################################################
##
## This code is the solution for Question 3 of Homework 5
## 
## 
##
## Created: December 05, 2018
## Name: Shubham Sharma
###########################################################

#######################################
# Loading libraries and attaching data 
#######################################

rm(list = ls())
graphics.off()

library(ElemStatLearn);
library(class);
library(glmnet);
library(pls);
library(leaps);
library(randomForest);
library(neuralnet)
library(Metrics)
data("spam");

#######################################
# Understanding the data
#######################################

str(spam)
summary(spam)

#######################################################
# Splitting into Training and test set (ratio = 7:3 )
#######################################################

spam<-spam[sample(nrow(spam)),];
spami=grep("spam",colnames(spam))
for (o in 1:nrow(spam)) {
  if(spam[o,spami]=="spam"){
    spam[o,spami+1] = 2
  }else{
    spam[o,spami+1] = 3
  }
}
spam<-subset(spam,select = -spam)

#######################################################
# Splitting into Training and test set (ratio = 55:45 )
#######################################################

set.seed(1);
vars=c("A.1","A.2","A.3","A.4","A.5","A.6","A.7","A.8","A.9","A.10","A.11","A.12","A.13","A.14","A.15","A.16","A.17","A.18","A.19","A.20","A.21","A.22","A.23","A.24","A.25","A.26","A.27","A.28","A.29","A.30","A.31","A.32","A.33","A.34","A.35","A.36","A.37","A.38","A.39","A.40","A.41","A.42","A.43","A.44","A.45","A.46","A.47","A.48","A.49","A.50","A.51","A.52","A.53","A.54","A.55","A.56","A.57","V59");
tupleSelc=sample(x=nrow(spam), size=0.55*nrow(spam))
trnset=spam[tupleSelc,vars]
tstset=spam[-tupleSelc,vars]

par(mfrow=c(8,7)) #13 graphs
for(i in names(spam)){
  plot(eval(parse(text=i)) ~ V59, data = spam, xlab="V59", ylab=i)
}
spam2<-spam
spam2[5,1]=10
tupleSelc_oulier=sample(x=nrow(spam2), size=0.55*nrow(spam2))
trnset_outlier=spam[tupleSelc_oulier,vars]
tstset_outlier=spam[-tupleSelc_oulier,vars]
set.seed(1);

n = names(spam)
f = as.formula(paste("V59 ~", paste(n[!n %in% "V59"], collapse = " + ")))

nn.6.original = neuralnet(V59 ~ A.1 + A.2 + A.3 + A.4 + A.5 + A.6 + A.7 + A.8 + A.9 + A.10 + 
                             A.11 + A.12 + A.13 + A.14 + A.15 + A.16 + A.17 + A.18 + A.19 + 
                             A.20 + A.21 + A.22 + A.23 + A.24 + A.25 + A.26 + A.27 + A.28 + 
                             A.29 + A.30 + A.31 + A.32 + A.33 + A.34 + A.35 + A.36 + A.37 + 
                             A.38 + A.39 + A.40 + A.41 + A.42 + A.43 + A.44 + A.45 + A.46 + 
                             A.47 + A.48 + A.49 + A.50 + A.51 + A.52 + A.53 + A.54 + A.55 + 
                             A.56 + A.57
                           , data=trnset
                           , hidden=1
                           , stepmax = 1e+09
                           , linear.output=TRUE)
nn.6.outlier = neuralnet(V59 ~ A.1 + A.2 + A.3 + A.4 + A.5 + A.6 + A.7 + A.8 + A.9 + A.10 + 
                            A.11 + A.12 + A.13 + A.14 + A.15 + A.16 + A.17 + A.18 + A.19 + 
                            A.20 + A.21 + A.22 + A.23 + A.24 + A.25 + A.26 + A.27 + A.28 + 
                            A.29 + A.30 + A.31 + A.32 + A.33 + A.34 + A.35 + A.36 + A.37 + 
                            A.38 + A.39 + A.40 + A.41 + A.42 + A.43 + A.44 + A.45 + A.46 + 
                            A.47 + A.48 + A.49 + A.50 + A.51 + A.52 + A.53 + A.54 + A.55 + 
                            A.56 + A.57
                          , data=trnset_outlier
                          , hidden=1
                          , stepmax = 1e+09
                          , linear.output=TRUE)
spam_original.preds.scaled = round(neuralnet::compute(nn.6.original, tstset[,1:57],rep = 1)$net.result[,1])
spam_oulier.preds.scaled = round(neuralnet::compute(nn.6.outlier, tstset[,1:57],rep = 1)$net.result[,1])
original6err = rmse(tstset$V59, spam_original.preds.scaled)
outlier6err = rmse(tstset$V59, spam_oulier.preds.scaled)
original6err
outlier6err

val_Of_zero_effect=0
for (i in 10:0.25) {
  spam2[5,1]=i
  tupleSelc_oulier = sample(x=nrow(spam2), size=0.55*nrow(spam2))
  trnset_outlier = spam[tupleSelc_oulier,vars]
  tstset_outlier = spam[-tupleSelc_oulier,vars]
  nn.6.outlier = neuralnet(V59 ~ A.1 + A.2 + A.3 + A.4 + A.5 + A.6 + A.7 + A.8 + A.9 + A.10 + 
                              A.11 + A.12 + A.13 + A.14 + A.15 + A.16 + A.17 + A.18 + A.19 + 
                              A.20 + A.21 + A.22 + A.23 + A.24 + A.25 + A.26 + A.27 + A.28 + 
                              A.29 + A.30 + A.31 + A.32 + A.33 + A.34 + A.35 + A.36 + A.37 + 
                              A.38 + A.39 + A.40 + A.41 + A.42 + A.43 + A.44 + A.45 + A.46 + 
                              A.47 + A.48 + A.49 + A.50 + A.51 + A.52 + A.53 + A.54 + A.55 + 
                              A.56 + A.57
                            , data=trnset_outlier
                            , hidden=1
                            , stepmax = 1e+09
                            , linear.output=TRUE)
  
  spam_oulier.preds.scaled = round(neuralnet::compute(nn.6.outlier, tstset[,1:57],rep = 1)$net.result[,1])
  outlier6err<-rmse(tstset$V59, spam_oulier.preds.scaled)
  
  if(outlier6err == original6err){
    val_Of_zero_effect=i;
  }
}
val_Of_zero_effect

# hidden layer used is 1 in place of 4 to minimize computation time