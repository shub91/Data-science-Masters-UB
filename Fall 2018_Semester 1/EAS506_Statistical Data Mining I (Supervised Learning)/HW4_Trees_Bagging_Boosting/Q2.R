###########################################################
##
## This code is the solution for Question 2 of Homework 4
## 
## 
##
## Created: November 24, 2018
## Name: Shubham Sharma
###########################################################

#######################################
# Loading libraries and attaching data 
#######################################

rm(list = ls())

library(rattle)
library(rpart.plot)
library(rpart)
library(party)
library(ISLR)
library(partykit)

#######################################
# Understanding the data
#######################################

set.seed(12345)

wine = read.csv('wine.data.txt', header = FALSE)
d_wine = data.frame(wine[, -c(2, 7, 8)])
colnames(d_wine) = c("WineType", "MalicAcid","Ash", "AlcaAsh", "Mg", "Phenols", "Proa", 
                         "Colour", "Hue", "OD", "Proline")

#Set the seed and put aside the test set
set.seed(12345)
test_in = sample(1:nrow(d_wine), .30*nrow(d_wine))
test = d_wine[test_in, ]
train = d_wine[-test_in, ]

y_true_train = as.factor(train$WineType) #Factors: 1-Barolo; 2-Grignolino; 3-Barbera
y_true_test = as.factor(test$WineType) #Factors: 1-Barolo; 2-Grignolino; 3-Barbera

#######################################
# Classification Tree
#######################################

model_control = rpart.control(minsplit = 5, xval = 10, cp = 0)
fit = rpart(WineType~., data = train, method = "class", control = model_control)

quartz()
fancyRpartPlot(fit)
text(fit, use.n = TRUE, cex = 0.8)

## Tree Pruning

min_cp = which.min(fit$cptable[, 4])
quartz()
plot(fit$cptable[, 4], ylab = "Cp Statistic", type = 'b')
Wine_Model <- prune(fit, cp = fit$cptable[min_cp, 1])

quartz()
fancyRpartPlot(Wine_Model)
text(Wine_Model, use.n = TRUE, all = TRUE,  cex = 0.6)

## Node count in test data
Wine_Model_Nodes = Wine_Model
Wine_Model_Nodes$frame$yval = as.numeric(rownames(Wine_Model_Nodes$frame))
Test_Nodes = predict(Wine_Model_Nodes, test, type="vector")
Test_NodesDF = data.frame(rowNum = c(1:length(Test_Nodes)),Test_Nodes)
test_nodes = data.frame(aggregate(rowNum~Test_Nodes,data = Test_NodesDF,FUN = length))

## Node count in train data
Train_Nodes = predict(Wine_Model_Nodes, train, type="vector")
Train_NodesDF = data.frame(rowNum = c(1:length(Train_Nodes)),Train_Nodes)
train_nodes = data.frame(aggregate(rowNum~Train_Nodes,data = Train_NodesDF,FUN = length))

Wine_Pred_test = predict(Wine_Model, newdata = test, type = 'class')
Wine_Pred_train = predict(Wine_Model, newdata = train, type='class')

## Misclassification error

mismatch = as.numeric(0)
for (i in 1:length(Wine_Pred_test))
{
  if(as.character(y_true_train[i]) != as.character(Wine_Pred_train[i]))
  {
    mismatch = mismatch + 1
  }
}
misclasstrain = mismatch/length(Wine_Pred_train)
mismatch = as.numeric(0)
for (i in 1:length(Wine_Pred_test))
{
  if(as.character(y_true_test[i]) != as.character(Wine_Pred_test[i]))
  {
    mismatch = mismatch + 1
  }
}
misclasstest = mismatch/length(Wine_Pred_test)

misclasstrain # 0.024
misclasstest # 0.0189

## Node count in test data
test_nodes
## Node count in train data
train_nodes