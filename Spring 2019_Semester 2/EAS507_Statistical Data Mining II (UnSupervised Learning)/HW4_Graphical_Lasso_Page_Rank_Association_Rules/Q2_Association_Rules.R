######################################################################
## This code is the solution for Question 2 of Homework 4 which deals                
## with Association rules on Titanic Dataset
## Shubham Sharma                                                   
## Created: April 29, 2019                                          
######################################################################

rm(list = ls())
graphics.off()

##################################################
## Loading the libraries
##################################################

library(Rgraphviz)
library(arules)

##################################################
## Attaching the data
##################################################

mydata = read.csv(file = "train.csv")
summary(mydata)
str(mydata) # 891 obs. of  12 variables

# 'Survived', 'Pclass','Sex','Age','Fare','Embarked' => Only these columns have been selected
# to form arules. These are significant features.

mydata = mydata[,c('Survived', 'Pclass','Sex','Age','Fare','Embarked')]
mydata = mydata[complete.cases(mydata[,c('Survived','Pclass','Sex','Age', 'Fare','Embarked')]),]

# Female = 1, Male = 2
mydata$Sex = as.integer(mydata$Sex)

# No-Value = 1, S = 2, Q = 3, C = 4 
mydata$Embarked = as.integer(mydata$Embarked)

##################################################
## Histograms to create categories for  variables
##################################################

quartz()
h1 = hist(mydata$Survived)

quartz()
h2 = hist(mydata$Pclass)

quartz()
h3 = hist(mydata$Sex)

quartz()
h4 = hist(mydata$Age)

quartz()
h5 = hist(mydata$Fare)

quartz()
h6 = hist(mydata$Embarked)

mydata[['Survived']] = ordered(cut(mydata[['Survived']], c(-1.5,0.5,1.5)), 
                               labels = c('drowned','lived'))

mydata[['Pclass']] = ordered(cut(mydata[['Pclass']], c(-1,1.5,2.5,3.5)), 
                             labels = c('1','2', '3'))

mydata[['Sex']] = ordered(cut(mydata[['Sex']], c(-1,1.5,2.5)), 
                          labels = c('F','M'))

mydata[['Age']] = ordered(cut(mydata[['Age']], c(0,22,45,100)), 
                          labels = c('kid','adult', 'old'))

mydata[['Fare']] = ordered(cut(mydata[['Fare']], c(0,50,150,700)), 
                           labels = c('low','medium', 'high'))

mydata[['Embarked']] = ordered(cut(mydata[['Embarked']], c(-1.5,1.5,2.5,3.5,4.5)), 
                               labels = c('Empty','C', 'Q','S'))

mydata = as(mydata, "transactions")

# Item Frequency Plot

quartz()
itemFrequencyPlot(mydata, support = 0.1, cex.names = 0.7)
setrule = apriori(mydata, parameter = list(support = 0.1, confidence = 0.7, maxlen = 15))
summary(setrule)

# Survived

alive =  subset(setrule, subset = rhs %in% 'Survived=lived' & lift > 1.1)
alive
inspect(head(sort(alive, by = 'confidence'), n=10))


# Did Not Survive

drowned =  subset(setrule, subset = rhs %in% 'Survived=drowned' & lift > 1.4)
drowned
inspect(head(sort(drowned, by = 'confidence'), n=10))