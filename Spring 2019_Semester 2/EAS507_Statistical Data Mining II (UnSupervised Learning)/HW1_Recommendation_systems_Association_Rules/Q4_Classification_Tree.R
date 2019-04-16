###########################################################
##
## This code is the solution for Question 4 of Homework 1
## dealing with Classification Trees
## 
## Created: February 21, 2018
## Name: Shubham Sharma
###########################################################

###########################################################
## Attaching Libraries and Data
## 
###########################################################

rm(list = ls())
library("ElemStatLearn")
library("rpart")
attach(marketing)

sum(is.na(marketing))

# Replacing NA values with median

d = marketing
for(i in 1:ncol(d)){
  d[is.na(d[,i]), i] <- median(d[,i], na.rm = TRUE)
}

# Generating reference sample

I = sample(unique(d$Income),size = length(d$Sex),replace = T)
f = data.frame("Income" = I)
f$Sex = sample(unique(d$Sex), size = length(d$Sex), replace = T)
f$Marital = sample(unique(d$Marital), size = length(d$Sex), replace = T)
f$Age = sample(unique(d$Age), size = length(d$Sex), replace = T)
f$Edu = sample(unique(d$Edu), size = length(d$Sex), replace = T)
f$Occupation = sample(unique(d$Occupation), size = length(d$Sex), replace=T)
f$Lived = sample(unique(d$Lived), size = length(d$Sex), replace=T)
f$Dual_Income = sample(unique(d$Dual_Income), size = length(d$Sex), replace=T)
f$Household = sample(unique(d$Household), size = length(d$Sex), replace=T)
f$Householdu18 = sample(unique(d$Householdu18), size = length(d$Sex), replace=T)
f$Status = sample(unique(d$Status), size = length(d$Sex), replace=T)
f$Home_Type = sample(unique(d$Home_Type), size = length(d$Sex), replace=T)
f$Ethnic = sample(unique(d$Ethnic), size = length(d$Sex), replace=T)
f$Language = sample(unique(d$Language), size = length(d$Sex), replace=T)

d["Y"] = 1
f["Y"] = 0
df = rbind(d,f)
df$Y = as.factor(as.character(df$Y))

# Looking into the data

head(df)
summary(df)
str(df)
sum(is.na(df))
#for (i in 1:15){
 # df[,i] = as.factor(df[,i])
#}
###########################################################
## Growing the classification tree
## 
###########################################################

model.control = rpart.control(minsplit = 5, xval = 10, cp = 0) 
# cp 0 since even if no improvement in a branch still wish to include in the model
fit.mar = rpart(Y~., data = df, method = "class", control = model.control)
# path.rpart(fit.mar,nodes = 90)
summary(fit.mar)
names(fit.mar)
fit.mar$cptable

###########################################################
## Pruning the tree
###########################################################

quartz()
plot(fit.mar$cptable[,4], ylab = "CV error")
min_cp = which.min(fit.mar$cptable[,4]) # 32
pruned = prune(fit.mar, cp = fit.mar$cptable[10,1]) # taking 10 nodes

###########################################################
## Plot both the trees: Original and Pruned
###########################################################

quartz()
plot(fit.mar, branch = 0.4, uniform = T, compress = T, main = "Full Tree" )
text(fit.mar, use.n = T, all = T, cex = 0.5)

quartz()
plot(pruned, branch = 0.4, uniform = F, compress = T, main = "Pruned Tree" )
text(pruned, cex = 0.5)

summary(pruned)
path.rpart(pruned,nodes = 31) # predicted class = 1 , prob = 0.956
