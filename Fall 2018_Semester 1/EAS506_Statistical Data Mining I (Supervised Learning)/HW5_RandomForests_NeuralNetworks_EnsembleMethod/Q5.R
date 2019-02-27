###########################################################
##
## This code is the solution for Question 5 of Homework 5
## 
## 
##
## Created: December 05, 2018
## Name: Shubham Sharma
###########################################################

#######################################
# Loading libraries and attaching data 
#######################################

# install.packages("devtools")
library(devtools)
# install_github("vqv/ggbiplot")
library(ggbiplot)
library(mclust)
rm(list = ls())
graphics.off()
data(banknote)

#######################################
# Understanding the data
#######################################

str(banknote)
summary(banknote)

#######################################################
# Splitting into Training and test set (ratio = 2:1 )
#######################################################

b_gen = subset(banknote, banknote$Status == "genuine")
b_fake = subset(banknote,banknote$Status == 'counterfeit')

gen_pca = prcomp(b_gen[,c(2:7)] , center = TRUE)
fake_pca = prcomp(b_fake[,c(2:7)] , center = TRUE)
all_pca = prcomp(banknote[,c(2:7)] , center = TRUE)

summary(gen_pca) # 92% variance in first 4 PCs
summary(fake_pca) # 90% in first 3 PCs
summary(all_pca) # 93% in first 3 PCs

###########################################################
# Loadings of different principle components for each case
###########################################################

print(gen_pca)
print(fake_pca)
print(all_pca)

###########################
# Plots for Comparison
###########################

# Principal Components by reduction in variance
quartz()
par(mfrow=c(1,3))
plot(gen_pca, type = "l", main = "Genuine Notes")
plot(fake_pca, type = "l", main = "CounterFeit Notes")
plot(all_pca, type = "l", main = "All Notes")

# Comparison of the principal components

quartz()
ggbiplot(gen_pca, ellipse = TRUE, circle = TRUE)
quartz()
ggbiplot(fake_pca, ellipse = TRUE, circle = TRUE)
quartz()
ggbiplot(all_pca, ellipse = TRUE, circle = TRUE)

