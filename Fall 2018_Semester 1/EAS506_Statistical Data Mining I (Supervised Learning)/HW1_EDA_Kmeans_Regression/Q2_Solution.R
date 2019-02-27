###########################################################
##
## This code is the solution for Question 2 of Homework 1
## dealing with Exploratory data analysis on Student
## Performance data
##
## Created: September 13, 2018
## Name: Shubham Sharma
###########################################################

rm(list = ls())
setwd("~/Documents/SDM/Q2")

#####################################################
# Finding relation between predictors and response
#####################################################

attach(dcf)

#### Converting all values to numerical
library(dplyr)
dcfi = sapply(dcf,is.factor)
dcfi2 = sapply(dcf[,dcfi],unclass)
dcfin = cbind(dcf[,!dcfi],dcfi2)

names(d1)
drop = c("sex.F", "sex.M", "address.R", "address.U", "age.15", "age.16", "age.17", "age.18", "age.19", "age.22", "Medu.0", "Medu.1", "Medu.2", "Medu.3", "Medu.4")
dcfin = dcfin[,!(names(dcfin) %in% drop)]
## Correlation values and p-values
library("Hmisc")

dcfcorr = round(cor(dcfin),3)

flat_cor_mat <- function(cor_r, cor_p){
  #This function provides a simple formatting of a correlation matrix
  #into a table with 4 columns containing :
  # Column 1 : row names (variable 1 for the correlation test)
  # Column 2 : column names (variable 2 for the correlation test)
  # Column 3 : the correlation coefficients
  # Column 4 : the p-values of the correlations
  library(tidyr)
  library(tibble)
  cor_r <- rownames_to_column(as.data.frame(cor_r), var = "row")
  cor_r <- gather(cor_r, column, cor, -1)
  cor_p <- rownames_to_column(as.data.frame(cor_p), var = "row")
  cor_p <- gather(cor_p, column, p, -1)
  cor_p_matrix <- left_join(cor_r, cor_p, by = c("row", "column"))
  cor_p_matrix
}

cor_3 <- rcorr(as.matrix(dcfin))

my_cor_matrix <- flat_cor_mat(cor_3$r, cor_3$P)
head(my_cor_matrix)

#### Regressing G1.x on all predictors 
lm.fit1x = lm(G1.x~., data=dcfin)
summary(lm.fit1x)
lm.fit1y = lm(G1.y~., data=dcfin)
summary(lm.fit1y)

dcfin1xp = data.frame(summary(lm.fit1x)$coef[summary(lm.fit1x)$coef[,4] <= .5, 4])
dcfin1x = select (dcfin, G1.x, age, Fedu, studytime.x, failures.x, failures.y, freetime.x, freetime.y, Dalc.x, absences.x, studytime.y, goout.y, goout.x, Dalc.y, absences.y, G1.y, school, sex, Fjob, reason, nursery, schoolsup.x, famsup.x)
dcfin1yp = data.frame(summary(lm.fit1y)$coef[summary(lm.fit1y)$coef[,4] <= .5, 4])
dcfin1y = select(dcfin, G1.y, age, Medu, traveltime.x, traveltime.y, studytime.x, freetime.x, failures.y, freetime.y, Dalc.x, studytime.y, goout.x, Dalc.y, absences.y, G1.x, school, sex, address, famsize, Fjob, reason, nursery, internet, guardian.x, schoolsup.x, famsup.x, activities.x, higher.x, paid.y)

## Second iteration
lm.fit2x = lm(G1.x~., data=dcfin1x)
summary(lm.fit2x)
lm.fit2y = lm(G1.y~., data=dcfin1y)
summary(lm.fit2y)

## Adding Interaction

summary(lm(G1.x~.+ goout.x:studytime.x,data=dcfin1x))
summary(lm(G1.y~.+ goout.y:studytime.y,data=dcfin1y))
summary(lm(G1.x~.+ goout.x:absences.x,data=dcfin1x))
summary(lm(G1.y~.+ goout.y:absences.y,data=dcfin1y))
summary(lm(G1.x~.+ absences.x:studytime.x,data=dcfin1x))
summary(lm(G1.y~.+ absences.y:studytime.y,data=dcfin1y))
summary(lm(G1.x~.+ freetime.x:studytime.x,data=dcfin1x))
summary(lm(G1.y~.+ freetime.y:studytime.y,data=dcfin1y))

summary(lm(G1.x~.+ Dalc.x:studytime.x,data=dcfin1x))
summary(lm(G1.y~.+ Dalc.y:studytime.y,data=dcfin1y))

summary(lm(G1.y~.+ goout.y:absences.y:studytime.y,data=dcfin1y))
summary(lm(G1.x~.+ goout.x:studytime.x:absences.x,data=dcfin1x))
summary(lm(G1.y~.+ goout.y:freetime.y:studytime.y,data=dcfin1y))
summary(lm(G1.x~.+ goout.x:studytime.x:freetime.x,data=dcfin1x))
summary(lm(G1.y~.+ goout.y:freetime.y:absences.y,data=dcfin1y))
summary(lm(G1.x~.+ goout.x:absences.x:freetime.x,data=dcfin1x))

summary(lm(G1.y~.+ goout.y:freetime.y:absences.y+ goout.x:studytime.x,data=dcfin1y))

savehistory()
q()