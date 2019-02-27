###########################################################
##
## This code is the solution for Question 1 of Homework 1
## dealing with Exploratory data analysis on Student
## Performance data
##
## Created: September 13, 2018
## Name: Shubham Sharma
###########################################################

rm(list = ls())
setwd("~/Documents/SDM/Q1")

############################
# Merge the two data files
############################

d1=read.table("student-mat.csv",sep=";",header=TRUE)
d2=read.table("student-por.csv",sep=";",header=TRUE)

d3=merge(d1,d2,by=c("school","sex","age","address","famsize","Pstatus","Medu","Fedu","Mjob","Fjob","reason","nursery","internet"))
print(nrow(d3)) # 382 students
write.csv(d3, file = "merged.csv", row.names = FALSE)
d3[!complete.cases(d3),] ## checking missing values
summary(d3) ## Getting an idea about the data
############################################################
## Checking distribution of first period grade and absences
############################################################

quartz()
par(mfrow = c(1,4))
densg1x = density(d3$G1.x)
xlimg1x = range(densg1x$x)
ylimg1x = range(densg1x$y)
hist(d3$G1.x,breaks = "Sturges", probability = T, xlim = xlimg1x, ylim = ylimg1x, xlab = "Math G1", main = "Distribution of Grades")
lines(densg1x)
densg1y = density(d3$G1.y)
xlimg1y = range(densg1y$x)
ylimg1y = range(densg1y$y)
hist(d3$G1.y,breaks = "Sturges", probability = T, xlim = xlimg1y, ylim = ylimg1y, xlab = "Portuguese G1", main = "Distribution of Grades")
lines(densg1y)
densabx = density(d3$absences.x)
xlimabx = range(densabx$x)
ylimabx = range(densabx$y)
hist(d3$absences.x,breaks = "Sturges", probability = T, xlim = xlimabx, ylim = ylimabx, xlab = "Absences in Math", main = "Distribution of Absences")
lines(densabx)
densaby = density(d3$absences.y)
xlimaby = range(densaby$x)
ylimaby = range(densaby$y)
hist(d3$absences.y,breaks = "Sturges", probability = T, xlim = xlimaby, ylim = ylimaby, xlab = "Absences in Portuguese", main = "Distribution of Absences")
lines(densaby)


############################################################
## Finding outliers
############################################################
library(dplyr)

jpeg("Distribution_absences_outliers")
par(mfrow = c(1,4))
boxplot(d3$absences.x, horizontal = FALSE, main = "Maths", xlab = "Absences")
boxplot(d3$absences.y, horizontal = FALSE, main = "Portuguese", xlab = "Absences")
stripchart(d3$absences.x, main = "Maths", vertical = TRUE, xlab = "Absences", method = "jitter", col="orange", pch=1)
stripchart(d3$absences.y, main = "Portuguese", vertical = TRUE, xlab = "Absences", method = "jitter", col="orange", pch=1)
dev.off()

dox = select(d3, age, Medu, Fedu, Mjob, Fjob, reason, traveltime.x, studytime.x, famrel.x, freetime.x, goout.x, Dalc.x, Walc.x, health.x, absences.x, G1.x)
doy = select(d3, age, Medu, Fedu, Mjob, Fjob, reason, traveltime.y, studytime.y, famrel.y, freetime.y, goout.y, Dalc.y, Walc.y, health.y, absences.y, G1.y)
quartz()
par(mfrow = c(1,2))
modx = lm(G1.x ~ ., data = dox)
cooksdx = cooks.distance(modx)
plot(cooksdx, pch="*",cex=2, main="Influential Obs by Cooks distance for Maths G1")
abline(h=4*mean(cooksdx, na.rm=T),col="red")
text(x=1:length(cooksdx)+1, y=cooksdx, labels=ifelse(cooksdx>4*mean(cooksdx, na.rm=T),names(cooksdx),""), col="red")
influentialx = as.numeric(names(cooksdx)[(cooksdx>4*mean(cooksdx, na.rm=T))])
dc1 = d3[-influentialx, ]
car::outlierTest(modx) ## Gives the extreme observation of the given model
mody = lm(G1.y ~ ., data = doy)
cooksdy = cooks.distance(mody)
plot(cooksdy, pch="*",cex=2, main="Influential Obs by Cooks distance for Portuguese G1")
abline(h=4*mean(cooksdy, na.rm=T),col="red")
text(x=1:length(cooksdy)+1, y=cooksdy, labels=ifelse(cooksdy>4*mean(cooksdy, na.rm=T),names(cooksdy),""), col="red")
influentialy = as.numeric(names(cooksdy)[(cooksdy>4*mean(cooksdy, na.rm=T))])
dc1 = dc1[-influentialy,]
car::outlierTest(mody)

quartz()
par(mfrow = c(1,2))
densg1xO = density(dc1$G1.x)
xlimg1xO = range(densg1xO$x)
ylimg1xO = range(densg1xO$y)
hist(dc1$G1.x,breaks = "Sturges", probability = T, xlim = xlimg1xO, ylim = ylimg1xO, xlab = "Math G1", main = "For Clean Data")
lines(densg1xO)
densg1yO = density(dc1$G1.y)
xlimg1yO = range(densg1y$x)
ylimg1yO = range(densg1y$y)
hist(dc1$G1.y,breaks = "Sturges", probability = T, xlim = xlimg1yO, ylim = ylimg1yO, xlab = "Portuguese G1", main = "For Clean Data")
lines(densg1yO)

sdabx = sd(d3$absences.x)
sdaby = sd(d3$absences.y)
mabx = mean(d3$absences.x)
maby = mean(d3$absences.y)
dc = d3
dc = as.data.frame(subset(d3, absences.x<((3.5*sdabx)+mabx)))
dc = as.data.frame(subset(dc, absences.y<((3.5*sdaby)+maby)))

quartz()
par(mfrow = c(1,4))
boxplot(d3$G1.x ~ cut(d3$absences.x, 7.5), data = d3, ylab = "G1", xlab = "Absences", main = "Math")
boxplot(d3$G1.y ~ cut(d3$absences.y, 7.5), data = d3, ylab = "G1", xlab = "Absences", main = "Potuguese")
boxplot(d3$G1.x ~ cut(d3$age, pretty(age)), data = d3, ylab = "G1", xlab = "Age", main = "Math")
boxplot(d3$G1.y ~ cut(d3$age, pretty(age)), data = d3, ylab = "G1", xlab = "Age", main = "Potuguese")

quartz()
par(mfrow = c(1,4))
densg1x = density(dc$G1.x)
xlimg1x = range(densg1x$x)
ylimg1x = range(densg1x$y)
hist(dc$G1.x,breaks = "Sturges", probability = T, xlim = xlimg1x, ylim = ylimg1x, xlab = "Math G1", main = "Distribution of Grades")
lines(densg1x)
densg1y = density(dc$G1.y)
xlimg1y = range(densg1y$x)
ylimg1y = range(densg1y$y)
hist(dc$G1.y,breaks = "Sturges", probability = T, xlim = xlimg1y, ylim = ylimg1y, xlab = "Portuguese G1", main = "Distribution of Grades")
lines(densg1y)
densabx = density(dc$absences.x)
xlimabx = range(densabx$x)
ylimabx = range(densabx$y)
hist(dc$absences.x,breaks = 5, probability = T, xlim = xlimabx, ylim = ylimabx, xlab = "Absences in Math", main = "Distribution of Absences")
lines(densabx)
densaby = density(dc$absences.y)
xlimaby = range(densaby$x)
ylimaby = range(densaby$y)
hist(dc$absences.y,breaks = "Sturges", probability = T, xlim = xlimaby, ylim = ylimaby, xlab = "Absences in Portuguese", main = "Distribution of Absences")
lines(densaby)

dc$G2.x = NULL
dc$G2.y = NULL
dc$G3.x = NULL
dc$G3.y = NULL
dc1$G2.x = NULL
dc1$G2.y = NULL
dc1$G3.x = NULL
dc1$G3.y = NULL

############################################################
## Finding relations between variables
############################################################

## Correlation Matrix: Converting to numerical values
library(ggcorrplot)
dc1i = dc1 
dc1i = sapply(dc1,is.factor)
dc1i2 = sapply(dc1[,dc1i],unclass)
dc1in = cbind(dc1[,!dc1i],dc1i2)

dc1inx = select(dc1in, G1.x, age, Medu, Fedu, traveltime.x, studytime.x, failures.x, famrel.x, freetime.x, goout.x, Dalc.x, Walc.x, health.x, absences.x, nursery, internet, schoolsup.x, famsup.x, paid.x, activities.x, higher.x, romantic.x)
dc1iny = select(dc1in, G1.y, age, Medu, Fedu, traveltime.y, studytime.y, failures.y, famrel.y, freetime.y, goout.y, Dalc.y, Walc.y, health.y, absences.y, nursery, internet, schoolsup.y, famsup.y, paid.y, activities.y, higher.y, romantic.y)
quartz()
ggcorrplot(cor(dc1inx), p.mat = cor_pmat(dc1inx), hc.order = TRUE, type='lower')
quartz()
ggcorrplot(cor(dc1iny), p.mat = cor_pmat(dc1iny), hc.order = TRUE, type='lower')

quartz()
par(mfrow = c(1,4))
xrangepx = range(d3$absences.x)
yrangepx = range(d3$G1.x)
plot(d3$G1.x ~ d3$absences.x, data = d3, xlim = xrangepx, ylim = yrangepx, main = "Math G1 Vs. Absences" , xlab = "Absences" , ylab = "G1", pch = 16)
rug(d3$absences.x)
rug(d3$G1.x)

xrangepy = range(d3$absences.y)
yrangepy = range(d3$G1.y)
plot(d3$G1.y ~ d3$absences.y, data = d3, xlim = xrangepy, ylim = yrangepy, main = "Portuguese G1 Vs. Absences" , xlab = "Absences" , ylab = "G1", pch = 16)
rug(d3$absences.y)
rug(d3$G1.y)

xrangepxc = range(dc$absences.x)
yrangepxc = range(dc$G1.x)
plot(dc$G1.x ~ dc$absences.x, data = dc, xlim = xrangepxc, ylim = yrangepxc, main = "G1 Vs. Absences Clean" , xlab = "Absences" , ylab = "Math G1", pch = 16)
rug(dc$absences.x)
rug(dc$G1.x)

xrangepyc = range(dc$absences.y)
yrangepyc = range(dc$G1.y)
plot(dc$G1.y ~ dc$absences.y, data = dc, xlim = xrangepyc, ylim = yrangepyc, main = "G1 Vs. Absences Clean" , xlab = "Absences" , ylab = "Portuguese G1", pch = 16)
rug(dc$absences.y)
rug(dc$G1.y)

quartz()
par(mfrow = c(1,4))
xrangepx = range(d3$absences.x)
yrangepx = range(d3$G1.x)
plot(d3$G1.x ~ d3$absences.x, data = d3, xlim = xrangepx, ylim = yrangepx, main = "Math G1 Vs. Absences" , xlab = "Absences" , ylab = "G1", pch = 16)
rug(d3$absences.x)
rug(d3$G1.x)

xrangepy = range(d3$absences.y)
yrangepy = range(d3$G1.y)
plot(d3$G1.y ~ d3$absences.y, data = d3, xlim = xrangepy, ylim = yrangepy, main = "Portuguese G1 Vs. Absences" , xlab = "Absences" , ylab = "G1", pch = 16)
rug(d3$absences.y)
rug(d3$G1.y)

xrangepxc = range(dc$absences.x)
yrangepxc = range(dc$G1.x)
plot(dc$G1.x ~ dc$absences.x, data = dc, xlim = xrangepxc, ylim = yrangepxc, main = "G1 Vs. Absences Clean" , xlab = "Absences" , ylab = "Math G1", pch = 16)
rug(dc$absences.x)
rug(dc$G1.x)

xrangepyc = range(dc$absences.y)
yrangepyc = range(dc$G1.y)
plot(dc$G1.y ~ dc$absences.y, data = dc, xlim = xrangepyc, ylim = yrangepyc, main = "G1 Vs. Absences Clean" , xlab = "Absences" , ylab = "Portuguese G1", pch = 16)
rug(dc$absences.y)
rug(dc$G1.y)

dgf = subset(d3, guardian.x == 'father')
dgm = subset(d3, guardian.x == 'mother')
quartz()
par(mfrow = c(1,4))
boxplot(G1.x ~ Medu, data = dgm, main = "Math; Guardian: Mother", xlab = "Medu", ylab = "G1")
boxplot(G1.y ~ Medu, data = dgm, main = "Portuguese; Guardian: Mother", xlab = "Medu", ylab = "G1")
boxplot(G1.x ~ Fedu, data = dgf, main = "Math; Guardian: Father", xlab = "Fedu", ylab = "G1")
boxplot(G1.y ~ Fedu, data = dgf, main = "Portuguese; Guardian: Father", xlab = "Fedu", ylab = "G1")

quartz()
par(mfrow = c(1,4))
plot(G1.x ~ Mjob, data = dgm, main = "Math; Guardian: Mother", xlab = "Mjob", ylab = "G1")
plot(G1.x ~ Fjob, data = dgf, main = "Math; Guardian: Father", xlab = "Fjob", ylab = "G1")
plot(G1.y ~ Mjob, data = dgm, main = "Portuguese; Guardian: Mother", xlab = "Mjob", ylab = "G1")
plot(G1.y ~ Fjob, data = dgf, main = "Portuguese; Guardian: Father", xlab = "Fjob", ylab = "G1")

## Changing to log of absences

dc1l = subset(dc1, absences.x > 0 & absences.y > 0)
dc1l[,c(30, 48)] = log(dc1l[,c(30, 48)])
quartz()
par(mfrow = c(1,2))
densabx = density(dc1l$absences.x)
xlimabx = range(densabx$x)
ylimabx = range(densabx$y)
hist(dc1l$absences.x,breaks = "Sturges", probability = T, xlim = xlimabx, xlab = "Absences in Math", main = "Distribution of log of Absences")
lines(densabx)
densaby = density(dc1l$absences.y)
xlimaby = range(densaby$x)
ylimaby = range(densaby$y)
hist(dc1l$absences.y,breaks = "Sturges", probability = T, xlim = xlimaby, xlab = "Absences in Portuguese", main = "Distribution of log of Absences")
lines(densaby)

cor(d3$G1.x, d3$absences.x)
cor(d3$G1.y, d3$absences.y)


############################################################
## Subsetting the data, One-hot encoding
############################################################

library(dplyr)
dohe = select(dc1, sex, age, address, Medu, Fedu, Mjob, Fjob, G1.x, G1.y)
for(unique_value in unique(dohe$Medu)){
  dohe[paste("Medu", unique_value, sep = ".")] = ifelse(dohe$Medu == unique_value, 1, 0)
}
for(unique_value in unique(dohe$sex)){
  dohe[paste("sex", unique_value, sep = ".")] = ifelse(dohe$sex == unique_value, 1, 0)
}
for(unique_value in unique(dohe$age)){
  dohe[paste("age", unique_value, sep = ".")] = ifelse(dohe$age == unique_value, 1, 0)
}
for(unique_value in unique(dohe$address)){
  dohe[paste("address", unique_value, sep = ".")] = ifelse(dohe$address == unique_value, 1, 0)
}
for(unique_value in unique(dohe$sex)){
  dohe[paste("sex", unique_value, sep = ".")] = ifelse(dohe$sex == unique_value, 1, 0)
}
for(unique_value in unique(dohe$sex)){
  dohe[paste("sex", unique_value, sep = ".")] = ifelse(dohe$sex == unique_value, 1, 0)
}
#quartz()
#par(mfrow = c(1,4))
#boxplot(G1.x ~ sex.F, data = dohe, main = "Males vs Females", xlab = "F: 1, M: 0", ylab = "Maths G1")
#boxplot(G1.y ~ sex.F, data = dohe, main = "Males vs Females", xlab = "F: 1, M: 0", ylab = "Portuguese G1")
#boxplot(G1.x ~ address.R, data = dohe, main = "Rural vs Urban", xlab = "R: 1, U: 0", ylab = "Maths G1")
#boxplot(G1.y ~ address.R, data = dohe, main = "Rural vs Urban", xlab = "R: 1, U: 0", ylab = "Portuguese G1")

quartz()
par(mfrow = c(1,4))
boxplot(G1.x ~ sex, data = d3, main = "Math", xlab = "Sex", ylab = "G1")
boxplot(G1.y ~ sex, data = d3, main = "Portuguese", xlab = "Sex", ylab = "G1")
boxplot(G1.x ~ address, data = dohe, main = "Math", xlab = "Address", ylab = "G1")
boxplot(G1.y ~ address, data = dohe, main = "Portuguese", xlab = "Address", ylab = "G1")

############################################################
## Clean Data Set
############################################################
dcf = cbind(dc1, select(dohe, sex.F, sex.M, address.R, address.U, age.15, age.16, age.17, age.18, age.19, age.22, Medu.0, Medu.1, Medu.2, Medu.3, Medu.4) )
dcf[,c(30, 48)] = log((dc1[,c(30, 48)]+1))
dcf$absences.x[is.infinite(dcf$absences.x)] = NA
dcf$absences.y[is.infinite(dcf$absences.y)] = NA
save(dcf, file="dclean.RData")
save.image(file = "Q1.RData")
savehistory()
q()

