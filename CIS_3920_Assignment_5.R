#Sangie Cheung
#Assignment #5: Classification & Clustering

#Part 1: Classification---------------------------------------------------------
#Section 1: The Data
getwd()
setwd("C:/Users/Honors/Downloads")
data=read.csv("telemarketing.csv")

str(data)
data$housing = as.factor(data$housing)
data$contact = as.factor(data$contact)
data$y = as.factor(data$y)

set.seed(1)
train.index = sample(1:nrow(data),nrow(data)*0.80) #80/20 split for train and test set
train = data[train.index,]#train set
test = data[-train.index,]#test set
data.frame(full.size=nrow(data),train.size=nrow(train),test.size=nrow(test))

#Section 2: Logistic Regression
##2A Model Building
model1 = glm(y~.,data=train,family=binomial) #regression on all predictors
summary(model1)

##2B Interpreting Coefficients
exp(coef(model1)) #convert coefficients from log-odds to odds-ratio 

##2C Model Comparison
model2 = glm(y~duration+campaign+contact,data=train,family=binomial) #reduced model
summary(model2)
data.frame(full.model=AIC(model1),reduced.model=AIC(model2)) #compare AIC

##2D Predictions
pred.prob = predict(model1,test,type="response") #predict y on test
pred.prob[1:10] # View first 10 predictions
pred.class = pred.prob

#threshold of 50%
pred.class[pred.prob>0.5] = "yes" 
pred.class[!pred.prob>0.5] = "no"
pred.class[1:10] # View first 10 predictions

c.matrix = table(actual=test$y,pred.class);c.matrix #confusion matrix

acc = mean(pred.class==test$y) #accuracy
sens.yes = c.matrix[4]/(c.matrix[2]+c.matrix[4]) #sensitivity of "yes"
prec.yes = c.matrix[4]/(c.matrix[3]+c.matrix[4]) #precision of "yes"
data.frame(acc,sens.yes,prec.yes)

#threshold of 80%
pred.class[pred.prob>0.8] = "yes" 
pred.class[!pred.prob>0.8] = "no"
pred.class[1:10] # View first 10 predictions

c.matrix = table(actual=test$y,pred.class);c.matrix #confusion matrix

acc = mean(pred.class==test$y) #accuracy
sens.yes = c.matrix[4]/(c.matrix[2]+c.matrix[4]) #sensitivity of "yes"
prec.yes = c.matrix[4]/(c.matrix[3]+c.matrix[4]) #precision of "yes"
data.frame(acc,sens.yes,prec.yes)

#Section 3: Random Forest
set.seed(1)
train.index = sample(1:nrow(data),nrow(data)*0.80) #80/20 split for train and test set
train = data[train.index,]#train set
test = data[-train.index,]#test set
data.frame(full.size=nrow(data),train.size=nrow(train),test.size=nrow(test))

#classification random forest on training set
library(randomForest)
cl.model = randomForest(y~.,data=train,importance=TRUE);cl.model 

#prediction on test set and confusion matrix
pred.cl = predict(cl.model,test)
c.matrix = table(test$y,pred.cl); c.matrix
acc = mean(pred.class==test$y)
sens.yes = c.matrix[4]/(c.matrix[2]+c.matrix[4])
prec.yes = c.matrix[4]/(c.matrix[3]+c.matrix[4])
data.frame(acc,sens.yes,prec.yes)

#Variable Important Plot
varImpPlot(cl.model,main="Variable Importance Plots")

#Part 2: Clustering-------------------------------------------------------------
#Data Preparation
library(ISLR)   #Loaded ISLR package
str(College)
data2=College[,2:18] #drop Private variable
scale.data=scale(data2) #standardize the data to mean 0 and standard deviation 1 
summary(scale.data)

#K-means clustering
set.seed(1)
km = kmeans(scale.data, 2, nstart=10) #create 2 clusters with 10 rerun k-means
km$cluster #Cluster assignments

km$withinss #WSS for each cluster
km$tot.withinss #Total WSS summed across all 
km$totss #Total sum of squares for entire dataset

clust.data = cbind(cluster=km$cluster,data2) #merge cluster with original data
head(clust.data)
aggregate(.~cluster,data=clust.data,FUN=mean) #mean by cluster

#Hierarchical clustering
hc.dist = dist(scale.data) #Calculate distance matrix from standardized data
hc = hclust(hc.dist,method="complete"); hc #Cluster using the "complete" linkage method
plot(hc)
abline(h=18,col="red") #line at h=18
hc.cut = cutree(hc,2); hc.cut #Cut tree at 2 clusters
hc.data = cbind(cluster=hc.cut,data2) 
aggregate(.~cluster,data=hc.data,FUN=mean) #mean by cluster
