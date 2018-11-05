############################ SVM Digit Recogniser #################################
# 1. Business Understanding
# 2. Data Understanding
# 3. Data Preparation
# 4. Model Building 
#  4.1 Linear kernel
#  4.2 RBF Kernel
# 5 Hyperparameter tuning and cross validation

#####################################################################################

# 1. Business Understanding: 

#The objective is to identify each of a large number of black-and-white
#rectangular pixel displays as one of the 10 digits in number system

#####################################################################################


#Loading Neccessary libraries

library(kernlab)
library(readr)
library(caret)
library(e1071)

#Loading Data (no headers in the data)

trainData<- read.csv("train.csv",stringsAsFactors = F)

set.seed(1)

sampleData<-trainData[sample(nrow(trainData),1000),]

str(sampleData)

#EDA on the data
# Let's ensure that the sample dataset we have picked is balanced across
# various labels

table(sampleData$label)
# 0   1   2   3   4   5   6   7   8   9 
# 99 105 105 117  97  76  93 102  90 116 

#Understanding Dimensions

dim(sampleData)

# printing few rows and few cols
head(sampleData[,150:200])

#checking missing value

sum(sapply(sampleData, function(x) sum(is.na(x))))
#No NA in the data


#Coverting our target variable into a factor variable

sampleData$label<-factor(sampleData$label)


# Split the sampleData into train and validation set

set.seed(1)
sampleData.indices = sample(1:nrow(sampleData), 0.8*nrow(sampleData))
train = sampleData[sampleData.indices, ]
validate = sampleData[-sampleData.indices, ]


#Constructing Models

set.seed(2)
#Using Linear SVM Kernel 
Model_linear <- ksvm(label~ ., data = train, scale = FALSE, kernel = "vanilladot")

Model_linear #Parameter cost C = 1

Eval_linear<- predict(Model_linear, validate[,-1])

#confusion matrix - Linear Kernel
conf_linear<-confusionMatrix(Eval_linear,validate$label)

conf_linear #Accuracy 86% for 800 obs in training dataset


#--------------------------------------------------------------

set.seed(3)

#Using Polynomial SVM Kernel with default degree = 1
Model_Poly <- ksvm(label~ ., data = train, scale = FALSE, kernel = "polydot")

Model_Poly #Default Hyperparameters : degree =1, scale =1 , offset = 1

Eval_poly<- predict(Model_Poly, validate[,-1])

#confusion matrix - Linear Kernel
conf_poly<-confusionMatrix(Eval_poly,validate$label)

conf_poly #Accuracy 86% for 800 obs in training dataset

# These results are similar to that of linear model. 
# Does polydot kernel with degree of 1 behaves similar to linear kernel?
# Seemslike it!

#------------------------------------------------------------

set.seed(4)
#Using Polynomial SVM Kernel with degree = 2
Model_Poly2 <- ksvm(label~ ., data = train, scale = FALSE, kernel = polydot(degree=2))

Model_Poly2 #Hyperparameters : degree =2, scale =1 , offset = 1

Eval_poly2<- predict(Model_Poly2, validate[,-1])

#confusion matrix - Linear Kernel
conf_poly2<-confusionMatrix(Eval_poly2,validate$label)

conf_poly2 #Accuracy 89% for 800 obs in training dataset
# slightly better results than linear kernel


#------------------------------------------------------------------

set.seed(4)
#Using RBF SVM Kernel
Model_RBF <- ksvm(label~ ., data = train, scale = FALSE, kernel = "rbfdot")

Model_RBF
# Hyperparameter: sigma for Model_RBF is 1.6e-7 i.e. a very small value
# Small value of sigma shows that kernel is of low degree complexity. 
# Is it overfitting the data?

Eval_RBF<- predict(Model_RBF, validate[,-1])

#confusion matrix - RBF Kernel
conf_RBF<-confusionMatrix(Eval_RBF,validate$label)

conf_RBF #Accuracy 87.5% for 800 obs in training dataset.

#Based on validation dataset it seems poly kernel with degree 2 performs better
#as compared to linear and RBF
 


############   Hyperparameter tuning and Cross Validation #####################

# Let's try to tune the hyperparameters for various models and 
# measure their accuracy to see if we can find a better fit than 89% accuracy
# Also, we will find if our models are stable using cross validation

#traincontrol function Controls the computational nuances of the train function.
# i.e. method =  CV means  Cross Validation.
#      Number = 5 implies Number of folds in CV.

trainControl <- trainControl(method="cv", number=5)

# Metric <- "Accuracy" implies our Evaluation metric is Accuracy.

metric <- "Accuracy"

#method = "svmLinear", "svmPoly" , and "svmRadial"

#-------------------------------------------------------------------

#Let's start method="svmLinear" to see if we can tune it to provide better
#performance and how stable it is. 

set.seed(1)


#Tuning parameter is C(cost)
grid <- expand.grid(.C=c(0.01,0.1,1,10))

fit.svmLinear <- train(label~., data=train, method="svmLinear", metric=metric, 
                 tuneGrid=grid, trControl=trainControl)

fit.svmLinear

#Accuracy was used to select the optimal model using the largest value.
#The final value used for the model was C = 0.01.

print(fit.svmLinear) 
#Accuracy was 86.7 for 5 fold validation with training dataset of 800 rows
#It didn't vary with the chosen value of C. 

plot(fit.svmLinear)

#Accuracy of 86.7% with cv shows svm with linear kernel is a stable model 

#Let's validate the accuracy against the validation set
eval_fit.svmLinear<- predict(fit.svmLinear, validate[,-1])

#confusion matrix 
conf_svmLinear<-confusionMatrix(eval_fit.svmLinear,validate$label)

conf_svmLinear 
#Accuracy is 86% .Same as what we found earlier for linear kernel


#--------------------------------------------------------------------------

#Let's try method="svmPoly" with degrees = 1, 2, 3 and 4 and to see how it fits.

set.seed(2)

#Tuning parameters are degree, scale, and C(cost)
grid <- expand.grid(.degree=c(1,2,3,4), .C=c(0.01,0.1,1,10), .scale = 1)

fit.svmPoly <- train(label~., data=train, method="svmPoly", metric=metric, 
                 tuneGrid=grid, trControl=trainControl)
fit.svmPoly

#Tuning parameter 'scale' was held constant at a value of 1
#Accuracy was used to select the optimal model using the largest value.
#The final values used for the model were degree = 2, scale = 1 and C = 0.01.

print(fit.svmPoly) 

# Accuracy at degree 2 and C=.01 is 87.1% 
# svmPoly chose degree 2 over degree 1, 3 and 4. 
# This tells us that 2nd degree polynomial is a better fitting model 
# for this dataset as compared to linear, 3rd degree or 4th degree polynomial
# This also tells us that any higher degree polynomial may be overfitting data
# and hence produces worse performance than 2nd degree during cross validation
# Also, it seems that there was not much variation with respect to the
# range of .C value that we provided in the grid.

plot(fit.svmPoly)

eval_fit.svmPoly<- predict(fit.svmPoly, validate[,-1])

conf_svmPoly<-confusionMatrix(eval_fit.svmPoly,validate$label)

conf_svmPoly 
#Accuracy is 89% with deg=2 and C=.01 same as we found for kernel polydot with deg 2.


#--------------------------------------------------------------------------------


# Let's tune svmRadial kernel as well as see how it performs

set.seed(3)
grid <- expand.grid(.sigma=c(10^-7,10^-3,10^-1,10^2), .C=c(0.01,0.1,1,10) )

#Let's try method="svmRadial" to see how it fits.

fit.svmRadial <- train(label~., data=train, method="svmRadial", metric=metric,
                       tuneGrid=grid, trControl=trainControl)

fit.svmRadial

#Accuracy was used to select the optimal model using the largest value.
#The final values used for the model were sigma = 1e-07 and C = 10.

print(fit.svmRadial) 
# Accuracy is 88.0% at the chosen value of sigma =1e-07 and C=10
# Accuracy for sigma = 1e-03 or higher drops drastically to 17% & lower
# Accuracy also drop drastically for C = .1 or .01 
# We know that C controls the width of the margin.
# A bigger value of c (c=10) conveys a very thin margin was used to
# produce 88.9% accuracy.

plot(fit.svmRadial)

eval_fit.svmRadial<- predict(fit.svmRadial, validate[,-1])

conf_svmRadial<-confusionMatrix(eval_fit.svmRadial,validate$label)

conf_svmRadial 

# Accuracy is 88% with sigma of 1e-07 and C=10

# It seems that Radial Kernel with sigma of 1e-07 and C=10 produces
# similar (but slightly worse 88% vs 89%) accuracy with Polynomial Kernel 
#of degree 2 and C=.01

# We know that polynomial kernel with degree 2 is a simpler model than
# radial kernel with sigma 1e-07. 

# We also know that C is indicator for penalty for missclassification
# Polynomial kernel is choosing a lower value of C.
# Thus allowing for missclassifications and wider margin 
# On the other hand svmRadial is choosing a higher value of C and 
# allowing for very narrow margin. 

# Theoretically Polynomial kernel which simpler and has lower value of C
# will generalize better than Radial.

#----------------------------------------------------------------------------

# Let's confirm if our conclusion is correct with a larger dataset in order 
# to solidify our conclusion. 
# Also we want to eliminate the possiblity of having
# complex model sub-perform on test data 
# due to small size of training dataset resulting in overfitting training data

sampleData2<-trainData[sample(nrow(trainData),10000),]

#Let's call the first column which are labels as "label"

#colnames(sampleData2)[1] <-"label"

sampleData2$label<-factor(sampleData2$label)

set.seed(1)
sampleData2.indices = sample(1:nrow(sampleData2), 0.8*nrow(sampleData2))
train = sampleData2[sampleData2.indices, ]
validate = sampleData2[-sampleData2.indices, ]

Model_Poly_1 <- ksvm(label~ ., data = train, scale = FALSE, kernel = polydot(degree=2), C=.01)
Model_Poly_1 #degree 2, scale =1 and Cost C =.01
Model_RBF_1 <- ksvm(label~ ., data = train, scale = FALSE, kernel = rbfdot(sigma=10^-7), C=10)
Model_RBF_1  #sigma = 1e-07 and C =10
#Let's validate the accuracy against the validation set

eval_fit.Model_Poly_1<- predict(Model_Poly_1, validate[,-1])
eval_fit.Model_RBF_1<- predict(Model_RBF_1, validate[,-1])

conf_Poly_1<-confusionMatrix(eval_fit.Model_Poly_1,validate$label)

conf_RBF_1<-confusionMatrix(eval_fit.Model_RBF_1,validate$label)

conf_Poly_1 #Accuracy 96.2% 
conf_RBF_1  #Accuracy 96.3%

# No signficant improvement in accruacy from Radial (complex) model
# despite using larger dataset. 
# We will pick our final model to be Model_Poly_1 because 
# it is a relatively simpler model and should generalize better than 
# complex model such as Model_RBF_1 

final_model<-Model_Poly_1

#--------------------------------------------------------------------

# #Lets predict the accuracy of our chosen model against the test dataset
# 
# testData<- read.csv("SVM Dataset/mnist_test.csv",stringsAsFactors = F, header = F)
# 
# str(testData)
# 
# colnames(testData)[1]<-"label"
# 
# testData$label<-as.factor(testData$label)
# 
# test_pred<-predict(final_model,testData[,-1])
# 
# conf_test<-confusionMatrix(test_pred,testData$label)
# 
# conf_test #Accuracy 96%

# Conclusion: 
#====================
# Our chosen svm model is of polynomial kernel with degree 2 and C =.01.
# It produces an accuracy of ~96%. 

#-----Predicting digits in test.csv of Kaggle project---------

testDigits<-read.csv("test.csv")

#View(testDigits)

test_pred<-predict(final_model,testDigits)

str(test_pred)

test_pred_df<-data.frame(Label=test_pred)

write.csv(test_pred,"test_upload.csv")


