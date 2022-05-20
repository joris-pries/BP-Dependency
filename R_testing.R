# library('Boruta')
# set.seed(777)
# #Boruta on the "small redundant XOR" problem; read ?srx for details
data(srx)
# Boruta(Y~.,data=srx)->Boruta.srx
# 
# 

library('caret')
library(mlbench)
library(gbm)
data(Sonar)
set.seed(998)
inTraining <- createDataPartition(Sonar$Class, p = .75, list = FALSE)
training <- Sonar[ inTraining,]
testing  <- Sonar[-inTraining,]

fitControl <- trainControl(method = "repeatedcv",
                           number = 10,
                           repeats = 10,
                           ## Estimate class probabilities
                           classProbs = TRUE,
                           ## Evaluate performance using 
                           ## the following function
                           summaryFunction = twoClassSummary)



set.seed(825)
gbmFit3 <- train(Class ~ ., data = training, 
                 method = "gbm", 
                 trControl = fitControl, 
                 verbose = FALSE, 
                 ## Specify which metric to optimize
                 metric = "ROC")
gbmFit3


gbmImp <- varImp(gbmFit3, scale = FALSE)
gbmImp


library(tidyverse)
tag <- read.csv("C:/Users/Joris/Downloads/tag_data.csv", row.names = 1)
tag <- as.matrix(tag)

## Select only models for regression
regModels <- tag[tag[,"Regression"] == 1,]

dataset <- Sonar







method_list = c('vglmAdjCat',
                #  'treebag',
                 # 'bagFDA',
                #  'bagEarth',
                #  'bagEarthGCV',
                #  'bayesglm',
                #  'gamboost',
                  'glmboost',
                  'blackboost')
# method_list = c('vglmAdjCat')
for (method in method_list) {
  tryCatch({
    fitted_model <- train(Class ~ ., data = dataset, 
                   method = method, 
                   verbose = FALSE,
                   trControl = fitControl, 
                   ## Specify which metric to optimize
                   metric = "ROC")
    
    Imp <- varImp(fitted_model, scale = FALSE)
    print(Imp)
  })
}


library(reticulate)
pd <- import("pandas")
pickle_data <- pd$read_pickle('E:/OneDrive/PhD/GitHub/Official_Feature_Importance/src/bp_feature_importance/datasets/decimal_system.pickle')

arr1 <- array(unlist(pickle_data[1]), dim= c(200,3))
colnames(arr1) <- c('1', '2', '3')
arr2 <- array(unlist(pickle_data[2]), dim= c(200))
colnames(arr2) <- c("Y")

dataset <- cbind(arr1, arr2)
colnames(dataset) <- c('1', '2', '3', 'Class')
#colClasses=c("Class"="character")


dataset<- as.data.frame(dataset)



classification <- F

if (classification == T) {
  dataset$Class <- as.factor(dataset$Class)
  arr2 <- as.factor(arr2)
  fitControl <- trainControl(method = "none", classProbs = F)
}

if (classification == F) {
  fitControl <- trainControl(method = "none", classProbs = F)
}



arr1 <- as.data.frame(arr1)
arr2 <- as.numeric(arr2)
# arr2 <- as.data.frame(arr2)
rdaFit <- train(x= arr1, y= arr2, 
                method = "adaboost", 
                tuneGrid=data.frame(nIter=1, method= '1'),
                trControl = fitControl,
                metric= 'Accuracy'
)






# 
# rdaFit <- train(Class ~ ., data= dataset, 
#                 method = "enet", 
#                 trControl = fitControl
#                 )

Imp <- varImp(rdaFit, scale = FALSE)
print(Imp$importance)


install.packages(c('CHAID', 'FCNN4R', 'HDclassif', 'KRLS', 'LiblineaR', 'LogicReg', 'RRF', 'RSNNS', 'RWeka', 'Rborist', 'ada', 'adabag', 'adaptDA', 'bartMachine', 'binda', 'bnclassify', 'brnn', 'bst', 'caTools', 'deepboost', 'deepnet', 'elmNN', 'evtree', 'extraTrees', 'fastAdaboost', 'fastICA', 'foba', 'frbs', 'gam', 'glmnet', 'gpls', 'h2o', 'hda', 'inTrees', 'infotheo', 'keras', 'kerndwd', 'kknn', 'kohonen', 'leaps', 'logicFS', 'monmlp', 'monomvn', 'msaenet', 'mxnet', 'naivebayes', 'neuralnet', 'nodeHarvest', 'obliqueRF', 'ordinalForest', 'ordinalNet', 'pamr', 'penalized', 'penalizedLDA', 'plsRglm', 'protoclass', 'qrnn', 'quantregForest', 'rFerns', 'randomForest', 'randomGLM', 'relaxo', 'rfUtilities', 'robustDA', 'rocc', 'rotationForest', 'rpartScore', 'rqPen', 'rrcov', 'rrcovHD', 'rrlda', 'sda', 'sdwd', 'snn', 'sparseLDA', 'sparsediscrim', 'spikeslab', 'spls', 'stepPlr', 'superpc', 'supervisedPRIM', 'vbmp', 'wsrf', 'xgboost'))







library('caret')
library(reticulate)
library(plyr)
library('vip')
pd <- import("pandas")
pickle_data <- pd$read_pickle('E:/OneDrive/PhD/GitHub/Official_Feature_Importance/src/bp_feature_importance/datasets/decimal_system.pickle')

arr1 <- as.data.frame(pickle_data[1])
colnames(arr1) <- c('1', '2', '3')
arr2 <- array(unlist(pickle_data[2]), dim= c(200))
#colnames(arr2) <- c("Y")


classification <- F

if (classification == T) {
  arr2 <- as.factor(arr2)
  fitControl <- trainControl(method = "none", classProbs = F)
}

if (classification == F) {
  fitControl <- trainControl(method = "none", classProbs = F)
  arr2 <- as.numeric(arr2)
}



arr1 <- as.data.frame(arr1)


rdaFit <- train(x= arr1, y= arr2, 
                method = "avNNet", 
                trControl = fitControl
)


data <- cbind(arr1, arr2)
moodel <- cforest(arr2 ~ ., data =  data)
varimp(moodel)
# Imp <- varImp(rdaFit, scale = FALSE)
# print(Imp$importance)


library('partykit')
varimp(rdaFit)

vi_scores <- vi(rdaFit, method = "firm", sort = FALSE)
# vi(rdaFit, method = "firm", sort = FALSE)$Importance
vi_scores


b <- Imp$importance


a <- (count(arr2)$freq / sum(count(arr2)$freq))

b * a
as.vector(rowSums(b * a))




library('iml')
library("rpart")
# We train a tree on the Boston dataset:
data("Boston", package = "MASS")
tree <- rpart(medv ~ ., data = Boston)
y <- Boston$medv
X <- Boston[-which(names(Boston) == "medv")]
mod <- Predictor$new(tree, data = X, y = y)
# Compute feature importances as the perform
imp <- FeatureImp$new(mod, loss = "mae")




library("vip")
#
# A projection pursuit regression example
#
# Load the sample data
data(mtcars)
# Fit a projection pursuit regression model
model <- ppr(mpg ~ ., data = mtcars, nterms = 1)
# Construct variable importance plot
vip(model, method = "firm")
# Better yet, store the variable importance scores and then plot
vi_scores <- vi(model, method = "firm")



library('caret')
library(reticulate)
library(plyr)
library('vip')
pd <- import("pandas")
pickle_data <- pd$read_pickle('E:/OneDrive/PhD/GitHub/Official_Feature_Importance/src/bp_feature_importance/datasets/decimal_system.pickle')

arr1 <- as.data.frame(pickle_data[1])
colnames(arr1) <- c('1', '2', '3')
arr2 <- array(unlist(pickle_data[2]), dim= c(200))
#colnames(arr2) <- c("Y")


classification <- T

if (classification == T) {
  arr2 <- as.factor(arr2)
  fitControl <- trainControl(method = "none", classProbs = F)
}

if (classification == F) {
  fitControl <- trainControl(method = "none", classProbs = F)
  arr2 <- as.numeric(arr2)
}



arr1 <- as.data.frame(arr1)
dataset <- cbind(arr1, arr2)


library('FSinR')
url <- "http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
adult <- read.csv(url, strip.white = TRUE, header = TRUE)
adult <- adult[,c(4,9,10,15)]

# Generate the evaluation function with Cramer
gr_evaluator <- gainRatio()
# Evaluate the features (parameters: dataset, target variable and features)
gr_evaluator(adult,'income',c('race','sex','education'))

data('iris')
filter_evaluator <- filterEvaluator('ReliefFeatureSetMeasure')
# Evaluates features directly (parameters: dataset, target variable and features)
filter_evaluator(iris,'Species',c('Sepal.Length','Sepal.Width','Petal.Length','Petal.Width'))

data('iris')
filterMeasure <- giniIndex(iris, "Species", c("Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"))















library('caret')
library(reticulate)
library(plyr)
library('vip')
pd <- import("pandas")
pickle_data <- pd$read_pickle('E:/OneDrive/PhD/GitHub/Official_Feature_Importance/src/bp_feature_importance/datasets/decimal_system.pickle')

X <- as.data.frame(pickle_data[1])

Y <- array(unlist(pickle_data[2]), dim= c(200))

library(FSinR)
names <- c('binaryConsistency', 'chiSquared', 'cramer', 'gainRatio', 'giniIndex', 'IEConsistency', 'IEPConsistency', 'mutualInformation',  'roughsetConsistency', 'ReliefFeatureSetMeasure', 'symmetricalUncertain')

for (name in names) {
  print(name)
  evaluator <- filterEvaluator('determinationCoefficient')
  dataset <- as.data.frame(cbind(X, Y))
  colnames(dataset) <- 1:ncol(dataset)
  
  # index <- 1:ncol(dataset)
  # dataset[ , index] <- lapply(dataset[ , index], as.factor)
  
  results <- c()
  Ycolnames <- colnames(dataset)[-(1:ncol(X))]
  Xcolnames <- colnames(dataset)[1:ncol(X)]
  for (i in Xcolnames){
    results[as.numeric(i)] <- evaluator(dataset, Ycolnames, i)
  }
  results
}  


evaluator(dataset, Ycolnames, c('1', '2'))





# devtools::install_github("giuseppec/featureImportance")
# 
# library(mlr)
# library(mlbench)
# library(ggplot2)
# library(gridExtra)
# library(featureImportance)
# set.seed(2018)
# 
# # Get boston housing data and look at the data
# data(BostonHousing, package = "mlbench")
# str(BostonHousing)
# # Create regression task for mlr
# boston.task = makeRegrTask(data = BostonHousing, target = "medv")
# 
# # Specify the machine learning algorithm with the mlr package
# lrn = makeLearner("regr.glm")
# 
# # Create indices for train and test data
# n = getTaskSize(boston.task)
# train.ind = 1:n
# test.ind = 1:n
# 
# # Create test data using test indices
# test = getTaskData(boston.task, subset = test.ind)
# 
# # Fit model on train data using train indices
# mod = train(lrn, boston.task, subset = train.ind)
# 
# # Measure feature importance on test data
# imp = featureImportance(mod, data = test)
# summary(imp)




