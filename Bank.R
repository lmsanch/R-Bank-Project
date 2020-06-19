## ----load libraries, echo = FALSE, message = FALSE, warning = FALSE, eval = TRUE----

library(tidyverse)
library(caret)
library(lubridate)

library(gridExtra)
library(knitr)
library(kableExtra)

library(ggplot2)
library(caret)
library(e1071)

library(rpart)
library(rpart.plot)
library(rattle)

library(randomForest)
library(caTools)
library(descr)

library(pROC)
library(Matrix)
library(xgboost)


## ----Install libraries & download files, echo = TRUE, message = FALSE, warning = FALSE, eval = TRUE----
if(!require(tidyverse)) 
  install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) 
  install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(lubridate)) 
  install.packages("lubridate", repos = "http://cran.us.r-project.org")
if(!require(gridExtra)) 
  install.packages("gridExtra", repos = "http://cran.us.r-project.org")
if(!require(knitr)) 
  install.packages("knitr", repos = "http://cran.us.r-project.org")
if(!require(kableExtra)) 
  install.packages("kableExtra", repos = "http://cran.us.r-project.org")
if(!require(ggplot2)) 
  install.packages("ggplot2", repos = "http://cran.us.r-project.org")
if(!require(caret)) 
  install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(e1071)) 
  install.packages("e1071", repos = "http://cran.us.r-project.org")
if(!require(rpart)) 
  install.packages("rpart", repos = "http://cran.us.r-project.org")
if(!require(rpart.plot)) 
  install.packages("rpart.plot", repos = "http://cran.us.r-project.org")
if(!require(rattle)) 
  install.packages("rattle", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) 
  install.packages("randomForest", repos = "http://cran.us.r-project.org")
if(!require(caTools)) 
  install.packages("caTools", repos = "http://cran.us.r-project.org")
if(!require(descr)) 
  install.packages("descr", repos = "http://cran.us.r-project.org")
if(!require(pROC)) 
  install.packages("pROC", repos = "http://cran.us.r-project.org")
if(!require(Matrix)) 
  install.packages("Matrix", repos = "http://cran.us.r-project.org")
if(!require(xgboost)) 
  install.packages("xgboot", repos = "http://cran.us.r-project.org")

# here I download the file
wd = getwd()
dl <- tempfile(tmpdir = wd)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip"
download.file(url, dl)
unzip(dl, junkpaths = TRUE)
data <- read.table("bank-additional-full.csv", 
                   header = T, 
                   sep = ";")
unlink(dl)


## ----edx_summary_first_8, echo=FALSE-------------------------------------
as_tibble(data[1:5, 1:8])%>%head()%>%
  kable()%>%kable_styling()%>%
  column_spec(3)%>%
  row_spec(0,bold=T)


## ----edx_summary_9-15, echo=FALSE----------------------------------------
as_tibble(data[1:5, 9:15])%>%head()%>%
  kable()%>%kable_styling()%>%
  column_spec(3)%>%
  row_spec(0,bold=T)


## ----edx_summary_16_21, echo=FALSE---------------------------------------
as_tibble(data[1:5, 16:21])%>%head()%>%
  kable()%>%kable_styling()%>%
  column_spec(3)%>%
  row_spec(0,bold=T)


## ----type, echo=TRUE-----------------------------------------------------
sapply(data, class)
sum(is.na(data))


## ----set seed, echo=FALSE------------------------------------------------
set.seed(1, sample.kind="Rounding")


## ----partitioning data, echo=TRUE----------------------------------------
test_index <- createDataPartition(y = data$y, times = 1, p = 0.25, list = FALSE)
train <- data[-test_index,]
test <- data[test_index,]


## ----decision tree, echo=TRUE--------------------------------------------
# setting up the decision tree
train_dt<-rpart(y ~ ., train , method = 'class')

# predictions and probbailities
predictions <- predict(train_dt, test , type = "class")
probabilities <- predict(train_dt, test , type = "prob")

# metrics
CrossTable(test$y, predictions,
           prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
           dnn = c('actual', 'predicted'))
# AUCROC
data_roc <-roc(test$y, probabilities[,2])
print(data_roc)


## ----metrics DT, echo=TRUE-----------------------------------------------

# par(mfrow=c(2,1))
fancyRpartPlot(train_dt , digits=2, caption="Decision Tree")
plot(data_roc, main="AUC-ROC for Decision Tree",col="blue")


## ----random forest, echo=TRUE--------------------------------------------
# setting up the Random Forest

rf_classifier = randomForest(y ~ ., train, 
                             ntree=200, 
                             mtry=3, 
                             importance=TRUE)

# predict
rf_pred <- predict( rf_classifier, 
                    test, 
                    type = "class")
rf_prob <- predict( rf_classifier, 
                    test, 
                    type = "prob")

# Cross table validation for CART
CrossTable(test$y, 
           rf_pred,
           prop.chisq = FALSE, 
           prop.c = FALSE, 
           prop.r = FALSE,
           dnn = c('actual default', 'predicted default'))
# ROC
rf_roc <-roc(test$y, 
             rf_prob[,2])
print(rf_roc)


## ----metrics RF, echo=TRUE-----------------------------------------------
# par(mfrow=c(2,1))
varImpPlot(rf_classifier)
plot(rf_roc, 
     main="AUC-ROC for Random Forest",
     percent=TRUE, 
     col="red")


## ----settting up sparse matrix, echo=TRUE--------------------------------
sparse.matrix.train= sparse.model.matrix(y~.-1, data = train) 
# creates sparse matrix from the train set
output.vector.train <- as.numeric(as.factor(train$y))-1 
# transforms y into "1" and "0", the values OK with xgboost
sparse.matrix.test=  sparse.model.matrix(y~.-1, data = test) 
# creates sparse matrix from the test set
output.vector.test <- as.numeric(as.factor(test$y))-1 
# transforms y into "1" and "0", the values OK with xgboost


## ----settting up XGBoost, echo=TRUE--------------------------------------
bst= xgboost(data = sparse.matrix.train, # train sparse matrix 
             label = output.vector.train, # output vector to be predicted 
             eval.metric = 'auc', # model maximizes auc
             objective = "binary:logistic", # clasification
             max.depth = 3,
             nround = 250,
             verbose = 0) #dont print out results of trainning to PDF

importance <- xgb.importance(feature_names = colnames(sparse.matrix.train), 
                             model = bst)


## ----XGBoost metrics, echo=TRUE------------------------------------------
probabilities.xgb <- predict(bst, sparse.matrix.test)
predictions.xgb <- as.numeric(probabilities.xgb > 0.5) # transform prediction to labels

# metrics
CrossTable(output.vector.test, 
           predictions.xgb,
           prop.chisq = FALSE, 
           prop.c = FALSE, 
           prop.r = FALSE,
           dnn = c('actual', 'predicted'))
roc.xgb <-roc(output.vector.test, probabilities.xgb)
print(roc.xgb)


## ----XGBoost ROC, echo=TRUE----------------------------------------------
# ROC
# par(mfrow=c(2,1))
xgb.plot.importance(importance_matrix = importance, top_n = 15)
plot(roc.xgb, main="AUC-ROC for XGBoost", percent=TRUE, col="blue")


## ----Plotting the data1, echo=FALSE--------------------------------------
par(mfrow=c(4,2))
for(i in 1:length(data))
{barplot(prop.table(table(data[,i])) , 
         xlab=names(data[i]), ylab= "Frequency (%)")}

