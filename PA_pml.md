# Human Activity Recognition


Using devices such as Jawbone Up, Nike FuelBand, and Fitbit, it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. 

We analyze the data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants to predict the manner in which they did the exercise. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website [here] (http://groupware.les.inf.puc-rio.br/har) (see the section on the Weight Lifting Exercise Dataset). 

The goal of the project is to predict the manner in which they did the exercise.


## Data preprocessing   
We load the following libraries that are used throughout the code.

```r
library(caret)
library(randomForest)
library(foreach)
library(doParallel)
```

The csv files containing the training and test data are downloaded into the working directory.

```r
if (!file.exists("training.csv")) {
    download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", 
                  destfile = "training.csv")
}
if (!file.exists("testing.csv")) {
    download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", 
                  destfile = "testing.csv")
}
```

We first load the training data

```r
dtTrain <- read.csv("training.csv", na.strings= c("NA",""," ","#DIV/0!"))
```
We remove the first 7 columns that contain the user names, timestamps and windows.

```
## [1] "X"                    "user_name"            "raw_timestamp_part_1"
## [4] "raw_timestamp_part_2" "cvtd_timestamp"       "new_window"          
## [7] "num_window"
```
There are several columns filled with too many NA's and thus not useful in prediction. We also remove these columns entirely.

```r
dtTrain <- dtTrain[8:dim(dtTrain)[2]]
selection <- colSums(is.na(dtTrain))==0
dtTrain2 <- subset(dtTrain, select=selection)
```
There are 53 variables with no NA's. They will be used in prediction.

We check for near zero covariates and remove them, if any, since they do not contribute well to prediction.

```r
selection2 <- nearZeroVar(dtTrain2, saveMetrics=TRUE)$nzv
```
Since there are 0 near zero covariates, we will use all the variables for training and prediction.

The preprocessed training data set is then split into training and cross-validation sets by 3:1.

```r
inTrain <- createDataPartition(dtTrain2$classe, p=3/4, list=FALSE)
training <- dtTrain2[inTrain, ]
validation <- dtTrain2[-inTrain, ]
```


## Model building  
We fit a random forests predictor relating the factor variable `classe` to the remaining variables. To speed up the learning process, we make use of parallel processing.


```r
registerDoParallel()
modFit1 <- foreach(ntree=rep(150, 6), .combine=combine, .multicombine=TRUE, .packages='randomForest') %dopar% {
    randomForest(training[-53], training$classe, ntree=ntree)
}
```
As parallel processing is used, the output `modFit1` does not contain out of sample error. We will estimate the error using cross validation data.


## Cross-validation  
The model is used to classify the cross-validation data. The accuracy of the model is found by comparing the prediction and the actual classification using the confusion matrix function.

```r
pred1 <- predict(modFit1, newdata=validation)
confusionMatrix(pred1, validation$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1395    1    0    0    0
##          B    0  945    5    0    0
##          C    0    3  850    7    2
##          D    0    0    0  797    0
##          E    0    0    0    0  899
## 
## Overall Statistics
##                                         
##                Accuracy : 0.996         
##                  95% CI : (0.994, 0.998)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.995         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             1.000    0.996    0.994    0.991    0.998
## Specificity             1.000    0.999    0.997    1.000    1.000
## Pos Pred Value          0.999    0.995    0.986    1.000    1.000
## Neg Pred Value          1.000    0.999    0.999    0.998    1.000
## Prevalence              0.284    0.194    0.174    0.164    0.184
## Detection Rate          0.284    0.193    0.173    0.163    0.183
## Detection Prevalence    0.285    0.194    0.176    0.163    0.183
## Balanced Accuracy       1.000    0.997    0.996    0.996    0.999
```

The random forests model yields a 99.6% prediction accuracy using the cross validation data. Thus, we estimate the out of sample error to be 0.4%.


## Prediction 
We now classify the testing data using the prediction model we have constructed earlier.

```r
dtTest <- read.csv("testing.csv")
dtTest <- dtTest[8:dim(dtTest)[2]]
testing <- subset(dtTest, select=selection)
pred <- predict(modFit1, newdata=testing)
```
We write the prediction `pred` for the 20 test cases into text files for submission.

```r
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}
pml_write_files(pred)
```
