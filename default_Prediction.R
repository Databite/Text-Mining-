#### install necessary packages #####package installation
if ("pacman" %in% rownames(installed.packages()) == FALSE) {
  install.packages("pacman")
} # Check if you have universal installer package, install if not
pacman::p_load("caret","vegan", "gpairs", "cluster", "corrplot", "MASS", "DAAG", "car", "factoextra", "NbClust", "readxl", "tidyverse", "lubridate", "HDclassif", "DataExplorer", "dummies", "fastAdaboost", "DMwR", "randomForest", "e1071", "ANN2", "pagedown", "klaR","mime","htmltools","httpuv","later","promises","klaR"," nnet", "haven", "tm", "textmineR", "SnowballC", "textstem","magrittr","wordcloud","sentimentr","portfolio","GuardianR","lda","LDAvis","SnowballC","pbapply","quanteda","Metrics","DMwR", "MLmetrics","kknn","nnet","rpart") #Check, and if needed install the necessary packages

df <- read.csv("D:\\kiva_data.csv", stringsAsFactors=FALSE)

########## Renaming and type setting of attributes within the dataset ###########

df = df %>%
  rename(story = en)
str(df)
df$id = 1:nrow(df)
df$status = as.factor(df$status)
df$sector = as.factor(df$sector)
df$country = as.factor(df$country)
df$gender = as.factor(df$gender)
df$nonpayment = as.factor(df$nonpayment)

########## Remove HTML Tags ###################
df = df %>% 
  mutate(story = gsub("<.*?>", "", story))

########## Retain distinct entries ################

dim(df)
df = distinct(df, story, .keep_all = TRUE)
dim(df)

############## Clustering the applicants based on all attributes but the story #######

cl <- df
cl$X <- NULL
cl$loan_nonpayment <- NULL
cl$id <- NULL
str(cl)
cl$status <- as.numeric(cl$status)
cl$sector <- as.numeric(cl$sector)
cl$story <- NULL
cl$country <- as.numeric(cl$country)
cl$gender <- as.numeric(cl$gender)
cl$countrygender <- as.numeric(as.factor(as.character(cl$countrygender)))
cl$CountrySector <- as.numeric(as.factor(as.character(cl$CountrySector)))
cl$loan_amount <- as.numeric (cl$loan_amount)
cl$nonpayment <- as.numeric (cl$nonpayment)

## Elbow plot
wss <- (nrow(cl)-1)*sum(apply(cl,2,var))

for (i in 1:30) wss[i] <- sum(kmeans(cl, 
                                     centers=i)$withinss)
plot(1:30, wss, type="b", xlab="Number of Clusters",
     ylab="Within groups sum of squares")

## Choose 4 clusters 

clust <- kmeans(cl, centers = 4, n = 4)

head(clust$cluster)
df <- cbind(df, cluster_name = clust$cluster)

#################### Sentiment Detection ######################

Sentiment <- sentiment_by(df$story)
summary(Sentiment$ave_sentiment)

qplot(Sentiment$ave_sentiment, geom = "histogram", binwidth = 0.1, main = "Review Sentiment Histogram")


rm(Sentiment, cl, clust)
########## Removing text that is not neccessary for topic identification or further analysis #############
dim(df)
df = distinct(df, story, .keep_all = TRUE)
dim(df)

idx = grep("Translated from Spanish", df$story)
df$story[idx[1]]
df$story = gsub("Translated from Spanish.*Kiva volunteer", " ", df$story, ignore.case = TRUE)
grep("Translated from Spanish", df$story)

grep("Mifex offers", df$story)
df$story = gsub("Mifex offers.*www.mifex.org", " ", df$story, ignore.case = TRUE)
grep("Mifex offers", df$story)

df$story = gsub("About KADET:.*within those communities", " ", df$story, ignore.case = TRUE)
df$story = gsub("Disclaimer: Due to recent events in.*making their loan.", " ", df$story, ignore.case = TRUE)

########## Topic Modelling ######################################
################### LDA #######################

####### Create Document Term Matrix ###########
# setting n-gram window to be exactly 1 -- unigrams are necessary for our analysis ########## Eliminating stopwords from our text based on english, french, spanish and SMART distionaries ##### converting all uppercase letters to lowercase ##### removing both puctuation and numbers from our text #### using Textstem library, we are lemmatizing words to their root forms instead of stemming #########

dtm_1_1 <- CreateDtm(doc_vec = df$story, # character vector of documents
                 doc_names = df$id, # document names
                 ngram_window = c(1,1), # minimum and maximum n-gram length
                 stopword_vec = c(tm::stopwords("english"), # stopwords from tm
                                  tm::stopwords("french"), # stopwords from tm
                                  tm::stopwords("spanish"), # stopwords from tm
                                  tm::stopwords("SMART")), # this is the default value
                 lower = TRUE, # lowercase - this is the default value
                 remove_punctuation = TRUE, # punctuation - this is the default
                 remove_numbers = TRUE, # numbers - this is the default
                 verbose = FALSE, # Turn off status bar for this demo,
                 stem_lemma_function = function(x) textstem::lemmatize_words(x),
                 cpus = 20) # default is all available cpus on the system


# Filter rare words
dim(dtm_1_1)
dtm_1_1 <- dtm_1_1[ , colSums(dtm_1_1 > 0) > 50 ]
dim(dtm_1_1)

max_num = nrow(dtm_1_1) * 0.3
max_num
dtm_1_1 <- dtm_1_1[ , colSums(dtm_1_1 > 0) <= max_num ]
dim(dtm_1_1)

# Capture how long each document is
df$doc_lengths = rowSums(dtm_1_1)

### identify frequent terms and highest 50 IDFs ###
tf_mat <- TermDocFreq(dtm = dtm_1_1)
head(tf_mat[ order(tf_mat$term_freq, decreasing = TRUE) , ], 50)
# write.csv(tf_mat, "term_frequency.csv")
head(tf_mat[ order(tf_mat$idf, decreasing = TRUE) , ], 50)

# look at the most frequent bigrams
tf_bigrams <- tf_mat[ stringr::str_detect(tf_mat$term, "_") , ]
head(tf_bigrams[ order(tf_bigrams$term_freq, decreasing = TRUE) , ], 50)
write.csv(head(tf_bigrams[ order(tf_bigrams$term_freq, decreasing = TRUE) , ], 50), "1.1_kiva_bi_grams.csv")

# look at the most frequent trigrams
tf_trigrams <- tf_mat[ stringr::str_detect(tf_mat$term, "_") , ]
head(tf_trigrams[ order(tf_trigrams$term_freq, decreasing = TRUE) , ], 50)
write.csv(head(tf_trigrams[ order(tf_trigrams$term_freq, decreasing = TRUE) , ], 50), "1.1_kiva_tri_grams.csv")


##### fitting an LDA Model #####

twenty_model <- FitLdaModel(dtm = dtm_1_1, 
                     k = 20, 
                     iterations = 500, # recommend a larger value, 500 or more
                     burnin = 180,
                     optimize_alpha = TRUE,
                     calc_likelihood = TRUE,
                     calc_coherence = TRUE,
                     calc_r2 = TRUE,
                     alpha = 0.1, # this is the default value
                     beta = 0.05, # this is the default value
                     cpus = 21) # Note, this is for a big machine  

plot(twenty_model$log_likelihood, type = "l")


twenty_model$top_terms <- GetTopTerms(phi = twenty_model$phi, M = 8)
head(twenty_model$top_terms) 

# probabilistic coherence, a measure of topic quality
# this measure can be used with any topic model, not just probabilistic ones
twenty_model$coherence <- CalcProbCoherence(phi = twenty_model$phi, dtm = dtm_1_1, M = 5)

# Get the prevalence of each topic
# You can make this discrete by applying a threshold, say 0.05, for
# topics in/out of docuemnts. 
twenty_model$prevalence <- colSums(twenty_model$theta) / sum(twenty_model$theta) * 100


# textmineR has a naive topic labeling tool based on probable bigrams
twenty_model$labels <- LabelTopics(assignments = twenty_model$theta > 0.20, 
                            dtm = dtm_1_1,
                            M = 2)

head(twenty_model$labels)


# put them together, with coherence into a summary table
twenty_model$summary <- data.frame(topic = rownames(twenty_model$phi),
                            label = twenty_model$labels,
                            coherence = round(twenty_model$coherence, 2),
                            prevalence = round(twenty_model$prevalence,1),
                            top_terms = apply(twenty_model$top_terms, 2, function(x){
                              paste(x, collapse = ", ")
                            }),
                            stringsAsFactors = FALSE)

twenty_model$summary %>%
  arrange(-prevalence)


write.csv(twenty_model$top_terms, file = "1.1_20_Topics_kiva.csv")

#### Adding topics to data frame #####

df = cbind(df, round(twenty_model$theta, digits=2))
head(df, n=10)


##### Model Building for Classification ######
# Helper function to print the confusion matrix and other performance metrics of the models.
printPerformance = function(pred, actual, positive="Yes") {
  print(caret::confusionMatrix(data=pred, reference=actual, positive=positive, dnn=c("Predicted", "Actual")))
}

#### Data Type casting ####

str(df$status)
str(df)
df$id <- NULL
df <- as.data.frame(unclass(df))#converting all characters variables to factors
str(df)
df$story <- NULL

# Splitting the Data into train and test

set.seed(123) # Set the seed to make it reproducible

train.index <- createDataPartition(df$status, p = .8, list = FALSE)
train <- df[ train.index,]
test  <- df[-train.index,]

actual <- test$status
formula <- status ~ .
positive <- "yes"

# Random Forest with great results
set.seed(123)
model_forest <- randomForest(status~ ., data=train, 
                             importance=TRUE,proximity=TRUE,
                             cutoff = c(0.5, 0.5),type="classification", na.action=na.exclude)
print(model_forest)   
plot(model_forest)
importance(model_forest)

varImpPlot(model_forest)#Shows the decrease in accuracy with the removal of a variables so non Annual removal the accuray can go down by 30% for example

predicted <- predict(model_forest, test, type="class") 

print(sprintf("Accuracy of Random Forest for k = 20:    %.3f", Accuracy(y_true=actual, y_pred=predicted)))
print(sprintf("Precision of Random Forest for k = 20:   %.3f", Precision(y_true=actual, y_pred=predicted)))
print(sprintf("Recall of Random Forest for k = 20:      %.3f", Recall(y_true=actual, y_pred=predicted)))
print(sprintf("F1 Score of Random Forest for k = 20:    %.3f", F1_Score(predicted, actual)))
print(sprintf("Sensitivity of Random Forest for k = 20: %.3f", Sensitivity(y_true=actual, y_pred=predicted)))
print(sprintf("Specificity of Random Forest for k = 20: %.3f", Specificity(y_true=predicted, y_pred=actual)))

# Naive Bayes...Accuracy of 0.73 not very good
set.seed(123)

nb_fit = naiveBayes(status ~ ., data=train)
nb_fit

NB_pred = predict(nb_fit, test, type="class") 
print(sprintf("Accuracy of NB for k = 20:    %.3f", Accuracy(y_true=actual, y_pred=NB_pred)))
print(sprintf("Precision of NB for k = 20:   %.3f", Precision(y_true=actual, y_pred=NB_pred)))
print(sprintf("F1 Score of NB for k = 20:    %.3f", F1_Score(NB_pred, actual)))
print(sprintf("Sensitivityof NB for k = 20: %.3f", Sensitivity(y_true=actual, y_pred=NB_pred)))
print(sprintf("Specificity of NB for k = 20: %.3f", Specificity(y_true=NB_pred, y_pred=actual)))

# Decision Tree
set.seed(123)

tree <- rpart(status ~ ., method="class", data=train)
tree_pred = predict(tree, test, type="class") 
print(sprintf("Accuracy of DT for k = 20:    %.3f", Accuracy(y_true=actual, y_pred=tree_pred)))
print(sprintf("Precision of DT for k = 20:   %.3f", Precision(y_true=actual, y_pred=tree_pred)))
print(sprintf("F1 Score of DT for k = 20:    %.3f", F1_Score(tree_pred, actual)))
print(sprintf("Sensitivity of DT for k = 20: %.3f", Sensitivity(y_true=actual, y_pred=tree_pred)))
print(sprintf("Specificity of DT for k = 20: %.3f", Specificity(y_true=tree_pred, y_pred=actual)))

#SVM...this is the second best
set.seed(123)
svm_fit <- svm(status ~ ., data = train)
summary(svm_fit)

svm_pred <- predict(svm_fit, test) 

print(sprintf("Accuracy of SVM for k = 20:    %.3f", Accuracy(y_true=actual, y_pred=svm_pred)))
print(sprintf("Precision of SVM for k = 20:   %.3f", Precision(y_true=actual, y_pred=svm_pred)))
print(sprintf("F1 Score of SVM for k = 20:    %.3f", F1_Score(svm_pred, actual)))
print(sprintf("Sensitivity of SVM for k = 20: %.3f", Sensitivity(y_true=actual, y_pred=svm_pred)))
print(sprintf("Specificity of SVM for k = 20: %.3f", Specificity(y_true=svm_pred, y_pred=actual)))


#kknn....
set.seed(123)

kknn_fit = train.kknn(status ~ ., train, kmax=7)
summary(kknn_fit)

kknn_pred = predict(kknn_fit, test)

print(sprintf("Accuracy of KNN for k = 20:    %.3f", Accuracy(y_true=actual, y_pred=kknn_pred)))
print(sprintf("Precision of KNN for k = 20:   %.3f", Precision(y_true=actual, y_pred=kknn_pred)))
print(sprintf("F1 Score of KNN for k = 20:    %.3f", F1_Score(kknn_pred, actual)))
print(sprintf("Sensitivity of KNN for k = 20: %.3f", Sensitivity(y_true=actual, y_pred=kknn_pred)))
print(sprintf("Specificity of KNN for k = 20: %.3f", Specificity(y_true=kknn_pred, y_pred=actual)))

# #NN...91%
# install.packages("nnet")
# library(nnet)
# set.seed(123)
# 
# train_nn <- (train)
# test_nn <- test
# str(train_nn)
# 
# train_nn$status <- as.numeric(train_nn$status)
# train_nn$sector <- as.numeric(train_nn$sector)
# train_nn$country <- as.numeric(train_nn$country)
# train_nn$gender <- as.numeric(train_nn$gender)
# train_nn$countrygender <- as.numeric(as.factor(as.character(train_nn$countrygender)))
# train_nn$CountrySector <- as.numeric(as.factor(as.character(train_nn$CountrySector)))
# train_nn$nonpayment <- as.numeric (train_nn$nonpayment)
# 
# test_nn$status <- as.numeric(test_nn$status)
# test_nn$sector <- as.numeric(test_nn$sector)
# test_nn$country <- as.numeric(test_nn$country)
# test_nn$gender <- as.numeric(test_nn$gender)
# test_nn$countrygender <- as.numeric(as.factor(as.character(test_nn$countrygender)))
# test_nn$CountrySector <- as.numeric(as.factor(as.character(test_nn$CountrySector)))
# test_nn$nonpayment <- as.numeric (test_nn$nonpayment)
# str(train)
# ##start Neural Network analysis
# 
# my.grid <- expand.grid( .size = c(1,2,4),.decay = c(0.25,1,2)) # Tuning grid for Neural Net
# 
# model_NN <- train(status~ ., data = train_nn, method = "nnet", tuneGrid = my.grid, trace = TRUE, na.action =na.omit)
# 
# plot(model_NN) #Visualize the relationship between the number of layers, decay and accuracy
# NN_prediction<-predict(model_NN, test_nn) #Predict classification 
# print(sprintf("Accuracy of NN for k = 20:    %.3f", Accuracy(y_true=test_nn$status, y_pred=NN_prediction)))
# print(sprintf("Precision of NN for k = 20:   %.3f", Precision(y_true=test_nn$status, y_pred=NN_prediction)))
# print(sprintf("F1 Score of NN for k = 20:    %.3f", F1_Score(NN_prediction, test_nn$status)))
# print(sprintf("Sensitivity of NN for k = 20: %.3f", Sensitivity(y_true=test_nn$status, y_pred=NN_prediction)))
# print(sprintf("Specificity of NN for k = 20: %.3f", Specificity(y_true=NN_prediction, y_pred=test_nn$status)))

### removing topics from this part
topics <- df[,14:33]
df[,14:33] <- NULL


##### Topic Modelling with Sensitive Study ###############

#### With k = 10 ##########
##### fitting an LDA Model #####

ten_model <- FitLdaModel(dtm = dtm_1_1, 
                     k = 10, 
                     iterations = 500, # recommend a larger value, 500 or more
                     burnin = 180,
                     optimize_alpha = TRUE,
                     calc_likelihood = TRUE,
                     calc_coherence = TRUE,
                     calc_r2 = TRUE,
                     alpha = 0.1, # this is the default value
                     beta = 0.05, # this is the default value
                     cpus = 21) # Note, this is for a big machine  

plot(ten_model$log_likelihood, type = "l")


ten_model$top_terms <- GetTopTerms(phi = ten_model$phi, M = 8)
head(ten_model$top_terms) 

# probabilistic coherence, a measure of topic quality
# this measure can be used with any topic model, not just probabilistic ones
ten_model$coherence <- CalcProbCoherence(phi = ten_model$phi, dtm = dtm_1_1, M = 5)

# Get the prevalence of each topic
# You can make this discrete by applying a threshold, say 0.05, for
# topics in/out of docuemnts. 
ten_model$prevalence <- colSums(ten_model$theta) / sum(ten_model$theta) * 100


# textmineR has a naive topic labeling tool based on probable bigrams
ten_model$labels <- LabelTopics(assignments = ten_model$theta > 0.20, 
                            dtm = dtm_1_1,
                            M = 2)

head(ten_model$labels)


# put them together, with coherence into a summary table
ten_model$summary <- data.frame(topic = rownames(ten_model$phi),
                            label = ten_model$labels,
                            coherence = round(ten_model$coherence, 2),
                            prevalence = round(ten_model$prevalence,1),
                            top_terms = apply(ten_model$top_terms, 2, function(x){
                              paste(x, collapse = ", ")
                            }),
                            stringsAsFactors = FALSE)

ten_model$summary %>%
  arrange(-prevalence)


write.csv(ten_model$top_terms, file = "1.1_10_Topics_kiva.csv")

#### Adding topics to data frame #####

df = cbind(df, round(ten_model$theta, digits=2))
head(df, n=10)

####################################################################
########## Model Building for k = 10 ###################

# Splitting the Data into train and test

set.seed(123) # Set the seed to make it reproducible

train.index <- createDataPartition(df$status, p = .8, list = FALSE)
train <- df[ train.index,]
test  <- df[-train.index,]

actual <- test$status
formula <- status ~ .
positive <- "yes"

# Random Forest with great results
set.seed(123)
model_forest <- randomForest(status~ ., data=train, 
                             importance=TRUE,proximity=TRUE,
                             cutoff = c(0.5, 0.5),type="classification", na.action=na.exclude)
print(model_forest)   
plot(model_forest)
importance(model_forest)

varImpPlot(model_forest)#Shows the decrease in accuracy with the removal of a variables so non Annual removal the accuray can go down by 30% for example

predicted <- predict(model_forest, test, type="class") 

print(sprintf("Accuracy of Random Forest for k = 10:    %.3f", Accuracy(y_true=actual, y_pred=predicted)))
print(sprintf("Precision of Random Forest for k = 10:   %.3f", Precision(y_true=actual, y_pred=predicted)))
print(sprintf("Recall of Random Forest for k = 10:      %.3f", Recall(y_true=actual, y_pred=predicted)))
print(sprintf("F1 Score of Random Forest for k = 10:    %.3f", F1_Score(predicted, actual)))
print(sprintf("Sensitivity of Random Forest for k = 10: %.3f", Sensitivity(y_true=actual, y_pred=predicted)))
print(sprintf("Specificity of Random Forest for k = 10: %.3f", Specificity(y_true=predicted, y_pred=actual)))

# Naive Bayes...Accuracy of 0.73 not very good
set.seed(123)

nb_fit = naiveBayes(status ~ ., data=train)
nb_fit

NB_pred = predict(nb_fit, test, type="class") 
print(sprintf("Accuracy of NB for k = 10:    %.3f", Accuracy(y_true=actual, y_pred=NB_pred)))
print(sprintf("Precision of NB for k = 10:   %.3f", Precision(y_true=actual, y_pred=NB_pred)))
print(sprintf("F1 Score of NB for k = 10:    %.3f", F1_Score(NB_pred, actual)))
print(sprintf("Sensitivityof NB for k = 10: %.3f", Sensitivity(y_true=actual, y_pred=NB_pred)))
print(sprintf("Specificity of NB for k = 10: %.3f", Specificity(y_true=NB_pred, y_pred=actual)))

# Decision Tree

set.seed(123)

tree <- rpart(status ~ ., method="class", data=train)
tree_pred = predict(tree, test, type="class") 
print(sprintf("Accuracy of DT for k = 10:    %.3f", Accuracy(y_true=actual, y_pred=tree_pred)))
print(sprintf("Precision of DT for k = 10:   %.3f", Precision(y_true=actual, y_pred=tree_pred)))
print(sprintf("F1 Score of DT for k = 10:    %.3f", F1_Score(tree_pred, actual)))
print(sprintf("Sensitivity of DT for k = 10: %.3f", Sensitivity(y_true=actual, y_pred=tree_pred)))
print(sprintf("Specificity of DT for k = 10: %.3f", Specificity(y_true=tree_pred, y_pred=actual)))

#SVM...this is the second best
set.seed(123)
svm_fit <- svm(status ~ ., data = train)
summary(svm_fit)

svm_pred <- predict(svm_fit, test) 

print(sprintf("Accuracy of SVM for k = 10:    %.3f", Accuracy(y_true=actual, y_pred=svm_pred)))
print(sprintf("Precision of SVM for k = 10:   %.3f", Precision(y_true=actual, y_pred=svm_pred)))
print(sprintf("F1 Score of SVM for k = 10:    %.3f", F1_Score(svm_pred, actual)))
print(sprintf("Sensitivity of SVM for k = 10: %.3f", Sensitivity(y_true=actual, y_pred=svm_pred)))
print(sprintf("Specificity of SVM for k = 10: %.3f", Specificity(y_true=svm_pred, y_pred=actual)))


#kknn....89%
set.seed(123)

kknn_fit = train.kknn(status ~ ., train, kmax=7)
summary(kknn_fit)

kknn_pred = predict(kknn_fit, test)

print(sprintf("Accuracy of KNN for k = 10:    %.3f", Accuracy(y_true=actual, y_pred=kknn_pred)))
print(sprintf("Precision of KNN for k = 10:   %.3f", Precision(y_true=actual, y_pred=kknn_pred)))
print(sprintf("F1 Score of KNN for k = 10:    %.3f", F1_Score(kknn_pred, actual)))
print(sprintf("Sensitivity of KNN for k = 10: %.3f", Sensitivity(y_true=actual, y_pred=kknn_pred)))
print(sprintf("Specificity of KNN for k = 10: %.3f", Specificity(y_true=kknn_pred, y_pred=actual)))

# #NN...91%
# install.packages("nnet")
# library(nnet)
# set.seed(123)
# 
# train_nn <- (train)
# test_nn <- test
# str(train_nn)
# 
# train_nn$status <- as.numeric(train_nn$status)
# train_nn$sector <- as.numeric(train_nn$sector)
# train_nn$country <- as.numeric(train_nn$country)
# train_nn$gender <- as.numeric(train_nn$gender)
# train_nn$countrygender <- as.numeric(as.factor(as.character(train_nn$countrygender)))
# train_nn$CountrySector <- as.numeric(as.factor(as.character(train_nn$CountrySector)))
# train_nn$nonpayment <- as.numeric (train_nn$nonpayment)
# 
# test_nn$status <- as.numeric(test_nn$status)
# test_nn$sector <- as.numeric(test_nn$sector)
# test_nn$country <- as.numeric(test_nn$country)
# test_nn$gender <- as.numeric(test_nn$gender)
# test_nn$countrygender <- as.numeric(as.factor(as.character(test_nn$countrygender)))
# test_nn$CountrySector <- as.numeric(as.factor(as.character(test_nn$CountrySector)))
# test_nn$nonpayment <- as.numeric (test_nn$nonpayment)
# str(train)
# ##start Neural Network analysis
# 
# my.grid <- expand.grid( .size = c(1,2,4),.decay = c(0.25,1,2)) # Tuning grid for Neural Net
# 
# model_NN <- train(status~ ., data = train_nn, method = "nnet", tuneGrid = my.grid, trace = TRUE, na.action =na.omit)
# 
# plot(model_NN) #Visualize the relationship between the number of layers, decay and accuracy
# NN_prediction<-predict(model_NN, test_nn) #Predict classification 
# print(sprintf("Accuracy of NN for k = 10:    %.3f", Accuracy(y_true=test_nn$status, y_pred=NN_prediction)))
# print(sprintf("Precision of NN for k = 10:   %.3f", Precision(y_true=test_nn$status, y_pred=NN_prediction)))
# print(sprintf("F1 Score of NN for k = 10:    %.3f", F1_Score(NN_prediction, test_nn$status)))
# print(sprintf("Sensitivity of NN for k = 10: %.3f", Sensitivity(y_true=test_nn$status, y_pred=NN_prediction)))
# print(sprintf("Specificity of NN for k = 10: %.3f", Specificity(y_true=NN_prediction, y_pred=test_nn$status)))

### removing topics from this part
topics <- cbind(df[,14:23])
### removing topics from this part
topics <- df[,14:23]
df[,14:23] <- NULL

##### Topic Modelling with Sensitive Study ###############

#### With k = 15 ##########

##### fitting an LDA Model #####

fifteen_model <- FitLdaModel(dtm = dtm_1_1, 
                         k = 15, 
                         iterations = 500, # recommend a larger value, 500 or more
                         burnin = 180,
                         optimize_alpha = TRUE,
                         calc_likelihood = TRUE,
                         calc_coherence = TRUE,
                         calc_r2 = TRUE,
                         alpha = 0.1, # this is the default value
                         beta = 0.05, # this is the default value
                         cpus = 21) # Note, this is for a big machine  

plot(fifteen_model$log_likelihood, type = "l")


fifteen_model$top_terms <- GetTopTerms(phi = fifteen_model$phi, M = 8)
head(fifteen_model$top_terms) 

# probabilistic coherence, a measure of topic quality
# this measure can be used with any topic model, not just probabilistic ones
fifteen_model$coherence <- CalcProbCoherence(phi = fifteen_model$phi, dtm = dtm_1_1, M = 5)

# Get the prevalence of each topic
# You can make this discrete by applying a threshold, say 0.05, for
# topics in/out of docuemnts. 
fifteen_model$prevalence <- colSums(fifteen_model$theta) / sum(fifteen_model$theta) * 100


# textmineR has a naive topic labeling tool based on probable bigrams
fifteen_model$labels <- LabelTopics(assignments = fifteen_model$theta > 0.20, 
                                dtm = dtm_1_1,
                                M = 2)

head(fifteen_model$labels)


# put them together, with coherence into a summary table
fifteen_model$summary <- data.frame(topic = rownames(fifteen_model$phi),
                                label = fifteen_model$labels,
                                coherence = round(fifteen_model$coherence, 2),
                                prevalence = round(fifteen_model$prevalence,1),
                                top_terms = apply(fifteen_model$top_terms, 2, function(x){
                                  paste(x, collapse = ", ")
                                }),
                                stringsAsFactors = FALSE)

fifteen_model$summary %>%
  arrange(-prevalence)


write.csv(fifteen_model$top_terms, file = "1.1_15_Topics_kiva.csv")

#### Adding topics to data frame #####

df = cbind(df, round(fifteen_model$theta, digits=2))
head(df, n=10)

####################################################################
########## Model Building for k = 15 ###################

# Splitting the Data into train and test

set.seed(123) # Set the seed to make it reproducible

train.index <- createDataPartition(df$status, p = .8, list = FALSE)
train <- df[ train.index,]
test  <- df[-train.index,]

actual <- test$status
formula <- status ~ .
positive <- "yes"

# Random Forest with great results
set.seed(123)
model_forest <- randomForest(status~ ., data=train, 
                             importance=TRUE,proximity=TRUE,
                             cutoff = c(0.5, 0.5),type="classification", na.action=na.exclude)
print(model_forest)   
plot(model_forest)
importance(model_forest)

varImpPlot(model_forest)#Shows the decrease in accuracy with the removal of a variables so non Annual removal the accuray can go down by 30% for example

predicted <- predict(model_forest, test, type="class") 

print(sprintf("Accuracy of Random Forest for k = 15:    %.3f", Accuracy(y_true=actual, y_pred=predicted)))
print(sprintf("AUC of Random Forest for k = 15:     %.3f", AUC(y_pred=predicted, y_true=actual)))
print(sprintf("Precision of Random Forest for k = 15:   %.3f", Precision(y_true=actual, y_pred=predicted)))
print(sprintf("Recall of Random Forest for k = 15:      %.3f", Recall(y_true=actual, y_pred=predicted)))
print(sprintf("F1 Score of Random Forest for k = 15:    %.3f", F1_Score(predicted, actual)))
print(sprintf("Sensitivity of Random Forest for k = 15: %.3f", Sensitivity(y_true=actual, y_pred=predicted)))
print(sprintf("Specificity of Random Forest for k = 15: %.3f", Specificity(y_true=predicted, y_pred=actual)))

# Naive Bayes...Accuracy of 0.73 not very good
set.seed(123)

nb_fit = naiveBayes(status ~ ., data=train)
nb_fit

NB_pred = predict(nb_fit, test, type="class") 
print(sprintf("Accuracy of NB for k = 15:    %.3f", Accuracy(y_true=actual, y_pred=NB_pred)))
print(sprintf("Precision of NB for k = 15:   %.3f", Precision(y_true=actual, y_pred=NB_pred)))
print(sprintf("F1 Score of NB for k = 15:    %.3f", F1_Score(NB_pred, actual)))
print(sprintf("Sensitivityof NB for k = 15: %.3f", Sensitivity(y_true=actual, y_pred=NB_pred)))
print(sprintf("Specificity of NB for k = 15: %.3f", Specificity(y_true=NB_pred, y_pred=actual)))

# Decision Tree

set.seed(123)

tree <- rpart(status ~ ., method="class", data=train)
tree_pred = predict(tree, test, type="class") 
print(sprintf("Accuracy of DT for k = 15:    %.3f", Accuracy(y_true=actual, y_pred=tree_pred)))
print(sprintf("Precision of DT for k = 15:   %.3f", Precision(y_true=actual, y_pred=tree_pred)))
print(sprintf("F1 Score of DT for k = 15:    %.3f", F1_Score(tree_pred, actual)))
print(sprintf("Sensitivity of DT for k = 15: %.3f", Sensitivity(y_true=actual, y_pred=tree_pred)))
print(sprintf("Specificity of DT for k = 15: %.3f", Specificity(y_true=tree_pred, y_pred=actual)))

#SVM...this is the second best

svm_fit <- svm(status ~ ., data = train)
summary(svm_fit)

svm_pred <- predict(svm_fit, test) 

print(sprintf("Accuracy of SVM for k = 15:    %.3f", Accuracy(y_true=actual, y_pred=svm_pred)))
print(sprintf("Precision of SVM for k = 15:   %.3f", Precision(y_true=actual, y_pred=svm_pred)))
print(sprintf("F1 Score of SVM for k = 15:    %.3f", F1_Score(svm_pred, actual)))
print(sprintf("Sensitivity of SVM for k = 15: %.3f", Sensitivity(y_true=actual, y_pred=svm_pred)))
print(sprintf("Specificity of SVM for k = 15: %.3f", Specificity(y_true=svm_pred, y_pred=actual)))


#kknn....89%
set.seed(123)
kknn_fit = train.kknn(status ~ ., train, kmax=7)
summary(kknn_fit)

kknn_pred = predict(kknn_fit, test)

print(sprintf("Accuracy of KNN for k = 15:    %.3f", Accuracy(y_true=actual, y_pred=kknn_pred)))
print(sprintf("Precision of KNN for k = 15:   %.3f", Precision(y_true=actual, y_pred=kknn_pred)))
print(sprintf("F1 Score of KNN for k = 15:    %.3f", F1_Score(kknn_pred, actual)))
print(sprintf("Sensitivity of KNN for k = 15: %.3f", Sensitivity(y_true=actual, y_pred=kknn_pred)))
print(sprintf("Specificity of KNN for k = 15: %.3f", Specificity(y_true=kknn_pred, y_pred=actual)))

# #NN...91%
# install.packages("nnet")
# library(nnet)
# set.seed(123)
# 
# train_nn <- (train)
# test_nn <- test
# str(train_nn)
# 
# train_nn$status <- as.numeric(train_nn$status)
# train_nn$sector <- as.numeric(train_nn$sector)
# train_nn$country <- as.numeric(train_nn$country)
# train_nn$gender <- as.numeric(train_nn$gender)
# train_nn$countrygender <- as.numeric(as.factor(as.character(train_nn$countrygender)))
# train_nn$CountrySector <- as.numeric(as.factor(as.character(train_nn$CountrySector)))
# train_nn$nonpayment <- as.numeric (train_nn$nonpayment)
# 
# test_nn$status <- as.numeric(test_nn$status)
# test_nn$sector <- as.numeric(test_nn$sector)
# test_nn$country <- as.numeric(test_nn$country)
# test_nn$gender <- as.numeric(test_nn$gender)
# test_nn$countrygender <- as.numeric(as.factor(as.character(test_nn$countrygender)))
# test_nn$CountrySector <- as.numeric(as.factor(as.character(test_nn$CountrySector)))
# test_nn$nonpayment <- as.numeric (test_nn$nonpayment)
# str(train)
# ##start Neural Network analysis
# 
# my.grid <- expand.grid( .size = c(1,2,4),.decay = c(0.25,1,2)) # Tuning grid for Neural Net
# 
# model_NN <- train(status~ ., data = train_nn, method = "nnet", tuneGrid = my.grid, trace = TRUE, na.action =na.omit)
# 
# plot(model_NN) #Visualize the relationship between the number of layers, decay and accuracy
# NN_prediction<-predict(model_NN, test_nn) #Predict classification 
# print(sprintf("Accuracy of NN for k = 15:    %.3f", Accuracy(y_true=test_nn$status, y_pred=NN_prediction)))
# print(sprintf("Precision of NN for k = 15:   %.3f", Precision(y_true=test_nn$status, y_pred=NN_prediction)))
# print(sprintf("F1 Score of NN for k = 15:    %.3f", F1_Score(NN_prediction, test_nn$status)))
# print(sprintf("Sensitivity of NN for k = 15: %.3f", Sensitivity(y_true=test_nn$status, y_pred=NN_prediction)))
# print(sprintf("Specificity of NN for k = 15: %.3f", Specificity(y_true=NN_prediction, y_pred=test_nn$status)))

### removing topics from this part
topics <- cbind(df[,14:28])
### removing topics from this part
topics <- df[,14:28]
df[,14:28] <- NULL

##### Topic Modelling with Sensitive Study ###############

#### With k = 25 ##########

##### fitting an LDA Model #####

twentyfive_model <- FitLdaModel(dtm = dtm_1_1, 
                             k = 25, 
                             iterations = 500, # recommend a larger value, 500 or more
                             burnin = 180,
                             optimize_alpha = TRUE,
                             calc_likelihood = TRUE,
                             calc_coherence = TRUE,
                             calc_r2 = TRUE,
                             alpha = 0.1, # this is the default value
                             beta = 0.05, # this is the default value
                             cpus = 21) # Note, this is for a big machine  

plot(twentyfive_model$log_likelihood, type = "l")


twentyfive_model$top_terms <- GetTopTerms(phi = twentyfive_model$phi, M = 8)
head(twentyfive_model$top_terms) 

# probabilistic coherence, a measure of topic quality
# this measure can be used with any topic model, not just probabilistic ones
twentyfive_model$coherence <- CalcProbCoherence(phi = twentyfive_model$phi, dtm = dtm_1_1, M = 5)

# Get the prevalence of each topic
# You can make this discrete by applying a threshold, say 0.05, for
# topics in/out of docuemnts. 
twentyfive_model$prevalence <- colSums(twentyfive_model$theta) / sum(twentyfive_model$theta) * 100


# textmineR has a naive topic labeling tool based on probable bigrams
twentyfive_model$labels <- LabelTopics(assignments = twentyfive_model$theta > 0.20, 
                                    dtm = dtm_1_1,
                                    M = 2)

head(twentyfive_model$labels)


# put them together, with coherence into a summary table
twentyfive_model$summary <- data.frame(topic = rownames(twentyfive_model$phi),
                                    label = twentyfive_model$labels,
                                    coherence = round(twentyfive_model$coherence, 2),
                                    prevalence = round(twentyfive_model$prevalence,1),
                                    top_terms = apply(twentyfive_model$top_terms, 2, function(x){
                                      paste(x, collapse = ", ")
                                    }),
                                    stringsAsFactors = FALSE)

twentyfive_model$summary %>%
  arrange(-prevalence)


write.csv(twentyfive_model$top_terms, file = "1.1_25_Topics_kiva.csv")

#### Adding topics to data frame #####

df = cbind(df, round(twentyfive_model$theta, digits=2))
head(df, n=10)

####################################################################
########## Model Building for k = 15 ###################

# Splitting the Data into train and test

set.seed(123) # Set the seed to make it reproducible

train.index <- createDataPartition(df$status, p = .8, list = FALSE)
train <- df[ train.index,]
test  <- df[-train.index,]

actual <- test$status
formula <- status ~ .
positive <- "yes"

# Random Forest with great results
set.seed(123)
model_forest <- randomForest(status~ ., data=train, 
                             importance=TRUE,proximity=TRUE,
                             cutoff = c(0.5, 0.5),type="classification", na.action=na.exclude)
print(model_forest)   
plot(model_forest)
importance(model_forest)

varImpPlot(model_forest)#Shows the decrease in accuracy with the removal of a variables so non Annual removal the accuray can go down by 30% for example

predicted <- predict(model_forest, test, type="class") 

print(sprintf("Accuracy of Random Forest for k = 25:    %.3f", Accuracy(y_true=actual, y_pred=predicted)))
print(sprintf("AUC of Random Forest for k = 25:     %.3f", AUC(y_pred=predicted, y_true=actual)))
print(sprintf("Precision of Random Forest for k = 25:   %.3f", Precision(y_true=actual, y_pred=predicted)))
print(sprintf("Recall of Random Forest for k = 25:      %.3f", Recall(y_true=actual, y_pred=predicted)))
print(sprintf("F1 Score of Random Forest for k = 25:    %.3f", F1_Score(predicted, actual)))
print(sprintf("Sensitivity of Random Forest for k = 25: %.3f", Sensitivity(y_true=actual, y_pred=predicted)))
print(sprintf("Specificity of Random Forest for k = 25: %.3f", Specificity(y_true=predicted, y_pred=actual)))

# Naive Bayes...Accuracy of 0.73 not very good
set.seed(123)

nb_fit = naiveBayes(status ~ ., data=train)
nb_fit

NB_pred = predict(nb_fit, test, type="class") 
print(sprintf("Accuracy of NB for k = 25:    %.3f", Accuracy(y_true=actual, y_pred=NB_pred)))
print(sprintf("Precision of NB for k = 25:   %.3f", Precision(y_true=actual, y_pred=NB_pred)))
print(sprintf("F1 Score of NB for k = 25:    %.3f", F1_Score(NB_pred, actual)))
print(sprintf("Sensitivityof NB for k = 25: %.3f", Sensitivity(y_true=actual, y_pred=NB_pred)))
print(sprintf("Specificity of NB for k = 25: %.3f", Specificity(y_true=NB_pred, y_pred=actual)))
# Decision Tree

set.seed(123)

tree <- rpart(status ~ ., method="class", data=train)
tree_pred = predict(tree, test, type="class") 
print(sprintf("Accuracy of DT for k = 25:    %.3f", Accuracy(y_true=actual, y_pred=tree_pred)))
print(sprintf("Precision of DT for k = 25:   %.3f", Precision(y_true=actual, y_pred=tree_pred)))
print(sprintf("F1 Score of DT for k = 25:    %.3f", F1_Score(tree_pred, actual)))
print(sprintf("Sensitivity of DT for k = 25: %.3f", Sensitivity(y_true=actual, y_pred=tree_pred)))
print(sprintf("Specificity of DT for k = 25: %.3f", Specificity(y_true=tree_pred, y_pred=actual)))

#SVM...this is the second best

svm_fit <- svm(status ~ ., data = train)
summary(svm_fit)

svm_pred <- predict(svm_fit, test) 

print(sprintf("Accuracy of SVM for k = 25:    %.3f", Accuracy(y_true=actual, y_pred=svm_pred)))
print(sprintf("Precision of SVM for k = 25:   %.3f", Precision(y_true=actual, y_pred=svm_pred)))
print(sprintf("F1 Score of SVM for k = 25:    %.3f", F1_Score(svm_pred, actual)))
print(sprintf("Sensitivity of SVM for k = 25: %.3f", Sensitivity(y_true=actual, y_pred=svm_pred)))
print(sprintf("Specificity of SVM for k = 25: %.3f", Specificity(y_true=svm_pred, y_pred=actual)))


#kknn....89%
set.seed(123)
kknn_fit = train.kknn(status ~ ., train, kmax=7)
summary(kknn_fit)

kknn_pred = predict(kknn_fit, test)

print(sprintf("Accuracy of KNN for k = 25:    %.3f", Accuracy(y_true=actual, y_pred=kknn_pred)))
print(sprintf("Precision of KNN for k = 25:   %.3f", Precision(y_true=actual, y_pred=kknn_pred)))
print(sprintf("F1 Score of KNN for k = 25:    %.3f", F1_Score(kknn_pred, actual)))
print(sprintf("Sensitivity of KNN for k = 25: %.3f", Sensitivity(y_true=actual, y_pred=kknn_pred)))
print(sprintf("Specificity of KNN for k = 25: %.3f", Specificity(y_true=kknn_pred, y_pred=actual)))

# #NN...91%
# install.packages("nnet")
# library(nnet)
# set.seed(123)
# 
# train_nn <- (train)
# test_nn <- test
# str(train_nn)
# 
# train_nn$status <- as.numeric(train_nn$status)
# train_nn$sector <- as.numeric(train_nn$sector)
# train_nn$country <- as.numeric(train_nn$country)
# train_nn$gender <- as.numeric(train_nn$gender)
# train_nn$countrygender <- as.numeric(as.factor(as.character(train_nn$countrygender)))
# train_nn$CountrySector <- as.numeric(as.factor(as.character(train_nn$CountrySector)))
# train_nn$nonpayment <- as.numeric (train_nn$nonpayment)
# 
# test_nn$status <- as.numeric(test_nn$status)
# test_nn$sector <- as.numeric(test_nn$sector)
# test_nn$country <- as.numeric(test_nn$country)
# test_nn$gender <- as.numeric(test_nn$gender)
# test_nn$countrygender <- as.numeric(as.factor(as.character(test_nn$countrygender)))
# test_nn$CountrySector <- as.numeric(as.factor(as.character(test_nn$CountrySector)))
# test_nn$nonpayment <- as.numeric (test_nn$nonpayment)
# str(train)
# ##start Neural Network analysis
# 
# my.grid <- expand.grid( .size = c(1,2,4),.decay = c(0.25,1,2)) # Tuning grid for Neural Net
# 
# model_NN <- train(status~ ., data = train_nn, method = "nnet", tuneGrid = my.grid, trace = TRUE, na.action =na.omit)
# 
# plot(model_NN) #Visualize the relationship between the number of layers, decay and accuracy
# NN_prediction<-predict(model_NN, test_nn) #Predict classification 
# print(sprintf("Accuracy of NN for k = 25:    %.3f", Accuracy(y_true=test_nn$status, y_pred=NN_prediction)))
# print(sprintf("Precision of NN for k = 25:   %.3f", Precision(y_true=test_nn$status, y_pred=NN_prediction)))
# print(sprintf("F1 Score of NN for k = 25:    %.3f", F1_Score(NN_prediction, test_nn$status)))
# print(sprintf("Sensitivity of NN for k = 25: %.3f", Sensitivity(y_true=test_nn$status, y_pred=NN_prediction)))
# print(sprintf("Specificity of NN for k = 25: %.3f", Specificity(y_true=NN_prediction, y_pred=test_nn$status)))


### removing topics from this part
topics <- cbind(df[,14:38])
### removing topics from this part
df[,14:38] <- NULL

##### Topic Modelling with Sensitive Study ###############

#### With k = 30 ##########

##### fitting an LDA Model #####

thirty_model <- FitLdaModel(dtm = dtm_1_1, 
                                k = 30, 
                                iterations = 500, # recommend a larger value, 500 or more
                                burnin = 180,
                                optimize_alpha = TRUE,
                                calc_likelihood = TRUE,
                                calc_coherence = TRUE,
                                calc_r2 = TRUE,
                                alpha = 0.1, # this is the default value
                                beta = 0.05, # this is the default value
                                cpus = 21) # Note, this is for a big machine  

plot(thirty_model$log_likelihood, type = "l")


thirty_model$top_terms <- GetTopTerms(phi = thirty_model$phi, M = 8)
head(thirty_model$top_terms) 

# probabilistic coherence, a measure of topic quality
# this measure can be used with any topic model, not just probabilistic ones
thirty_model$coherence <- CalcProbCoherence(phi = thirty_model$phi, dtm = dtm_1_1, M = 5)

# Get the prevalence of each topic
# You can make this discrete by applying a threshold, say 0.05, for
# topics in/out of docuemnts. 
thirty_model$prevalence <- colSums(thirty_model$theta) / sum(thirty_model$theta) * 100


# textmineR has a naive topic labeling tool based on probable bigrams
thirty_model$labels <- LabelTopics(assignments = thirty_model$theta > 0.20, 
                                       dtm = dtm_1_1,
                                       M = 2)

head(thirty_model$labels)


# put them together, with coherence into a summary table
thirty_model$summary <- data.frame(topic = rownames(thirty_model$phi),
                                       label = thirty_model$labels,
                                       coherence = round(thirty_model$coherence, 2),
                                       prevalence = round(thirty_model$prevalence,1),
                                       top_terms = apply(thirty_model$top_terms, 2, function(x){
                                         paste(x, collapse = ", ")
                                       }),
                                       stringsAsFactors = FALSE)

thirty_model$summary %>%
  arrange(-prevalence)


write.csv(thirty_model$top_terms, file = "1.1_30_Topics_kiva.csv")

#### Adding topics to data frame #####

df = cbind(df, round(thirty_model$theta, digits=2))
head(df, n=10)

####################################################################
########## Model Building for k = 15 ###################

# Splitting the Data into train and test

set.seed(123) # Set the seed to make it reproducible

train.index <- createDataPartition(df$status, p = .8, list = FALSE)
train <- df[ train.index,]
test  <- df[-train.index,]

actual <- test$status
formula <- status ~ .
positive <- "yes"

# Random Forest with great results
set.seed(123)
model_forest <- randomForest(status~ ., data=train, 
                             importance=TRUE,proximity=TRUE,
                             cutoff = c(0.5, 0.5),type="classification", na.action=na.exclude)
print(model_forest)   
plot(model_forest)
importance(model_forest)

varImpPlot(model_forest)#Shows the decrease in accuracy with the removal of a variables so non Annual removal the accuray can go down by 30% for example

predicted <- predict(model_forest, test, type="class") 

print(sprintf("Accuracy of Random Forest for k = 30:    %.3f", Accuracy(y_true=actual, y_pred=predicted)))
print(sprintf("AUC of Random Forest for k = 30:     %.3f", AUC(y_pred=predicted, y_true=actual)))
print(sprintf("Precision of Random Forest for k = 30:   %.3f", Precision(y_true=actual, y_pred=predicted)))
print(sprintf("Recall of Random Forest for k = 30:      %.3f", Recall(y_true=actual, y_pred=predicted)))
print(sprintf("F1 Score of Random Forest for k = 30:    %.3f", F1_Score(predicted, actual)))
print(sprintf("Sensitivity of Random Forest for k = 30: %.3f", Sensitivity(y_true=actual, y_pred=predicted)))
print(sprintf("Specificity of Random Forest for k = 30: %.3f", Specificity(y_true=predicted, y_pred=actual)))

# Naive Bayes...Accuracy of 0.73 not very good
set.seed(123)

nb_fit = naiveBayes(status ~ ., data=train)
nb_fit

NB_pred = predict(nb_fit, test, type="class") 
print(sprintf("Accuracy of NB for k = 30:    %.3f", Accuracy(y_true=actual, y_pred=NB_pred)))
print(sprintf("Precision of NB for k = 30:   %.3f", Precision(y_true=actual, y_pred=NB_pred)))
print(sprintf("F1 Score of NB for k = 30:    %.3f", F1_Score(NB_pred, actual)))
print(sprintf("Sensitivityof NB for k = 30: %.3f", Sensitivity(y_true=actual, y_pred=NB_pred)))
print(sprintf("Specificity of NB for k = 30: %.3f", Specificity(y_true=NB_pred, y_pred=actual)))


# Decision Tree
set.seed(123)

tree <- rpart(status ~ ., method="class", data=train)
tree_pred = predict(tree, test, type="class") 
print(sprintf("Accuracy of DT for k = 30:    %.3f", Accuracy(y_true=actual, y_pred=tree_pred)))
print(sprintf("Precision of DT for k = 30:   %.3f", Precision(y_true=actual, y_pred=tree_pred)))
print(sprintf("F1 Score of DT for k = 30:    %.3f", F1_Score(tree_pred, actual)))
print(sprintf("Sensitivity of DT for k = 30: %.3f", Sensitivity(y_true=actual, y_pred=tree_pred)))
print(sprintf("Specificity of DT for k = 30: %.3f", Specificity(y_true=tree_pred, y_pred=actual)))

#SVM...this is the second best
set.seed(123)
svm_fit <- svm(status ~ ., data = train)
summary(svm_fit)

svm_pred <- predict(svm_fit, test) 

print(sprintf("Accuracy of SVM for k = 30:    %.3f", Accuracy(y_true=actual, y_pred=svm_pred)))
print(sprintf("Precision of SVM for k = 30:   %.3f", Precision(y_true=actual, y_pred=svm_pred)))
print(sprintf("F1 Score of SVM for k = 30:    %.3f", F1_Score(svm_pred, actual)))
print(sprintf("Sensitivity of SVM for k = 30: %.3f", Sensitivity(y_true=actual, y_pred=svm_pred)))
print(sprintf("Specificity of SVM for k = 30: %.3f", Specificity(y_true=svm_pred, y_pred=actual)))


#kknn....89%
set.seed(123)
kknn_fit = train.kknn(status ~ ., train, kmax=7)
summary(kknn_fit)

kknn_pred = predict(kknn_fit, test)

print(sprintf("Accuracy of KNN for k = 30:    %.3f", Accuracy(y_true=actual, y_pred=kknn_pred)))
print(sprintf("Precision of KNN for k = 30:   %.3f", Precision(y_true=actual, y_pred=kknn_pred)))
print(sprintf("F1 Score of KNN for k = 30:    %.3f", F1_Score(kknn_pred, actual)))
print(sprintf("Sensitivity of KNN for k = 30: %.3f", Sensitivity(y_true=actual, y_pred=kknn_pred)))
print(sprintf("Specificity of KNN for k = 30: %.3f", Specificity(y_true=kknn_pred, y_pred=actual)))

# #NN...91%
# install.packages("nnet")
# library(nnet)
# set.seed(123)
# 
# train_nn <- (train)
# test_nn <- test
# str(train_nn)
# 
# train_nn$status <- as.numeric(train_nn$status)
# train_nn$sector <- as.numeric(train_nn$sector)
# train_nn$country <- as.numeric(train_nn$country)
# train_nn$gender <- as.numeric(train_nn$gender)
# train_nn$countrygender <- as.numeric(as.factor(as.character(train_nn$countrygender)))
# train_nn$CountrySector <- as.numeric(as.factor(as.character(train_nn$CountrySector)))
# train_nn$nonpayment <- as.numeric (train_nn$nonpayment)
# 
# test_nn$status <- as.numeric(test_nn$status)
# test_nn$sector <- as.numeric(test_nn$sector)
# test_nn$country <- as.numeric(test_nn$country)
# test_nn$gender <- as.numeric(test_nn$gender)
# test_nn$countrygender <- as.numeric(as.factor(as.character(test_nn$countrygender)))
# test_nn$CountrySector <- as.numeric(as.factor(as.character(test_nn$CountrySector)))
# test_nn$nonpayment <- as.numeric (test_nn$nonpayment)
# str(train)
# ##start Neural Network analysis
# 
# my.grid <- expand.grid( .size = c(1,2,4),.decay = c(0.25,1,2)) # Tuning grid for Neural Net
# 
# model_NN <- train(status~ ., data = train_nn, method = "nnet", tuneGrid = my.grid, trace = TRUE, na.action =na.omit)
# 
# plot(model_NN) #Visualize the relationship between the number of layers, decay and accuracy
# NN_prediction<-predict(model_NN, test_nn) #Predict classification 
# print(sprintf("Accuracy of NN for k = 30:    %.3f", Accuracy(y_true=test_nn$status, y_pred=NN_prediction)))
# print(sprintf("Precision of NN for k = 30:   %.3f", Precision(y_true=test_nn$status, y_pred=NN_prediction)))
# print(sprintf("F1 Score of NN for k = 30:    %.3f", F1_Score(NN_prediction, test_nn$status)))
# print(sprintf("Sensitivity of NN for k = 30: %.3f", Sensitivity(y_true=test_nn$status, y_pred=NN_prediction)))
# print(sprintf("Specificity of NN for k = 30: %.3f", Specificity(y_true=NN_prediction, y_pred=test_nn$status)))

### removing topics from this part
topics <- cbind(df[,14:43])
### removing topics from this part
df[,14:43] <- NULL


##########################################################################
######### Topic Modelling ################################################
################### LDA #######################

df <- read.csv("D:\\kiva_data.csv", stringsAsFactors=FALSE)

########## Renaming and type setting of attributes within the dataset ###########

df = df %>%
  rename(story = en)
str(df)
df$id = 1:nrow(df)
df$status = as.factor(df$status)
df$sector = as.factor(df$sector)
df$country = as.factor(df$country)
df$gender = as.factor(df$gender)
df$nonpayment = as.factor(df$nonpayment)

########## Remove HTML Tags ###################
df = df %>% 
  mutate(story = gsub("<.*?>", "", story))

########## Retain distinct entries ################

dim(df)
df = distinct(df, story, .keep_all = TRUE)
dim(df)

############## Clustering the applicants based on all attributes but the story #######

cl <- df
cl$X <- NULL
cl$loan_nonpayment <- NULL
cl$id <- NULL
str(cl)
cl$status <- as.numeric(cl$status)
cl$sector <- as.numeric(cl$sector)
cl$story <- NULL
cl$country <- as.numeric(cl$country)
cl$gender <- as.numeric(cl$gender)
cl$countrygender <- as.numeric(as.factor(as.character(cl$countrygender)))
cl$CountrySector <- as.numeric(as.factor(as.character(cl$CountrySector)))
cl$loan_amount <- as.numeric (cl$loan_amount)
cl$nonpayment <- as.numeric (cl$nonpayment)

## Elbow plot
wss <- (nrow(cl)-1)*sum(apply(cl,2,var))

for (i in 1:30) wss[i] <- sum(kmeans(cl, 
                                     centers=i)$withinss)
plot(1:30, wss, type="b", xlab="Number of Clusters",
     ylab="Within groups sum of squares")

## Choose 4 clusters 

clust <- kmeans(cl, centers = 4, n = 4)

head(clust$cluster)
df <- cbind(df, cluster_name = clust$cluster)

#################### Sentiment Detection ######################

Sentiment <- sentiment_by(df$story)
summary(Sentiment$ave_sentiment)

qplot(Sentiment$ave_sentiment, geom = "histogram", binwidth = 0.1, main = "Review Sentiment Histogram")
head(sentiment)
df$ave_sentiment <- Sentiment$ave_sentiment
df$sd_sentiment <- Sentiment$sd
df$word_count <- Sentiment$word_count

rm(Sentiment, cl, clust)
########## Removing text that is not neccessary for topic identification or further analysis #############
dim(df)
df = distinct(df, story, .keep_all = TRUE)
dim(df)

idx = grep("Translated from Spanish", df$story)
df$story[idx[1]]
df$story = gsub("Translated from Spanish.*Kiva volunteer", " ", df$story, ignore.case = TRUE)
grep("Translated from Spanish", df$story)

grep("Mifex offers", df$story)
df$story = gsub("Mifex offers.*www.mifex.org", " ", df$story, ignore.case = TRUE)
grep("Mifex offers", df$story)

df$story = gsub("About KADET:.*within those communities", " ", df$story, ignore.case = TRUE)
df$story = gsub("Disclaimer: Due to recent events in.*making their loan.", " ", df$story, ignore.case = TRUE)



####### Create Document Term Matrix ###########
# setting n-gram window to be exactly 1 -- unigrams are necessary for our analysis ########## Eliminating stopwords from our text based on english, french, spanish and SMART distionaries ##### converting all uppercase letters to lowercase ##### removing both puctuation and numbers from our text #### using Textstem library, we are lemmatizing words to their root forms instead of stemming #########

dtm_2_2 <- CreateDtm(doc_vec = df$story, # character vector of documents
                     doc_names = df$id, # document names
                     ngram_window = c(2,2), # minimum and maximum n-gram length
                     stopword_vec = c(tm::stopwords("english"), # stopwords from tm
                                      tm::stopwords("french"), # stopwords from tm
                                      tm::stopwords("spanish"), # stopwords from tm
                                      tm::stopwords("SMART")), # this is the default value
                     lower = TRUE, # lowercase - this is the default value
                     remove_punctuation = TRUE, # punctuation - this is the default
                     remove_numbers = TRUE, # numbers - this is the default
                     verbose = FALSE, # Turn off status bar for this demo,
                     stem_lemma_function = function(x) textstem::lemmatize_words(x),
                     cpus = 20) # default is all available cpus on the system


# Filter rare words
dim(dtm_2_2)
dtm_2_2 <- dtm_1_3[ , colSums(dtm_2_2 > 0) > 50 ]
dim(dtm_2_2)

max_num = nrow(dtm_2_2) * 0.3
max_num
dtm_2_2 <- dtm_2_2[ , colSums(dtm_2_2 > 0) <= max_num ]
dim(dtm_2_2)

# Capture how long each document is
df$doc_lengths = rowSums(dtm_2_2)

### identify frequent terms and highest 50 IDFs ###
tf_mat <- TermDocFreq(dtm = dtm_2_2)
head(tf_mat[ order(tf_mat$term_freq, decreasing = TRUE) , ], 50)
# write.csv(tf_mat, "term_frequency.csv")
head(tf_mat[ order(tf_mat$idf, decreasing = TRUE) , ], 50)

# look at the most frequent bigrams
tf_bigrams <- tf_mat[ stringr::str_detect(tf_mat$term, "_") , ]
head(tf_bigrams[ order(tf_bigrams$term_freq, decreasing = TRUE) , ], 50)
write.csv(head(tf_bigrams[ order(tf_bigrams$term_freq, decreasing = TRUE) , ], 50), "2.2_kiva_bi_grams.csv")

# look at the most frequent trigrams
tf_trigrams <- tf_mat[ stringr::str_detect(tf_mat$term, "_") , ]
head(tf_trigrams[ order(tf_trigrams$term_freq, decreasing = TRUE) , ], 50)
write.csv(head(tf_trigrams[ order(tf_trigrams$term_freq, decreasing = TRUE) , ], 50), "2.2_kiva_tri_grams.csv")


##### fitting an LDA Model #####

twenty_model <- FitLdaModel(dtm = dtm_2_2, 
                            k = 20, 
                            iterations = 500, # recommend a larger value, 500 or more
                            burnin = 180,
                            optimize_alpha = TRUE,
                            calc_likelihood = TRUE,
                            calc_coherence = TRUE,
                            calc_r2 = TRUE,
                            alpha = 0.1, # this is the default value
                            beta = 0.05, # this is the default value
                            cpus = 21) # Note, this is for a big machine  

plot(twenty_model$log_likelihood, type = "l")


twenty_model$top_terms <- GetTopTerms(phi = twenty_model$phi, M = 8)
head(twenty_model$top_terms) 

# probabilistic coherence, a measure of topic quality
# this measure can be used with any topic model, not just probabilistic ones
twenty_model$coherence <- CalcProbCoherence(phi = twenty_model$phi, dtm = dtm_2_2, M = 5)

# Get the prevalence of each topic
# You can make this discrete by applying a threshold, say 0.05, for
# topics in/out of docuemnts. 
twenty_model$prevalence <- colSums(twenty_model$theta) / sum(twenty_model$theta) * 100


# textmineR has a naive topic labeling tool based on probable bigrams
twenty_model$labels <- LabelTopics(assignments = twenty_model$theta > 0.20, 
                                   dtm = dtm_2_2,
                                   M = 2)

head(twenty_model$labels)


# put them together, with coherence into a summary table
twenty_model$summary <- data.frame(topic = rownames(twenty_model$phi),
                                   label = twenty_model$labels,
                                   coherence = round(twenty_model$coherence, 2),
                                   prevalence = round(twenty_model$prevalence,1),
                                   top_terms = apply(twenty_model$top_terms, 2, function(x){
                                     paste(x, collapse = ", ")
                                   }),
                                   stringsAsFactors = FALSE)

twenty_model$summary %>%
  arrange(-prevalence)

write.csv(twenty_model$top_terms, file = "2.2_20_Topics_kiva.csv")

#### Adding topics to data frame #####

df = cbind(df, round(twenty_model$theta, digits=2))
head(df, n=10)


##### Model Building for Classification ######
# Helper function to print the confusion matrix and other performance metrics of the models.
printPerformance = function(pred, actual, positive="Yes") {
  print(caret::confusionMatrix(data=pred, reference=actual, positive=positive, dnn=c("Predicted", "Actual")))
}

#### Data Type casting ####

str(df$status)
str(df)
df$id <- NULL
df <- as.data.frame(unclass(df))#converting all characters variables to factors
str(df)
df$story <- NULL

# Splitting the Data into train and test

set.seed(123) # Set the seed to make it reproducible

train.index <- createDataPartition(df$status, p = .8, list = FALSE)
train <- df[ train.index,]
test  <- df[-train.index,]

actual <- test$status
formula <- status ~ .
positive <- "yes"

# Random Forest with great results
set.seed(123)
model_forest <- randomForest(status~ ., data=train, 
                             importance=TRUE,proximity=TRUE,
                             cutoff = c(0.5, 0.5),type="classification", na.action=na.exclude)
print(model_forest)   
plot(model_forest)
importance(model_forest)

varImpPlot(model_forest)#Shows the decrease in accuracy with the removal of a variables so non Annual removal the accuray can go down by 30% for example

predicted <- predict(model_forest, test, type="class") 

print(sprintf("Accuracy of Random Forest for k = 20:    %.3f", Accuracy(y_true=actual, y_pred=predicted)))
print(sprintf("AUC of Random Forest for k = 20:     %.3f", AUC(y_pred=predicted, y_true=actual)))
print(sprintf("Precision of Random Forest for k = 20:   %.3f", Precision(y_true=actual, y_pred=predicted)))
print(sprintf("Recall of Random Forest for k = 20:      %.3f", Recall(y_true=actual, y_pred=predicted)))
print(sprintf("F1 Score of Random Forest for k = 20:    %.3f", F1_Score(predicted, actual)))
print(sprintf("Sensitivity of Random Forest for k = 20: %.3f", Sensitivity(y_true=actual, y_pred=predicted)))
print(sprintf("Specificity of Random Forest for k = 20: %.3f", Specificity(y_true=predicted, y_pred=actual)))

# Naive Bayes...Accuracy of 0.73 not very good
set.seed(123)

nb_fit = naiveBayes(status ~ ., data=train)
nb_fit

NB_pred = predict(nb_fit, test, type="class") 
print(sprintf("Accuracy of NB for k = 20:    %.3f", Accuracy(y_true=actual, y_pred=NB_pred)))
print(sprintf("Precision of NB for k = 20:   %.3f", Precision(y_true=actual, y_pred=NB_pred)))
print(sprintf("F1 Score of NB for k = 20:    %.3f", F1_Score(NB_pred, actual)))
print(sprintf("Sensitivityof NB for k = 20: %.3f", Sensitivity(y_true=actual, y_pred=NB_pred)))
print(sprintf("Specificity of NB for k = 20: %.3f", Specificity(y_true=NB_pred, y_pred=actual)))

# Decision Tree
set.seed(123)

tree <- rpart(status ~ ., method="class", data=train)
tree_pred = predict(tree, test, type="class") 
print(sprintf("Accuracy of DT for k = 20:    %.3f", Accuracy(y_true=actual, y_pred=tree_pred)))
print(sprintf("Precision of DT for k = 20:   %.3f", Precision(y_true=actual, y_pred=tree_pred)))
print(sprintf("F1 Score of DT for k = 20:    %.3f", F1_Score(tree_pred, actual)))
print(sprintf("Sensitivity of DT for k = 20: %.3f", Sensitivity(y_true=actual, y_pred=tree_pred)))
print(sprintf("Specificity of DT for k = 20: %.3f", Specificity(y_true=tree_pred, y_pred=actual)))

#SVM...this is the second best
set.seed(123)
svm_fit <- svm(status ~ ., data = train)
summary(svm_fit)

svm_pred <- predict(svm_fit, test) 

print(sprintf("Accuracy of SVM for k = 20:    %.3f", Accuracy(y_true=actual, y_pred=svm_pred)))
print(sprintf("Precision of SVM for k = 20:   %.3f", Precision(y_true=actual, y_pred=svm_pred)))
print(sprintf("F1 Score of SVM for k = 20:    %.3f", F1_Score(svm_pred, actual)))
print(sprintf("Sensitivity of SVM for k = 20: %.3f", Sensitivity(y_true=actual, y_pred=svm_pred)))
print(sprintf("Specificity of SVM for k = 20: %.3f", Specificity(y_true=svm_pred, y_pred=actual)))


#kknn
set.seed(123)
kknn_fit = train.kknn(status ~ ., train, kmax=7)
summary(kknn_fit)

kknn_pred = predict(kknn_fit, test)

print(sprintf("Accuracy of KNN for k = 20:    %.3f", Accuracy(y_true=actual, y_pred=kknn_pred)))
print(sprintf("Precision of KNN for k = 20:   %.3f", Precision(y_true=actual, y_pred=kknn_pred)))
print(sprintf("F1 Score of KNN for k = 20:    %.3f", F1_Score(kknn_pred, actual)))
print(sprintf("Sensitivity of KNN for k = 20: %.3f", Sensitivity(y_true=actual, y_pred=kknn_pred)))
print(sprintf("Specificity of KNN for k = 20: %.3f", Specificity(y_true=kknn_pred, y_pred=actual)))

# #NN
# install.packages("nnet")
# library(nnet)
# set.seed(123)
# 
# train_nn <- (train)
# test_nn <- test
# str(train_nn)
# 
# train_nn$status <- as.numeric(train_nn$status)
# train_nn$sector <- as.numeric(train_nn$sector)
# train_nn$country <- as.numeric(train_nn$country)
# train_nn$gender <- as.numeric(train_nn$gender)
# train_nn$countrygender <- as.numeric(as.factor(as.character(train_nn$countrygender)))
# train_nn$CountrySector <- as.numeric(as.factor(as.character(train_nn$CountrySector)))
# train_nn$nonpayment <- as.numeric (train_nn$nonpayment)
# 
# test_nn$status <- as.numeric(test_nn$status)
# test_nn$sector <- as.numeric(test_nn$sector)
# test_nn$country <- as.numeric(test_nn$country)
# test_nn$gender <- as.numeric(test_nn$gender)
# test_nn$countrygender <- as.numeric(as.factor(as.character(test_nn$countrygender)))
# test_nn$CountrySector <- as.numeric(as.factor(as.character(test_nn$CountrySector)))
# test_nn$nonpayment <- as.numeric (test_nn$nonpayment)
# str(train)
# ##start Neural Network analysis
# 
# my.grid <- expand.grid( .size = c(1,2,4),.decay = c(0.25,1,2)) # Tuning grid for Neural Net
# 
# model_NN <- train(status~ ., data = train_nn, method = "nnet", tuneGrid = my.grid, trace = TRUE, na.action =na.omit)
# 
# plot(model_NN) #Visualize the relationship between the number of layers, decay and accuracy
# NN_prediction<-predict(model_NN, test_nn) #Predict classification 
# print(sprintf("Accuracy of NN for k = 20:    %.3f", Accuracy(y_true=test_nn$status, y_pred=NN_prediction)))
# print(sprintf("Precision of NN for k = 20:   %.3f", Precision(y_true=test_nn$status, y_pred=NN_prediction)))
# print(sprintf("F1 Score of NN for k = 20:    %.3f", F1_Score(NN_prediction, test_nn$status)))
# print(sprintf("Sensitivity of NN for k = 20: %.3f", Sensitivity(y_true=test_nn$status, y_pred=NN_prediction)))
# print(sprintf("Specificity of NN for k = 20: %.3f", Specificity(y_true=NN_prediction, y_pred=test_nn$status)))


### removing topics from this part
topics <- cbind(topics, df[,14:33])
df[,14:33] <- NULL


##### Topic Modelling with Sensitive Study ###############

#### With k = 10 ##########
##### fitting an LDA Model #####

ten_model <- FitLdaModel(dtm = dtm_2_2, 
                         k = 10, 
                         iterations = 500, # recommend a larger value, 500 or more
                         burnin = 180,
                         optimize_alpha = TRUE,
                         calc_likelihood = TRUE,
                         calc_coherence = TRUE,
                         calc_r2 = TRUE,
                         alpha = 0.1, # this is the default value
                         beta = 0.05, # this is the default value
                         cpus = 21) # Note, this is for a big machine  

plot(ten_model$log_likelihood, type = "l")


ten_model$top_terms <- GetTopTerms(phi = ten_model$phi, M = 8)
head(ten_model$top_terms) 

# probabilistic coherence, a measure of topic quality
# this measure can be used with any topic model, not just probabilistic ones
ten_model$coherence <- CalcProbCoherence(phi = ten_model$phi, dtm = dtm_2_2, M = 5)

# Get the prevalence of each topic
# You can make this discrete by applying a threshold, say 0.05, for
# topics in/out of docuemnts. 
ten_model$prevalence <- colSums(ten_model$theta) / sum(ten_model$theta) * 100


# textmineR has a naive topic labeling tool based on probable bigrams
ten_model$labels <- LabelTopics(assignments = ten_model$theta > 0.20, 
                                dtm = dtm_2_2,
                                M = 2)

head(ten_model$labels)


# put them together, with coherence into a summary table
ten_model$summary <- data.frame(topic = rownames(ten_model$phi),
                                label = ten_model$labels,
                                coherence = round(ten_model$coherence, 2),
                                prevalence = round(ten_model$prevalence,1),
                                top_terms = apply(ten_model$top_terms, 2, function(x){
                                  paste(x, collapse = ", ")
                                }),
                                stringsAsFactors = FALSE)

ten_model$summary %>%
  arrange(-prevalence)

write.csv(ten_model$top_terms, file = "2.2_10_Topics_kiva.csv")

#### Adding topics to data frame #####

df = cbind(df, round(ten_model$theta, digits=2))
head(df, n=10)

####################################################################
########## Model Building for k = 10 ###################

# Splitting the Data into train and test

set.seed(123) # Set the seed to make it reproducible

train.index <- createDataPartition(df$status, p = .8, list = FALSE)
train <- df[ train.index,]
test  <- df[-train.index,]

actual <- test$status
formula <- status ~ .
positive <- "yes"

# Random Forest
set.seed(123)
model_forest <- randomForest(status~ ., data=train, 
                             importance=TRUE,proximity=TRUE,
                             cutoff = c(0.5, 0.5),type="classification", na.action=na.exclude)
print(model_forest)   
plot(model_forest)
importance(model_forest)

varImpPlot(model_forest)#Shows the decrease in accuracy with the removal of a variables so non Annual removal the accuray can go down by 30% for example

predicted <- predict(model_forest, test, type="class") 

print(sprintf("Accuracy of Random Forest for k = 10:    %.3f", Accuracy(y_true=actual, y_pred=predicted)))
print(sprintf("AUC of Random Forest for k = 10:     %.3f", AUC(y_pred=predicted, y_true=actual)))
print(sprintf("Precision of Random Forest for k = 10:   %.3f", Precision(y_true=actual, y_pred=predicted)))
print(sprintf("Recall of Random Forest for k = 10:      %.3f", Recall(y_true=actual, y_pred=predicted)))
print(sprintf("F1 Score of Random Forest for k = 10:    %.3f", F1_Score(predicted, actual)))
print(sprintf("Sensitivity of Random Forest for k = 10: %.3f", Sensitivity(y_true=actual, y_pred=predicted)))
print(sprintf("Specificity of Random Forest for k = 10: %.3f", Specificity(y_true=predicted, y_pred=actual)))

# Naive Bayes
set.seed(123)

nb_fit = naiveBayes(status ~ ., data=train)
nb_fit

NB_pred = predict(nb_fit, test, type="class") 
print(sprintf("Accuracy of NB for k = 10:    %.3f", Accuracy(y_true=actual, y_pred=NB_pred)))
print(sprintf("Precision of NB for k = 10:   %.3f", Precision(y_true=actual, y_pred=NB_pred)))
print(sprintf("F1 Score of NB for k = 10:    %.3f", F1_Score(NB_pred, actual)))
print(sprintf("Sensitivityof NB for k = 10: %.3f", Sensitivity(y_true=actual, y_pred=NB_pred)))
print(sprintf("Specificity of NB for k = 10: %.3f", Specificity(y_true=NB_pred, y_pred=actual)))

# Decision Tree
set.seed(123)

tree <- rpart(status ~ ., method="class", data=train)
tree_pred = predict(tree, test, type="class") 
print(sprintf("Accuracy of DT for k = 10:    %.3f", Accuracy(y_true=actual, y_pred=tree_pred)))
print(sprintf("Precision of DT for k = 10:   %.3f", Precision(y_true=actual, y_pred=tree_pred)))
print(sprintf("F1 Score of DT for k = 10:    %.3f", F1_Score(tree_pred, actual)))
print(sprintf("Sensitivity of DT for k = 10: %.3f", Sensitivity(y_true=actual, y_pred=tree_pred)))
print(sprintf("Specificity of DT for k = 10: %.3f", Specificity(y_true=tree_pred, y_pred=actual)))

#SVM
set.seed(123)
svm_fit <- svm(status ~ ., data = train)
summary(svm_fit)

svm_pred <- predict(svm_fit, test) 

print(sprintf("Accuracy of SVM for k = 10:    %.3f", Accuracy(y_true=actual, y_pred=svm_pred)))
print(sprintf("Precision of SVM for k = 10:   %.3f", Precision(y_true=actual, y_pred=svm_pred)))
print(sprintf("F1 Score of SVM for k = 10:    %.3f", F1_Score(svm_pred, actual)))
print(sprintf("Sensitivity of SVM for k = 10: %.3f", Sensitivity(y_true=actual, y_pred=svm_pred)))
print(sprintf("Specificity of SVM for k = 10: %.3f", Specificity(y_true=svm_pred, y_pred=actual)))


#kknn
set.seed(123)
kknn_fit = train.kknn(status ~ ., train, kmax=7)
summary(kknn_fit)

kknn_pred = predict(kknn_fit, test)

print(sprintf("Accuracy of KNN for k = 10:    %.3f", Accuracy(y_true=actual, y_pred=kknn_pred)))
print(sprintf("Precision of KNN for k = 10:   %.3f", Precision(y_true=actual, y_pred=kknn_pred)))
print(sprintf("F1 Score of KNN for k = 10:    %.3f", F1_Score(kknn_pred, actual)))
print(sprintf("Sensitivity of KNN for k = 10: %.3f", Sensitivity(y_true=actual, y_pred=kknn_pred)))
print(sprintf("Specificity of KNN for k = 10: %.3f", Specificity(y_true=kknn_pred, y_pred=actual)))

# #NN.
# install.packages("nnet")
# library(nnet)
# set.seed(123)
# 
# train_nn <- (train)
# test_nn <- test
# str(train_nn)
# 
# train_nn$status <- as.numeric(train_nn$status)
# train_nn$sector <- as.numeric(train_nn$sector)
# train_nn$country <- as.numeric(train_nn$country)
# train_nn$gender <- as.numeric(train_nn$gender)
# train_nn$countrygender <- as.numeric(as.factor(as.character(train_nn$countrygender)))
# train_nn$CountrySector <- as.numeric(as.factor(as.character(train_nn$CountrySector)))
# train_nn$nonpayment <- as.numeric (train_nn$nonpayment)
# 
# test_nn$status <- as.numeric(test_nn$status)
# test_nn$sector <- as.numeric(test_nn$sector)
# test_nn$country <- as.numeric(test_nn$country)
# test_nn$gender <- as.numeric(test_nn$gender)
# test_nn$countrygender <- as.numeric(as.factor(as.character(test_nn$countrygender)))
# test_nn$CountrySector <- as.numeric(as.factor(as.character(test_nn$CountrySector)))
# test_nn$nonpayment <- as.numeric (test_nn$nonpayment)
# str(train)
# ##start Neural Network analysis
# 
# my.grid <- expand.grid( .size = c(1,2,4),.decay = c(0.25,1,2)) # Tuning grid for Neural Net
# 
# model_NN <- train(status~ ., data = train_nn, method = "nnet", tuneGrid = my.grid, trace = TRUE, na.action =na.omit)
# 
# plot(model_NN) #Visualize the relationship between the number of layers, decay and accuracy
# NN_prediction<-predict(model_NN, test_nn) #Predict classification 
# print(sprintf("Accuracy of NN for k = 10:    %.3f", Accuracy(y_true=test_nn$status, y_pred=NN_prediction)))
# print(sprintf("Precision of NN for k = 10:   %.3f", Precision(y_true=test_nn$status, y_pred=NN_prediction)))
# print(sprintf("F1 Score of NN for k = 10:    %.3f", F1_Score(NN_prediction, test_nn$status)))
# print(sprintf("Sensitivity of NN for k = 10: %.3f", Sensitivity(y_true=test_nn$status, y_pred=NN_prediction)))
# print(sprintf("Specificity of NN for k = 10: %.3f", Specificity(y_true=NN_prediction, y_pred=test_nn$status)))

### removing topics from this part
topics <- cbind(topics, df[,14:23])
### removing topics from this part
df[,14:23] <- NULL


##### Topic Modelling with Sensitive Study ###############

#### With k = 15 ##########

##### fitting an LDA Model #####

fifteen_model <- FitLdaModel(dtm = dtm_2_2, 
                             k = 15, 
                             iterations = 500, # recommend a larger value, 500 or more
                             burnin = 180,
                             optimize_alpha = TRUE,
                             calc_likelihood = TRUE,
                             calc_coherence = TRUE,
                             calc_r2 = TRUE,
                             alpha = 0.1, # this is the default value
                             beta = 0.05, # this is the default value
                             cpus = 21) # Note, this is for a big machine  

plot(fifteen_model$log_likelihood, type = "l")


fifteen_model$top_terms <- GetTopTerms(phi = fifteen_model$phi, M = 8)
head(fifteen_model$top_terms) 

# probabilistic coherence, a measure of topic quality
# this measure can be used with any topic model, not just probabilistic ones
fifteen_model$coherence <- CalcProbCoherence(phi = fifteen_model$phi, dtm = dtm_2_2, M = 5)

# Get the prevalence of each topic
# You can make this discrete by applying a threshold, say 0.05, for
# topics in/out of docuemnts. 
fifteen_model$prevalence <- colSums(fifteen_model$theta) / sum(fifteen_model$theta) * 100


# textmineR has a naive topic labeling tool based on probable bigrams
fifteen_model$labels <- LabelTopics(assignments = fifteen_model$theta > 0.20, 
                                    dtm = dtm_2_2,
                                    M = 2)

head(fifteen_model$labels)


# put them together, with coherence into a summary table
fifteen_model$summary <- data.frame(topic = rownames(fifteen_model$phi),
                                    label = fifteen_model$labels,
                                    coherence = round(fifteen_model$coherence, 2),
                                    prevalence = round(fifteen_model$prevalence,1),
                                    top_terms = apply(fifteen_model$top_terms, 2, function(x){
                                      paste(x, collapse = ", ")
                                    }),
                                    stringsAsFactors = FALSE)

fifteen_model$summary %>%
  arrange(-prevalence)

write.csv(fifteen_model$top_terms, file = "2.2_15_Topics_kiva.csv")

#### Adding topics to data frame #####

df = cbind(df, round(fifteen_model$theta, digits=2))
head(df, n=10)

####################################################################
########## Model Building for k = 15 ###################

# Splitting the Data into train and test

set.seed(123) # Set the seed to make it reproducible

train.index <- createDataPartition(df$status, p = .8, list = FALSE)
train <- df[ train.index,]
test  <- df[-train.index,]

actual <- test$status
formula <- status ~ .
positive <- "yes"

# Random Forest with great results
set.seed(123)
model_forest <- randomForest(status~ ., data=train, 
                             importance=TRUE,proximity=TRUE,
                             cutoff = c(0.5, 0.5),type="classification", na.action=na.exclude)
print(model_forest)   
plot(model_forest)
importance(model_forest)

varImpPlot(model_forest)#Shows the decrease in accuracy with the removal of a variables so non Annual removal the accuray can go down by 30% for example

predicted <- predict(model_forest, test, type="class") 

print(sprintf("Accuracy of Random Forest for k = 15:    %.3f", Accuracy(y_true=actual, y_pred=predicted)))
print(sprintf("AUC of Random Forest for k = 15:     %.3f", AUC(y_pred=predicted, y_true=actual)))
print(sprintf("Precision of Random Forest for k = 15:   %.3f", Precision(y_true=actual, y_pred=predicted)))
print(sprintf("Recall of Random Forest for k = 15:      %.3f", Recall(y_true=actual, y_pred=predicted)))
print(sprintf("F1 Score of Random Forest for k = 15:    %.3f", F1_Score(predicted, actual)))
print(sprintf("Sensitivity of Random Forest for k = 15: %.3f", Sensitivity(y_true=actual, y_pred=predicted)))
print(sprintf("Specificity of Random Forest for k = 15: %.3f", Specificity(y_true=predicted, y_pred=actual)))

# Naive Bayes
set.seed(123)

nb_fit = naiveBayes(status ~ ., data=train)
nb_fit

NB_pred = predict(nb_fit, test, type="class") 
print(sprintf("Accuracy of NB for k = 15:    %.3f", Accuracy(y_true=actual, y_pred=NB_pred)))
print(sprintf("Precision of NB for k = 15:   %.3f", Precision(y_true=actual, y_pred=NB_pred)))
print(sprintf("F1 Score of NB for k = 15:    %.3f", F1_Score(NB_pred, actual)))
print(sprintf("Sensitivityof NB for k = 15: %.3f", Sensitivity(y_true=actual, y_pred=NB_pred)))
print(sprintf("Specificity of NB for k = 15: %.3f", Specificity(y_true=NB_pred, y_pred=actual)))

# Decision Tree
set.seed(123)

tree <- rpart(status ~ ., method="class", data=train)
tree_pred = predict(tree, test, type="class") 
print(sprintf("Accuracy of DT for k = 15:    %.3f", Accuracy(y_true=actual, y_pred=tree_pred)))
print(sprintf("Precision of DT for k = 15:   %.3f", Precision(y_true=actual, y_pred=tree_pred)))
print(sprintf("F1 Score of DT for k = 15:    %.3f", F1_Score(tree_pred, actual)))
print(sprintf("Sensitivity of DT for k = 15: %.3f", Sensitivity(y_true=actual, y_pred=tree_pred)))
print(sprintf("Specificity of DT for k = 15: %.3f", Specificity(y_true=tree_pred, y_pred=actual)))

#SVM.
set.seed(123)
svm_fit <- svm(status ~ ., data = train)
summary(svm_fit)

svm_pred <- predict(svm_fit, test) 

print(sprintf("Accuracy of SVM for k = 15:    %.3f", Accuracy(y_true=actual, y_pred=svm_pred)))
print(sprintf("Precision of SVM for k = 15:   %.3f", Precision(y_true=actual, y_pred=svm_pred)))
print(sprintf("F1 Score of SVM for k = 15:    %.3f", F1_Score(svm_pred, actual)))
print(sprintf("Sensitivity of SVM for k = 15: %.3f", Sensitivity(y_true=actual, y_pred=svm_pred)))
print(sprintf("Specificity of SVM for k = 15: %.3f", Specificity(y_true=svm_pred, y_pred=actual)))


#kknn.
set.seed(123)

kknn_fit = train.kknn(status ~ ., train, kmax=7)
summary(kknn_fit)

kknn_pred = predict(kknn_fit, test)

print(sprintf("Accuracy of KNN for k = 15:    %.3f", Accuracy(y_true=actual, y_pred=kknn_pred)))
print(sprintf("Precision of KNN for k = 15:   %.3f", Precision(y_true=actual, y_pred=kknn_pred)))
print(sprintf("F1 Score of KNN for k = 15:    %.3f", F1_Score(kknn_pred, actual)))
print(sprintf("Sensitivity of KNN for k = 15: %.3f", Sensitivity(y_true=actual, y_pred=kknn_pred)))
print(sprintf("Specificity of KNN for k = 15: %.3f", Specificity(y_true=kknn_pred, y_pred=actual)))

# #NN...
# install.packages("nnet")
# library(nnet)
# set.seed(123)
# 
# train_nn <- (train)
# test_nn <- test
# str(train_nn)
# 
# train_nn$status <- as.numeric(train_nn$status)
# train_nn$sector <- as.numeric(train_nn$sector)
# train_nn$country <- as.numeric(train_nn$country)
# train_nn$gender <- as.numeric(train_nn$gender)
# train_nn$countrygender <- as.numeric(as.factor(as.character(train_nn$countrygender)))
# train_nn$CountrySector <- as.numeric(as.factor(as.character(train_nn$CountrySector)))
# train_nn$nonpayment <- as.numeric (train_nn$nonpayment)
# 
# test_nn$status <- as.numeric(test_nn$status)
# test_nn$sector <- as.numeric(test_nn$sector)
# test_nn$country <- as.numeric(test_nn$country)
# test_nn$gender <- as.numeric(test_nn$gender)
# test_nn$countrygender <- as.numeric(as.factor(as.character(test_nn$countrygender)))
# test_nn$CountrySector <- as.numeric(as.factor(as.character(test_nn$CountrySector)))
# test_nn$nonpayment <- as.numeric (test_nn$nonpayment)
# str(train)
# ##start Neural Network analysis
# 
# my.grid <- expand.grid( .size = c(1,2,4),.decay = c(0.25,1,2)) # Tuning grid for Neural Net
# 
# model_NN <- train(status~ ., data = train_nn, method = "nnet", tuneGrid = my.grid, trace = TRUE, na.action =na.omit)
# 
# plot(model_NN) #Visualize the relationship between the number of layers, decay and accuracy
# NN_prediction<-predict(model_NN, test_nn) #Predict classification 
# print(sprintf("Accuracy of NN for k = 15:    %.3f", Accuracy(y_true=test_nn$status, y_pred=NN_prediction)))
# print(sprintf("Precision of NN for k = 15:   %.3f", Precision(y_true=test_nn$status, y_pred=NN_prediction)))
# print(sprintf("F1 Score of NN for k = 15:    %.3f", F1_Score(NN_prediction, test_nn$status)))
# print(sprintf("Sensitivity of NN for k = 15: %.3f", Sensitivity(y_true=test_nn$status, y_pred=NN_prediction)))
# print(sprintf("Specificity of NN for k = 15: %.3f", Specificity(y_true=NN_prediction, y_pred=test_nn$status)))

### removing topics from this part
topics <- cbind(topics, df[,14:28])
### removing topics from this part
df[,14:28] <- NULL


##### Topic Modelling with Sensitive Study ###############

#### With k = 25 ##########

##### fitting an LDA Model #####

twentyfive_model <- FitLdaModel(dtm = dtm_2_2, 
                                k = 25, 
                                iterations = 500, # recommend a larger value, 500 or more
                                burnin = 180,
                                optimize_alpha = TRUE,
                                calc_likelihood = TRUE,
                                calc_coherence = TRUE,
                                calc_r2 = TRUE,
                                alpha = 0.1, # this is the default value
                                beta = 0.05, # this is the default value
                                cpus = 21) # Note, this is for a big machine  

plot(twentyfive_model$log_likelihood, type = "l")

twentyfive_model$top_terms <- GetTopTerms(phi = twentyfive_model$phi, M = 8)
head(twentyfive_model$top_terms) 

# probabilistic coherence, a measure of topic quality
# this measure can be used with any topic model, not just probabilistic ones
twentyfive_model$coherence <- CalcProbCoherence(phi = twentyfive_model$phi, dtm = dtm_2_2, M = 5)

# Get the prevalence of each topic
# You can make this discrete by applying a threshold, say 0.05, for
# topics in/out of docuemnts. 
twentyfive_model$prevalence <- colSums(twentyfive_model$theta) / sum(twentyfive_model$theta) * 100


# textmineR has a naive topic labeling tool based on probable bigrams
twentyfive_model$labels <- LabelTopics(assignments = twentyfive_model$theta > 0.20, 
                                       dtm = dtm_2_2,
                                       M = 2)

head(twentyfive_model$labels)


# put them together, with coherence into a summary table
twentyfive_model$summary <- data.frame(topic = rownames(twentyfive_model$phi),
                                       label = twentyfive_model$labels,
                                       coherence = round(twentyfive_model$coherence, 2),
                                       prevalence = round(twentyfive_model$prevalence,1),
                                       top_terms = apply(twentyfive_model$top_terms, 2, function(x){
                                         paste(x, collapse = ", ")
                                       }),
                                       stringsAsFactors = FALSE)

twentyfive_model$summary %>%
  arrange(-prevalence)

write.csv(twentyfive_model$top_terms, file = "2.2_25_Topics_kiva.csv")

#### Adding topics to data frame #####

df = cbind(df, round(twentyfive_model$theta, digits=2))
head(df, n=10)

####################################################################
########## Model Building for k = 15 ###################

# Splitting the Data into train and test

set.seed(123) # Set the seed to make it reproducible

train.index <- createDataPartition(df$status, p = .8, list = FALSE)
train <- df[ train.index,]
test  <- df[-train.index,]

actual <- test$status
formula <- status ~ .
positive <- "yes"

# Random Forest with great results
set.seed(123)
model_forest <- randomForest(status~ ., data=train, 
                             importance=TRUE,proximity=TRUE,
                             cutoff = c(0.5, 0.5),type="classification", na.action=na.exclude)
print(model_forest)   
plot(model_forest)
importance(model_forest)

varImpPlot(model_forest)#Shows the decrease in accuracy with the removal of a variables so non Annual removal the accuray can go down by 30% for example

predicted <- predict(model_forest, test, type="class") 

print(sprintf("Accuracy of Random Forest for k = 25:    %.3f", Accuracy(y_true=actual, y_pred=predicted)))
print(sprintf("AUC of Random Forest for k = 25:     %.3f", AUC(y_pred=predicted, y_true=actual)))
print(sprintf("Precision of Random Forest for k = 25:   %.3f", Precision(y_true=actual, y_pred=predicted)))
print(sprintf("Recall of Random Forest for k = 25:      %.3f", Recall(y_true=actual, y_pred=predicted)))
print(sprintf("F1 Score of Random Forest for k = 25:    %.3f", F1_Score(predicted, actual)))
print(sprintf("Sensitivity of Random Forest for k = 25: %.3f", Sensitivity(y_true=actual, y_pred=predicted)))
print(sprintf("Specificity of Random Forest for k = 25: %.3f", Specificity(y_true=predicted, y_pred=actual)))

# Naive Bayes.
set.seed(123)

nb_fit = naiveBayes(status ~ ., data=train)
nb_fit

NB_pred = predict(nb_fit, test, type="class") 
print(sprintf("Accuracy of NB for k = 25:    %.3f", Accuracy(y_true=actual, y_pred=NB_pred)))
print(sprintf("Precision of NB for k = 25:   %.3f", Precision(y_true=actual, y_pred=NB_pred)))
print(sprintf("F1 Score of NB for k = 25:    %.3f", F1_Score(NB_pred, actual)))
print(sprintf("Sensitivityof NB for k = 25: %.3f", Sensitivity(y_true=actual, y_pred=NB_pred)))
print(sprintf("Specificity of NB for k = 25: %.3f", Specificity(y_true=NB_pred, y_pred=actual)))

# Decision Tree
set.seed(123)

tree <- rpart(status ~ ., method="class", data=train)
tree_pred = predict(tree, test, type="class") 
print(sprintf("Accuracy of DT for k = 25:    %.3f", Accuracy(y_true=actual, y_pred=tree_pred)))
print(sprintf("Precision of DT for k = 25:   %.3f", Precision(y_true=actual, y_pred=tree_pred)))
print(sprintf("F1 Score of DT for k = 25:    %.3f", F1_Score(tree_pred, actual)))
print(sprintf("Sensitivity of DT for k = 25: %.3f", Sensitivity(y_true=actual, y_pred=tree_pred)))
print(sprintf("Specificity of DT for k = 25: %.3f", Specificity(y_true=tree_pred, y_pred=actual)))

#SVM.
set.seed(123)
svm_fit <- svm(status ~ ., data = train)
summary(svm_fit)

svm_pred <- predict(svm_fit, test) 

print(sprintf("Accuracy of SVM for k = 25:    %.3f", Accuracy(y_true=actual, y_pred=svm_pred)))
print(sprintf("Precision of SVM for k = 25:   %.3f", Precision(y_true=actual, y_pred=svm_pred)))
print(sprintf("F1 Score of SVM for k = 25:    %.3f", F1_Score(svm_pred, actual)))
print(sprintf("Sensitivity of SVM for k = 25: %.3f", Sensitivity(y_true=actual, y_pred=svm_pred)))
print(sprintf("Specificity of SVM for k = 25: %.3f", Specificity(y_true=svm_pred, y_pred=actual)))


#kknn...
set.seed(123)

kknn_fit = train.kknn(status ~ ., train, kmax=7)
summary(kknn_fit)

kknn_pred = predict(kknn_fit, test)

print(sprintf("Accuracy of KNN for k = 25:    %.3f", Accuracy(y_true=actual, y_pred=kknn_pred)))
print(sprintf("Precision of KNN for k = 25:   %.3f", Precision(y_true=actual, y_pred=kknn_pred)))
print(sprintf("F1 Score of KNN for k = 25:    %.3f", F1_Score(kknn_pred, actual)))
print(sprintf("Sensitivity of KNN for k = 25: %.3f", Sensitivity(y_true=actual, y_pred=kknn_pred)))
print(sprintf("Specificity of KNN for k = 25: %.3f", Specificity(y_true=kknn_pred, y_pred=actual)))

# #NN...91%
# install.packages("nnet")
# library(nnet)
# set.seed(123)
# 
# train_nn <- (train)
# test_nn <- test
# str(train_nn)
# 
# train_nn$status <- as.numeric(train_nn$status)
# train_nn$sector <- as.numeric(train_nn$sector)
# train_nn$country <- as.numeric(train_nn$country)
# train_nn$gender <- as.numeric(train_nn$gender)
# train_nn$countrygender <- as.numeric(as.factor(as.character(train_nn$countrygender)))
# train_nn$CountrySector <- as.numeric(as.factor(as.character(train_nn$CountrySector)))
# train_nn$nonpayment <- as.numeric (train_nn$nonpayment)
# 
# test_nn$status <- as.numeric(test_nn$status)
# test_nn$sector <- as.numeric(test_nn$sector)
# test_nn$country <- as.numeric(test_nn$country)
# test_nn$gender <- as.numeric(test_nn$gender)
# test_nn$countrygender <- as.numeric(as.factor(as.character(test_nn$countrygender)))
# test_nn$CountrySector <- as.numeric(as.factor(as.character(test_nn$CountrySector)))
# test_nn$nonpayment <- as.numeric (test_nn$nonpayment)
# str(train)
# ##start Neural Network analysis
# 
# my.grid <- expand.grid( .size = c(1,2,4),.decay = c(0.25,1,2)) # Tuning grid for Neural Net
# 
# model_NN <- train(status~ ., data = train_nn, method = "nnet", tuneGrid = my.grid, trace = TRUE, na.action =na.omit)
# 
# plot(model_NN) #Visualize the relationship between the number of layers, decay and accuracy
# NN_prediction<-predict(model_NN, test_nn) #Predict classification 
# print(sprintf("Accuracy of NN for k = 25:    %.3f", Accuracy(y_true=test_nn$status, y_pred=NN_prediction)))
# print(sprintf("Precision of NN for k = 25:   %.3f", Precision(y_true=test_nn$status, y_pred=NN_prediction)))
# print(sprintf("F1 Score of NN for k = 25:    %.3f", F1_Score(NN_prediction, test_nn$status)))
# print(sprintf("Sensitivity of NN for k = 25: %.3f", Sensitivity(y_true=test_nn$status, y_pred=NN_prediction)))
# print(sprintf("Specificity of NN for k = 25: %.3f", Specificity(y_true=NN_prediction, y_pred=test_nn$status)))

### removing topics from this part
topics <- cbind(topics, df[,14:38])
### removing topics from this part
df[,14:38] <- NULL

##### Topic Modelling with Sensitive Study ###############

#### With k = 30 ##########

##### fitting an LDA Model #####

thirty_model <- FitLdaModel(dtm = dtm_2_2, 
                            k = 30, 
                            iterations = 500, # recommend a larger value, 500 or more
                            burnin = 180,
                            optimize_alpha = TRUE,
                            calc_likelihood = TRUE,
                            calc_coherence = TRUE,
                            calc_r2 = TRUE,
                            alpha = 0.1, # this is the default value
                            beta = 0.05, # this is the default value
                            cpus = 21) # Note, this is for a big machine  



# #NN...91%
# install.packages("nnet")
# library(nnet)
# set.seed(123)
# 
# train_nn <- (train)
# test_nn <- test
# str(train_nn)
# 
# train_nn$status <- as.numeric(train_nn$status)
# train_nn$sector <- as.numeric(train_nn$sector)
# train_nn$country <- as.numeric(train_nn$country)
# train_nn$gender <- as.numeric(train_nn$gender)
# train_nn$countrygender <- as.numeric(as.factor(as.character(train_nn$countrygender)))
# train_nn$CountrySector <- as.numeric(as.factor(as.character(train_nn$CountrySector)))
# train_nn$nonpayment <- as.numeric (train_nn$nonpayment)
# 
# test_nn$status <- as.numeric(test_nn$status)
# test_nn$sector <- as.numeric(test_nn$sector)
# test_nn$country <- as.numeric(test_nn$country)
# test_nn$gender <- as.numeric(test_nn$gender)
# test_nn$countrygender <- as.numeric(as.factor(as.character(test_nn$countrygender)))
# test_nn$CountrySector <- as.numeric(as.factor(as.character(test_nn$CountrySector)))
# test_nn$nonpayment <- as.numeric (test_nn$nonpayment)
# str(train)
# ##start Neural Network analysis
# 
# my.grid <- expand.grid( .size = c(1,2,4),.decay = c(0.25,1,2)) # Tuning grid for Neural Net
# 
# model_NN <- train(status~ ., data = train_nn, method = "nnet", tuneGrid = my.grid, trace = TRUE, na.action =na.omit)
# 
# plot(model_NN) #Visualize the relationship between the number of layers, decay and accuracy
# NN_prediction<-predict(model_NN, test_nn) #Predict classification 
# print(sprintf("Accuracy of NN for k = 30:    %.3f", Accuracy(y_true=test_nn$status, y_pred=NN_prediction)))
# print(sprintf("Precision of NN for k = 30:   %.3f", Precision(y_true=test_nn$status, y_pred=NN_prediction)))
# print(sprintf("F1 Score of NN for k = 30:    %.3f", F1_Score(NN_prediction, test_nn$status)))
# print(sprintf("Sensitivity of NN for k = 30: %.3f", Sensitivity(y_true=test_nn$status, y_pred=NN_prediction)))
# print(sprintf("Specificity of NN for k = 30: %.3f", Specificity(y_true=NN_prediction, y_pred=test_nn$status)))

### removing topics from this part
topics <- cbind(topics,df[,14:43])
### removing topics from this part
df[,14:43] <- NULL



##########################################################################
######### Topic Modelling ################################################
################### LDA #######################

df <- read.csv("D:\\kiva_data.csv", stringsAsFactors=FALSE)

########## Renaming and type setting of attributes within the dataset ###########

df = df %>%
  rename(story = en)
str(df)
df$id = 1:nrow(df)
df$status = as.factor(df$status)
df$sector = as.factor(df$sector)
df$country = as.factor(df$country)
df$gender = as.factor(df$gender)
df$nonpayment = as.factor(df$nonpayment)

########## Remove HTML Tags ###################
df = df %>% 
  mutate(story = gsub("<.*?>", "", story))

########## Retain distinct entries ################

dim(df)
df = distinct(df, story, .keep_all = TRUE)
dim(df)

############## Clustering the applicants based on all attributes but the story #######

cl <- df
cl$X <- NULL
cl$loan_nonpayment <- NULL
cl$id <- NULL
str(cl)
cl$status <- as.numeric(cl$status)
cl$sector <- as.numeric(cl$sector)
cl$story <- NULL
cl$country <- as.numeric(cl$country)
cl$gender <- as.numeric(cl$gender)
cl$countrygender <- as.numeric(as.factor(as.character(cl$countrygender)))
cl$CountrySector <- as.numeric(as.factor(as.character(cl$CountrySector)))
cl$loan_amount <- as.numeric (cl$loan_amount)
cl$nonpayment <- as.numeric (cl$nonpayment)

## Elbow plot
wss <- (nrow(cl)-1)*sum(apply(cl,2,var))

for (i in 1:30) wss[i] <- sum(kmeans(cl, 
                                     centers=i)$withinss)
plot(1:30, wss, type="b", xlab="Number of Clusters",
     ylab="Within groups sum of squares")

## Choose 4 clusters 

clust <- kmeans(cl, centers = 4, n = 4)

head(clust$cluster)
df <- cbind(df, cluster_name = clust$cluster)

#################### Sentiment Detection ######################

Sentiment <- sentiment_by(df$story)
summary(Sentiment$ave_sentiment)

qplot(Sentiment$ave_sentiment, geom = "histogram", binwidth = 0.1, main = "Review Sentiment Histogram")
head(sentiment)
df$ave_sentiment <- Sentiment$ave_sentiment
df$sd_sentiment <- Sentiment$sd
df$word_count <- Sentiment$word_count

rm(Sentiment, cl, clust)
########## Removing text that is not neccessary for topic identification or further analysis #############
dim(df)
df = distinct(df, story, .keep_all = TRUE)
dim(df)

idx = grep("Translated from Spanish", df$story)
df$story[idx[1]]
df$story = gsub("Translated from Spanish.*Kiva volunteer", " ", df$story, ignore.case = TRUE)
grep("Translated from Spanish", df$story)

grep("Mifex offers", df$story)
df$story = gsub("Mifex offers.*www.mifex.org", " ", df$story, ignore.case = TRUE)
grep("Mifex offers", df$story)

df$story = gsub("About KADET:.*within those communities", " ", df$story, ignore.case = TRUE)
df$story = gsub("Disclaimer: Due to recent events in.*making their loan.", " ", df$story, ignore.case = TRUE)



########## Topic Modelling ######################################
################### LDA #######################

####### Create Document Term Matrix ###########
# setting n-gram window to be exactly 1 -- unigrams are necessary for our analysis ########## Eliminating stopwords from our text based on english, french, spanish and SMART distionaries ##### converting all uppercase letters to lowercase ##### removing both puctuation and numbers from our text #### using Textstem library, we are lemmatizing words to their root forms instead of stemming #########

dtm_3_3 <- CreateDtm(doc_vec = df$story, # character vector of documents
                     doc_names = df$id, # document names
                     ngram_window = c(3,3), # minimum and maximum n-gram length
                     stopword_vec = c(tm::stopwords("english"), # stopwords from tm
                                      tm::stopwords("french"), # stopwords from tm
                                      tm::stopwords("spanish"), # stopwords from tm
                                      tm::stopwords("SMART")), # this is the default value
                     lower = TRUE, # lowercase - this is the default value
                     remove_punctuation = TRUE, # punctuation - this is the default
                     remove_numbers = TRUE, # numbers - this is the default
                     verbose = FALSE, # Turn off status bar for this demo,
                     stem_lemma_function = function(x) textstem::lemmatize_words(x),
                     cpus = 20) # default is all available cpus on the system


# Filter rare words
dim(dtm_3_3)
dtm_3_3 <- dtm_3_3[ , colSums(dtm_3_3 > 0) > 50 ]
dim(dtm_3_3)

max_num = nrow(dtm_3_3) * 0.3
max_num
dtm_3_3 <- dtm_3_3[ , colSums(dtm_3_3 > 0) <= max_num ]
dim(dtm_3_3)

# Capture how long each document is
df$doc_lengths = rowSums(dtm_3_3)

### identify frequent terms and highest 50 IDFs ###
tf_mat <- TermDocFreq(dtm = dtm_3_3)
head(tf_mat[ order(tf_mat$term_freq, decreasing = TRUE) , ], 50)
# write.csv(tf_mat, "term_frequency.csv")
head(tf_mat[ order(tf_mat$idf, decreasing = TRUE) , ], 50)

# look at the most frequent bigrams
tf_bigrams <- tf_mat[ stringr::str_detect(tf_mat$term, "_") , ]
head(tf_bigrams[ order(tf_bigrams$term_freq, decreasing = TRUE) , ], 50)
write.csv(head(tf_bigrams[ order(tf_bigrams$term_freq, decreasing = TRUE) , ], 50), "3.3_kiva_bi_grams.csv")

# look at the most frequent trigrams
tf_trigrams <- tf_mat[ stringr::str_detect(tf_mat$term, "_") , ]
head(tf_trigrams[ order(tf_trigrams$term_freq, decreasing = TRUE) , ], 50)
write.csv(head(tf_trigrams[ order(tf_trigrams$term_freq, decreasing = TRUE) , ], 50), "3.3_kiva_tri_grams.csv")


##### fitting an LDA Model #####

twenty_model <- FitLdaModel(dtm = dtm_3_3, 
                            k = 20, 
                            iterations = 500, # recommend a larger value, 500 or more
                            burnin = 180,
                            optimize_alpha = TRUE,
                            calc_likelihood = TRUE,
                            calc_coherence = TRUE,
                            calc_r2 = TRUE,
                            alpha = 0.1, # this is the default value
                            beta = 0.05, # this is the default value
                            cpus = 21) # Note, this is for a big machine  

plot(twenty_model$log_likelihood, type = "l")


twenty_model$top_terms <- GetTopTerms(phi = twenty_model$phi, M = 8)
head(twenty_model$top_terms) 

# probabilistic coherence, a measure of topic quality
# this measure can be used with any topic model, not just probabilistic ones
twenty_model$coherence <- CalcProbCoherence(phi = twenty_model$phi, dtm = dtm_3_3, M = 5)

# Get the prevalence of each topic
# You can make this discrete by applying a threshold, say 0.05, for
# topics in/out of docuemnts. 
twenty_model$prevalence <- colSums(twenty_model$theta) / sum(twenty_model$theta) * 100


# textmineR has a naive topic labeling tool based on probable bigrams
twenty_model$labels <- LabelTopics(assignments = twenty_model$theta > 0.20, 
                                   dtm = dtm_3_3,
                                   M = 2)

head(twenty_model$labels)


# put them together, with coherence into a summary table
twenty_model$summary <- data.frame(topic = rownames(twenty_model$phi),
                                   label = twenty_model$labels,
                                   coherence = round(twenty_model$coherence, 2),
                                   prevalence = round(twenty_model$prevalence,1),
                                   top_terms = apply(twenty_model$top_terms, 2, function(x){
                                     paste(x, collapse = ", ")
                                   }),
                                   stringsAsFactors = FALSE)

twenty_model$summary %>%
  arrange(-prevalence)

write.csv(twenty_model$top_terms, file = "3.3_20_Topics_kiva.csv")

#### Adding topics to data frame #####

df = cbind(df, round(twenty_model$theta, digits=2))
head(df, n=10)

##### Model Building for Classification ######
# Helper function to print the confusion matrix and other performance metrics of the models.
printPerformance = function(pred, actual, positive="Yes") {
  print(caret::confusionMatrix(data=pred, reference=actual, positive=positive, dnn=c("Predicted", "Actual")))
}

#### Data Type casting ####

str(df$status)
str(df)
df$id <- NULL
df <- as.data.frame(unclass(df))#converting all characters variables to factors
str(df)
df$story <- NULL

# Splitting the Data into train and test

set.seed(123) # Set the seed to make it reproducible

train.index <- createDataPartition(df$status, p = .8, list = FALSE)
train <- df[ train.index,]
test  <- df[-train.index,]

actual <- test$status
formula <- status ~ .
positive <- "yes"

# Random Forest with great results
set.seed(123)
model_forest <- randomForest(status~ ., data=train, 
                             importance=TRUE,proximity=TRUE,
                             cutoff = c(0.5, 0.5),type="classification", na.action=na.exclude)
print(model_forest)   
plot(model_forest)
importance(model_forest)

varImpPlot(model_forest)#Shows the decrease in accuracy with the removal of a variables so non Annual removal the accuray can go down by 30% for example

predicted <- predict(model_forest, test, type="class") 

print(sprintf("Accuracy of Random Forest for k = 20:    %.3f", Accuracy(y_true=actual, y_pred=predicted)))
print(sprintf("AUC of Random Forest for k = 20:     %.3f", AUC(y_pred=predicted, y_true=actual)))
print(sprintf("Precision of Random Forest for k = 20:   %.3f", Precision(y_true=actual, y_pred=predicted)))
print(sprintf("Recall of Random Forest for k = 20:      %.3f", Recall(y_true=actual, y_pred=predicted)))
print(sprintf("F1 Score of Random Forest for k = 20:    %.3f", F1_Score(predicted, actual)))
print(sprintf("Sensitivity of Random Forest for k = 20: %.3f", Sensitivity(y_true=actual, y_pred=predicted)))
print(sprintf("Specificity of Random Forest for k = 20: %.3f", Specificity(y_true=predicted, y_pred=actual)))

# Naive Bayes.
set.seed(123)

nb_fit = naiveBayes(status ~ ., data=train)
nb_fit

NB_pred = predict(nb_fit, test, type="class") 
print(sprintf("Accuracy of NB for k = 20:    %.3f", Accuracy(y_true=actual, y_pred=NB_pred)))
print(sprintf("Precision of NB for k = 20:   %.3f", Precision(y_true=actual, y_pred=NB_pred)))
print(sprintf("F1 Score of NB for k = 20:    %.3f", F1_Score(NB_pred, actual)))
print(sprintf("Sensitivityof NB for k = 20: %.3f", Sensitivity(y_true=actual, y_pred=NB_pred)))
print(sprintf("Specificity of NB for k = 20: %.3f", Specificity(y_true=NB_pred, y_pred=actual)))

# Decision Tree
set.seed(123)

tree <- rpart(status ~ ., method="class", data=train)
tree_pred = predict(tree, test, type="class") 
print(sprintf("Accuracy of DT for k = 20:    %.3f", Accuracy(y_true=actual, y_pred=tree_pred)))
print(sprintf("Precision of DT for k = 20:   %.3f", Precision(y_true=actual, y_pred=tree_pred)))
print(sprintf("F1 Score of DT for k = 20:    %.3f", F1_Score(tree_pred, actual)))
print(sprintf("Sensitivity of DT for k = 20: %.3f", Sensitivity(y_true=actual, y_pred=tree_pred)))
print(sprintf("Specificity of DT for k = 20: %.3f", Specificity(y_true=tree_pred, y_pred=actual)))

#SVM.
set.seed(123)
svm_fit <- svm(status ~ ., data = train)
summary(svm_fit)

svm_pred <- predict(svm_fit, test) 

print(sprintf("Accuracy of SVM for k = 20:    %.3f", Accuracy(y_true=actual, y_pred=svm_pred)))
print(sprintf("Precision of SVM for k = 20:   %.3f", Precision(y_true=actual, y_pred=svm_pred)))
print(sprintf("F1 Score of SVM for k = 20:    %.3f", F1_Score(svm_pred, actual)))
print(sprintf("Sensitivity of SVM for k = 20: %.3f", Sensitivity(y_true=actual, y_pred=svm_pred)))
print(sprintf("Specificity of SVM for k = 20: %.3f", Specificity(y_true=svm_pred, y_pred=actual)))


#kknn...
set.seed(123)
kknn_fit = train.kknn(status ~ ., train, kmax=7)
summary(kknn_fit)

kknn_pred = predict(kknn_fit, test)

print(sprintf("Accuracy of KNN for k = 20:    %.3f", Accuracy(y_true=actual, y_pred=kknn_pred)))
print(sprintf("Precision of KNN for k = 20:   %.3f", Precision(y_true=actual, y_pred=kknn_pred)))
print(sprintf("F1 Score of KNN for k = 20:    %.3f", F1_Score(kknn_pred, actual)))
print(sprintf("Sensitivity of KNN for k = 20: %.3f", Sensitivity(y_true=actual, y_pred=kknn_pred)))
print(sprintf("Specificity of KNN for k = 20: %.3f", Specificity(y_true=kknn_pred, y_pred=actual)))

# #NN...91%
# install.packages("nnet")
# library(nnet)
# set.seed(123)
# 
# train_nn <- (train)
# test_nn <- test
# str(train_nn)
# 
# train_nn$status <- as.numeric(train_nn$status)
# train_nn$sector <- as.numeric(train_nn$sector)
# train_nn$country <- as.numeric(train_nn$country)
# train_nn$gender <- as.numeric(train_nn$gender)
# train_nn$countrygender <- as.numeric(as.factor(as.character(train_nn$countrygender)))
# train_nn$CountrySector <- as.numeric(as.factor(as.character(train_nn$CountrySector)))
# train_nn$nonpayment <- as.numeric (train_nn$nonpayment)
# 
# test_nn$status <- as.numeric(test_nn$status)
# test_nn$sector <- as.numeric(test_nn$sector)
# test_nn$country <- as.numeric(test_nn$country)
# test_nn$gender <- as.numeric(test_nn$gender)
# test_nn$countrygender <- as.numeric(as.factor(as.character(test_nn$countrygender)))
# test_nn$CountrySector <- as.numeric(as.factor(as.character(test_nn$CountrySector)))
# test_nn$nonpayment <- as.numeric (test_nn$nonpayment)
# str(train)
# ##start Neural Network analysis
# 
# my.grid <- expand.grid( .size = c(1,2,4),.decay = c(0.25,1,2)) # Tuning grid for Neural Net
# 
# model_NN <- train(status~ ., data = train_nn, method = "nnet", tuneGrid = my.grid, trace = TRUE, na.action =na.omit)
# 
# plot(model_NN) #Visualize the relationship between the number of layers, decay and accuracy
# NN_prediction<-predict(model_NN, test_nn) #Predict classification 
# print(sprintf("Accuracy of NN for k = 20:    %.3f", Accuracy(y_true=test_nn$status, y_pred=NN_prediction)))
# print(sprintf("Precision of NN for k = 20:   %.3f", Precision(y_true=test_nn$status, y_pred=NN_prediction)))
# print(sprintf("F1 Score of NN for k = 20:    %.3f", F1_Score(NN_prediction, test_nn$status)))
# print(sprintf("Sensitivity of NN for k = 20: %.3f", Sensitivity(y_true=test_nn$status, y_pred=NN_prediction)))
# print(sprintf("Specificity of NN for k = 20: %.3f", Specificity(y_true=NN_prediction, y_pred=test_nn$status)))

### removing topics from this part
topics <- cbind(topics,df[,14:33])
df[,14:33] <- NULL


##### Topic Modelling with Sensitive Study ###############

#### With k = 10 ##########
##### fitting an LDA Model #####

ten_model <- FitLdaModel(dtm = dtm_3_3, 
                         k = 10, 
                         iterations = 500, # recommend a larger value, 500 or more
                         burnin = 180,
                         optimize_alpha = TRUE,
                         calc_likelihood = TRUE,
                         calc_coherence = TRUE,
                         calc_r2 = TRUE,
                         alpha = 0.1, # this is the default value
                         beta = 0.05, # this is the default value
                         cpus = 21) # Note, this is for a big machine  

plot(ten_model$log_likelihood, type = "l")


ten_model$top_terms <- GetTopTerms(phi = ten_model$phi, M = 8)
head(ten_model$top_terms) 

# probabilistic coherence, a measure of topic quality
# this measure can be used with any topic model, not just probabilistic ones
ten_model$coherence <- CalcProbCoherence(phi = ten_model$phi, dtm = dtm_3_3, M = 5)

# Get the prevalence of each topic
# You can make this discrete by applying a threshold, say 0.05, for
# topics in/out of docuemnts. 
ten_model$prevalence <- colSums(ten_model$theta) / sum(ten_model$theta) * 100


# textmineR has a naive topic labeling tool based on probable bigrams
ten_model$labels <- LabelTopics(assignments = ten_model$theta > 0.20, 
                                dtm = dtm_3_3,
                                M = 2)

head(ten_model$labels)


# put them together, with coherence into a summary table
ten_model$summary <- data.frame(topic = rownames(ten_model$phi),
                                label = ten_model$labels,
                                coherence = round(ten_model$coherence, 2),
                                prevalence = round(ten_model$prevalence,1),
                                top_terms = apply(ten_model$top_terms, 2, function(x){
                                  paste(x, collapse = ", ")
                                }),
                                stringsAsFactors = FALSE)

ten_model$summary %>%
  arrange(-prevalence)


write.csv(ten_model$top_terms, file = "3.3_10_Topics_kiva.csv")

#### Adding topics to data frame #####

df = cbind(df, round(ten_model$theta, digits=2))
head(df, n=10)

####################################################################
########## Model Building for k = 10 ###################

# Splitting the Data into train and test

set.seed(123) # Set the seed to make it reproducible

train.index <- createDataPartition(df$status, p = .8, list = FALSE)
train <- df[ train.index,]
test  <- df[-train.index,]

actual <- test$status
formula <- status ~ .
positive <- "yes"

# Random Forest with great results
set.seed(123)
model_forest <- randomForest(status~ ., data=train, 
                             importance=TRUE,proximity=TRUE,
                             cutoff = c(0.5, 0.5),type="classification", na.action=na.exclude)
print(model_forest)   
plot(model_forest)
importance(model_forest)

varImpPlot(model_forest)#Shows the decrease in accuracy with the removal of a variables so non Annual removal the accuray can go down by 30% for example

predicted <- predict(model_forest, test, type="class") 

print(sprintf("Accuracy of Random Forest for k = 10:    %.3f", Accuracy(y_true=actual, y_pred=predicted)))
print(sprintf("AUC of Random Forest for k = 10:     %.3f", AUC(y_pred=predicted, y_true=actual)))
print(sprintf("Precision of Random Forest for k = 10:   %.3f", Precision(y_true=actual, y_pred=predicted)))
print(sprintf("Recall of Random Forest for k = 10:      %.3f", Recall(y_true=actual, y_pred=predicted)))
print(sprintf("F1 Score of Random Forest for k = 10:    %.3f", F1_Score(predicted, actual)))
print(sprintf("Sensitivity of Random Forest for k = 10: %.3f", Sensitivity(y_true=actual, y_pred=predicted)))
print(sprintf("Specificity of Random Forest for k = 10: %.3f", Specificity(y_true=predicted, y_pred=actual)))

# Naive Bayes...Accuracy of 0.73 not very good
set.seed(123)

nb_fit = naiveBayes(status ~ ., data=train)
nb_fit

NB_pred = predict(nb_fit, test, type="class") 
print(sprintf("Accuracy of NB for k = 10:    %.3f", Accuracy(y_true=actual, y_pred=NB_pred)))
print(sprintf("Precision of NB for k = 10:   %.3f", Precision(y_true=actual, y_pred=NB_pred)))
print(sprintf("F1 Score of NB for k = 10:    %.3f", F1_Score(NB_pred, actual)))
print(sprintf("Sensitivityof NB for k = 10: %.3f", Sensitivity(y_true=actual, y_pred=NB_pred)))
print(sprintf("Specificity of NB for k = 10: %.3f", Specificity(y_true=NB_pred, y_pred=actual)))

# Decision Tree
set.seed(123)

tree <- rpart(status ~ ., method="class", data=train)
tree_pred = predict(tree, test, type="class") 
print(sprintf("Accuracy of DT for k = 10:    %.3f", Accuracy(y_true=actual, y_pred=tree_pred)))
print(sprintf("Precision of DT for k = 10:   %.3f", Precision(y_true=actual, y_pred=tree_pred)))
print(sprintf("F1 Score of DT for k = 10:    %.3f", F1_Score(tree_pred, actual)))
print(sprintf("Sensitivity of DT for k = 10: %.3f", Sensitivity(y_true=actual, y_pred=tree_pred)))
print(sprintf("Specificity of DT for k = 10: %.3f", Specificity(y_true=tree_pred, y_pred=actual)))

#SVM...this is the second best
set.seed(123)
svm_fit <- svm(status ~ ., data = train)
summary(svm_fit)

svm_pred <- predict(svm_fit, test) 

print(sprintf("Accuracy of SVM for k = 10:    %.3f", Accuracy(y_true=actual, y_pred=svm_pred)))
print(sprintf("Precision of SVM for k = 10:   %.3f", Precision(y_true=actual, y_pred=svm_pred)))
print(sprintf("F1 Score of SVM for k = 10:    %.3f", F1_Score(svm_pred, actual)))
print(sprintf("Sensitivity of SVM for k = 10: %.3f", Sensitivity(y_true=actual, y_pred=svm_pred)))
print(sprintf("Specificity of SVM for k = 10: %.3f", Specificity(y_true=svm_pred, y_pred=actual)))


#kknn....89%
set.seed(123)

kknn_fit = train.kknn(status ~ ., train, kmax=7)
summary(kknn_fit)

kknn_pred = predict(kknn_fit, test)

print(sprintf("Accuracy of KNN for k = 10:    %.3f", Accuracy(y_true=actual, y_pred=kknn_pred)))
print(sprintf("Precision of KNN for k = 10:   %.3f", Precision(y_true=actual, y_pred=kknn_pred)))
print(sprintf("F1 Score of KNN for k = 10:    %.3f", F1_Score(kknn_pred, actual)))
print(sprintf("Sensitivity of KNN for k = 10: %.3f", Sensitivity(y_true=actual, y_pred=kknn_pred)))
print(sprintf("Specificity of KNN for k = 10: %.3f", Specificity(y_true=kknn_pred, y_pred=actual)))

# #NN...91%
# install.packages("nnet")
# library(nnet)
# set.seed(123)
# 
# train_nn <- (train)
# test_nn <- test
# str(train_nn)
# 
# train_nn$status <- as.numeric(train_nn$status)
# train_nn$sector <- as.numeric(train_nn$sector)
# train_nn$country <- as.numeric(train_nn$country)
# train_nn$gender <- as.numeric(train_nn$gender)
# train_nn$countrygender <- as.numeric(as.factor(as.character(train_nn$countrygender)))
# train_nn$CountrySector <- as.numeric(as.factor(as.character(train_nn$CountrySector)))
# train_nn$nonpayment <- as.numeric (train_nn$nonpayment)
# 
# test_nn$status <- as.numeric(test_nn$status)
# test_nn$sector <- as.numeric(test_nn$sector)
# test_nn$country <- as.numeric(test_nn$country)
# test_nn$gender <- as.numeric(test_nn$gender)
# test_nn$countrygender <- as.numeric(as.factor(as.character(test_nn$countrygender)))
# test_nn$CountrySector <- as.numeric(as.factor(as.character(test_nn$CountrySector)))
# test_nn$nonpayment <- as.numeric (test_nn$nonpayment)
# str(train)
# ##start Neural Network analysis
# 
# my.grid <- expand.grid( .size = c(1,2,4),.decay = c(0.25,1,2)) # Tuning grid for Neural Net
# 
# model_NN <- train(status~ ., data = train_nn, method = "nnet", tuneGrid = my.grid, trace = TRUE, na.action =na.omit)
# 
# plot(model_NN) #Visualize the relationship between the number of layers, decay and accuracy
# NN_prediction<-predict(model_NN, test_nn) #Predict classification 
# print(sprintf("Accuracy of NN for k = 10:    %.3f", Accuracy(y_true=test_nn$status, y_pred=NN_prediction)))
# print(sprintf("Precision of NN for k = 10:   %.3f", Precision(y_true=test_nn$status, y_pred=NN_prediction)))
# print(sprintf("F1 Score of NN for k = 10:    %.3f", F1_Score(NN_prediction, test_nn$status)))
# print(sprintf("Sensitivity of NN for k = 10: %.3f", Sensitivity(y_true=test_nn$status, y_pred=NN_prediction)))
# print(sprintf("Specificity of NN for k = 10: %.3f", Specificity(y_true=NN_prediction, y_pred=test_nn$status)))

### removing topics from this part
topics <- cbind(topics, df[,14:23])
### removing topics from this part
df[,14:23] <- NULL

##### Topic Modelling with Sensitive Study ###############

#### With k = 15 ##########

##### fitting an LDA Model #####

fifteen_model <- FitLdaModel(dtm = dtm_3_3, 
                             k = 15, 
                             iterations = 500, # recommend a larger value, 500 or more
                             burnin = 180,
                             optimize_alpha = TRUE,
                             calc_likelihood = TRUE,
                             calc_coherence = TRUE,
                             calc_r2 = TRUE,
                             alpha = 0.1, # this is the default value
                             beta = 0.05, # this is the default value
                             cpus = 21) # Note, this is for a big machine  

plot(fifteen_model$log_likelihood, type = "l")


fifteen_model$top_terms <- GetTopTerms(phi = fifteen_model$phi, M = 8)
head(fifteen_model$top_terms) 

# probabilistic coherence, a measure of topic quality
# this measure can be used with any topic model, not just probabilistic ones
fifteen_model$coherence <- CalcProbCoherence(phi = fifteen_model$phi, dtm = dtm_3_3, M = 5)

# Get the prevalence of each topic
# You can make this discrete by applying a threshold, say 0.05, for
# topics in/out of docuemnts. 
fifteen_model$prevalence <- colSums(fifteen_model$theta) / sum(fifteen_model$theta) * 100


# textmineR has a naive topic labeling tool based on probable bigrams
fifteen_model$labels <- LabelTopics(assignments = fifteen_model$theta > 0.20, 
                                    dtm = dtm_3_3,
                                    M = 2)

head(fifteen_model$labels)


# put them together, with coherence into a summary table
fifteen_model$summary <- data.frame(topic = rownames(fifteen_model$phi),
                                    label = fifteen_model$labels,
                                    coherence = round(fifteen_model$coherence, 2),
                                    prevalence = round(fifteen_model$prevalence,1),
                                    top_terms = apply(fifteen_model$top_terms, 2, function(x){
                                      paste(x, collapse = ", ")
                                    }),
                                    stringsAsFactors = FALSE)

fifteen_model$summary %>%
  arrange(-prevalence)


write.csv(fifteen_model$top_terms, file = "3.3_15_Topics_kiva.csv")

#### Adding topics to data frame #####

df = cbind(df, round(fifteen_model$theta, digits=2))
head(df, n=10)

####################################################################
########## Model Building for k = 15 ###################

# Splitting the Data into train and test

set.seed(123) # Set the seed to make it reproducible

train.index <- createDataPartition(df$status, p = .8, list = FALSE)
train <- df[ train.index,]
test  <- df[-train.index,]

actual <- test$status
formula <- status ~ .
positive <- "yes"

# Random Forest with great results
set.seed(123)
model_forest <- randomForest(status~ ., data=train, 
                             importance=TRUE,proximity=TRUE,
                             cutoff = c(0.5, 0.5),type="classification", na.action=na.exclude)
print(model_forest)   
plot(model_forest)
importance(model_forest)

varImpPlot(model_forest)#Shows the decrease in accuracy with the removal of a variables so non Annual removal the accuray can go down by 30% for example

predicted <- predict(model_forest, test, type="class") 

print(sprintf("Accuracy of Random Forest for k = 15:    %.3f", Accuracy(y_true=actual, y_pred=predicted)))
print(sprintf("AUC of Random Forest for k = 15:     %.3f", AUC(y_pred=predicted, y_true=actual)))
print(sprintf("Precision of Random Forest for k = 15:   %.3f", Precision(y_true=actual, y_pred=predicted)))
print(sprintf("Recall of Random Forest for k = 15:      %.3f", Recall(y_true=actual, y_pred=predicted)))
print(sprintf("F1 Score of Random Forest for k = 15:    %.3f", F1_Score(predicted, actual)))
print(sprintf("Sensitivity of Random Forest for k = 15: %.3f", Sensitivity(y_true=actual, y_pred=predicted)))
print(sprintf("Specificity of Random Forest for k = 15: %.3f", Specificity(y_true=predicted, y_pred=actual)))

# Naive Bayes...Accuracy of 0.73 not very good
set.seed(123)

nb_fit = naiveBayes(status ~ ., data=train)
nb_fit

NB_pred = predict(nb_fit, test, type="class") 
print(sprintf("Accuracy of NB for k = 15:    %.3f", Accuracy(y_true=actual, y_pred=NB_pred)))
print(sprintf("Precision of NB for k = 15:   %.3f", Precision(y_true=actual, y_pred=NB_pred)))
print(sprintf("F1 Score of NB for k = 15:    %.3f", F1_Score(NB_pred, actual)))
print(sprintf("Sensitivityof NB for k = 15: %.3f", Sensitivity(y_true=actual, y_pred=NB_pred)))
print(sprintf("Specificity of NB for k = 15: %.3f", Specificity(y_true=NB_pred, y_pred=actual)))
# Decision Tree

set.seed(123)

tree <- rpart(status ~ ., method="class", data=train)
tree_pred = predict(tree, test, type="class") 
print(sprintf("Accuracy of DT for k = 15:    %.3f", Accuracy(y_true=actual, y_pred=tree_pred)))
print(sprintf("Precision of DT for k = 15:   %.3f", Precision(y_true=actual, y_pred=tree_pred)))
print(sprintf("F1 Score of DT for k = 15:    %.3f", F1_Score(tree_pred, actual)))
print(sprintf("Sensitivity of DT for k = 15: %.3f", Sensitivity(y_true=actual, y_pred=tree_pred)))
print(sprintf("Specificity of DT for k = 15: %.3f", Specificity(y_true=tree_pred, y_pred=actual)))

#SVM...this is the second best

svm_fit <- svm(status ~ ., data = train)
summary(svm_fit)

svm_pred <- predict(svm_fit, test) 

print(sprintf("Accuracy of SVM for k = 15:    %.3f", Accuracy(y_true=actual, y_pred=svm_pred)))
print(sprintf("Precision of SVM for k = 15:   %.3f", Precision(y_true=actual, y_pred=svm_pred)))
print(sprintf("F1 Score of SVM for k = 15:    %.3f", F1_Score(svm_pred, actual)))
print(sprintf("Sensitivity of SVM for k = 15: %.3f", Sensitivity(y_true=actual, y_pred=svm_pred)))
print(sprintf("Specificity of SVM for k = 15: %.3f", Specificity(y_true=svm_pred, y_pred=actual)))


#kknn....89%
set.seed(123)
kknn_fit = train.kknn(status ~ ., train, kmax=7)
summary(kknn_fit)

kknn_pred = predict(kknn_fit, test)

print(sprintf("Accuracy of KNN for k = 15:    %.3f", Accuracy(y_true=actual, y_pred=kknn_pred)))
print(sprintf("Precision of KNN for k = 15:   %.3f", Precision(y_true=actual, y_pred=kknn_pred)))
print(sprintf("F1 Score of KNN for k = 15:    %.3f", F1_Score(kknn_pred, actual)))
print(sprintf("Sensitivity of KNN for k = 15: %.3f", Sensitivity(y_true=actual, y_pred=kknn_pred)))
print(sprintf("Specificity of KNN for k = 15: %.3f", Specificity(y_true=kknn_pred, y_pred=actual)))

# #NN...91%
# install.packages("nnet")
# library(nnet)
# set.seed(123)
# 
# train_nn <- (train)
# test_nn <- test
# str(train_nn)
# 
# train_nn$status <- as.numeric(train_nn$status)
# train_nn$sector <- as.numeric(train_nn$sector)
# train_nn$country <- as.numeric(train_nn$country)
# train_nn$gender <- as.numeric(train_nn$gender)
# train_nn$countrygender <- as.numeric(as.factor(as.character(train_nn$countrygender)))
# train_nn$CountrySector <- as.numeric(as.factor(as.character(train_nn$CountrySector)))
# train_nn$nonpayment <- as.numeric (train_nn$nonpayment)
# 
# test_nn$status <- as.numeric(test_nn$status)
# test_nn$sector <- as.numeric(test_nn$sector)
# test_nn$country <- as.numeric(test_nn$country)
# test_nn$gender <- as.numeric(test_nn$gender)
# test_nn$countrygender <- as.numeric(as.factor(as.character(test_nn$countrygender)))
# test_nn$CountrySector <- as.numeric(as.factor(as.character(test_nn$CountrySector)))
# test_nn$nonpayment <- as.numeric (test_nn$nonpayment)
# str(train)
# ##start Neural Network analysis
# 
# my.grid <- expand.grid( .size = c(1,2,4),.decay = c(0.25,1,2)) # Tuning grid for Neural Net
# 
# model_NN <- train(status~ ., data = train_nn, method = "nnet", tuneGrid = my.grid, trace = TRUE, na.action =na.omit)
# 
# plot(model_NN) #Visualize the relationship between the number of layers, decay and accuracy
# NN_prediction<-predict(model_NN, test_nn) #Predict classification 
# print(sprintf("Accuracy of NN for k = 15:    %.3f", Accuracy(y_true=test_nn$status, y_pred=NN_prediction)))
# print(sprintf("Precision of NN for k = 15:   %.3f", Precision(y_true=test_nn$status, y_pred=NN_prediction)))
# print(sprintf("F1 Score of NN for k = 15:    %.3f", F1_Score(NN_prediction, test_nn$status)))
# print(sprintf("Sensitivity of NN for k = 15: %.3f", Sensitivity(y_true=test_nn$status, y_pred=NN_prediction)))
# print(sprintf("Specificity of NN for k = 15: %.3f", Specificity(y_true=NN_prediction, y_pred=test_nn$status)))

### removing topics from this part
topics <- cbind(topics, df[,14:28])
### removing topics from this part
df[,14:28] <- NULL


##### Topic Modelling with Sensitive Study ###############

#### With k = 25 ##########

##### fitting an LDA Model #####

twentyfive_model <- FitLdaModel(dtm = dtm_3_3, 
                                k = 25, 
                                iterations = 500, # recommend a larger value, 500 or more
                                burnin = 180,
                                optimize_alpha = TRUE,
                                calc_likelihood = TRUE,
                                calc_coherence = TRUE,
                                calc_r2 = TRUE,
                                alpha = 0.1, # this is the default value
                                beta = 0.05, # this is the default value
                                cpus = 21) # Note, this is for a big machine  

plot(twentyfive_model$log_likelihood, type = "l")


twentyfive_model$top_terms <- GetTopTerms(phi = twentyfive_model$phi, M = 8)
head(twentyfive_model$top_terms) 

# probabilistic coherence, a measure of topic quality
# this measure can be used with any topic model, not just probabilistic ones
twentyfive_model$coherence <- CalcProbCoherence(phi = twentyfive_model$phi, dtm = dtm_3_3, M = 5)

# Get the prevalence of each topic
# You can make this discrete by applying a threshold, say 0.05, for
# topics in/out of docuemnts. 
twentyfive_model$prevalence <- colSums(twentyfive_model$theta) / sum(twentyfive_model$theta) * 100


# textmineR has a naive topic labeling tool based on probable bigrams
twentyfive_model$labels <- LabelTopics(assignments = twentyfive_model$theta > 0.20, 
                                       dtm = dtm_3_3,
                                       M = 2)

head(twentyfive_model$labels)


# put them together, with coherence into a summary table
twentyfive_model$summary <- data.frame(topic = rownames(twentyfive_model$phi),
                                       label = twentyfive_model$labels,
                                       coherence = round(twentyfive_model$coherence, 2),
                                       prevalence = round(twentyfive_model$prevalence,1),
                                       top_terms = apply(twentyfive_model$top_terms, 2, function(x){
                                         paste(x, collapse = ", ")
                                       }),
                                       stringsAsFactors = FALSE)

twentyfive_model$summary %>%
  arrange(-prevalence)


write.csv(twentyfive_model$top_terms, file = "3.3_25_Topics_kiva.csv")

#### Adding topics to data frame #####

df = cbind(df, round(twentyfive_model$theta, digits=2))
head(df, n=10)

####################################################################
########## Model Building for k = 25 ###################

# Splitting the Data into train and test

set.seed(123) # Set the seed to make it reproducible

train.index <- createDataPartition(df$status, p = .8, list = FALSE)
train <- df[ train.index,]
test  <- df[-train.index,]

actual <- test$status
formula <- status ~ .
positive <- "yes"

# Random Forest with great results
set.seed(123)
model_forest <- randomForest(status~ ., data=train, 
                             importance=TRUE,proximity=TRUE,
                             cutoff = c(0.5, 0.5),type="classification", na.action=na.exclude)
print(model_forest)   
plot(model_forest)
importance(model_forest)

varImpPlot(model_forest)#Shows the decrease in accuracy with the removal of a variables so non Annual removal the accuray can go down by 30% for example

predicted <- predict(model_forest, test, type="class") 

print(sprintf("Accuracy of Random Forest for k = 25:    %.3f", Accuracy(y_true=actual, y_pred=predicted)))
print(sprintf("AUC of Random Forest for k = 25:     %.3f", AUC(y_pred=predicted, y_true=actual)))
print(sprintf("Precision of Random Forest for k = 25:   %.3f", Precision(y_true=actual, y_pred=predicted)))
print(sprintf("Recall of Random Forest for k = 25:      %.3f", Recall(y_true=actual, y_pred=predicted)))
print(sprintf("F1 Score of Random Forest for k = 25:    %.3f", F1_Score(predicted, actual)))
print(sprintf("Sensitivity of Random Forest for k = 25: %.3f", Sensitivity(y_true=actual, y_pred=predicted)))
print(sprintf("Specificity of Random Forest for k = 25: %.3f", Specificity(y_true=predicted, y_pred=actual)))

# Naive Bayes...Accuracy of 0.73 not very good
set.seed(123)

nb_fit = naiveBayes(status ~ ., data=train)
nb_fit

NB_pred = predict(nb_fit, test, type="class") 
print(sprintf("Accuracy of NB for k = 25:    %.3f", Accuracy(y_true=actual, y_pred=NB_pred)))
print(sprintf("Precision of NB for k = 25:   %.3f", Precision(y_true=actual, y_pred=NB_pred)))
print(sprintf("F1 Score of NB for k = 25:    %.3f", F1_Score(NB_pred, actual)))
print(sprintf("Sensitivityof NB for k = 25: %.3f", Sensitivity(y_true=actual, y_pred=NB_pred)))
print(sprintf("Specificity of NB for k = 25: %.3f", Specificity(y_true=NB_pred, y_pred=actual)))
# Decision Tree
set.seed(123)

tree <- rpart(status ~ ., method="class", data=train)
tree_pred = predict(tree, test, type="class") 
print(sprintf("Accuracy of DT for k = 25:    %.3f", Accuracy(y_true=actual, y_pred=tree_pred)))
print(sprintf("Precision of DT for k = 25:   %.3f", Precision(y_true=actual, y_pred=tree_pred)))
print(sprintf("F1 Score of DT for k = 25:    %.3f", F1_Score(tree_pred, actual)))
print(sprintf("Sensitivity of DT for k = 25: %.3f", Sensitivity(y_true=actual, y_pred=tree_pred)))
print(sprintf("Specificity of DT for k = 25: %.3f", Specificity(y_true=tree_pred, y_pred=actual)))

#SVM...this is the second best
set.seed(123)
svm_fit <- svm(status ~ ., data = train)
summary(svm_fit)

svm_pred <- predict(svm_fit, test) 

print(sprintf("Accuracy of SVM for k = 25:    %.3f", Accuracy(y_true=actual, y_pred=svm_pred)))
print(sprintf("Precision of SVM for k = 25:   %.3f", Precision(y_true=actual, y_pred=svm_pred)))
print(sprintf("F1 Score of SVM for k = 25:    %.3f", F1_Score(svm_pred, actual)))
print(sprintf("Sensitivity of SVM for k = 25: %.3f", Sensitivity(y_true=actual, y_pred=svm_pred)))
print(sprintf("Specificity of SVM for k = 25: %.3f", Specificity(y_true=svm_pred, y_pred=actual)))


#kknn....89%
set.seed(123)
kknn_fit = train.kknn(status ~ ., train, kmax=7)
summary(kknn_fit)

kknn_pred = predict(kknn_fit, test)

print(sprintf("Accuracy of KNN for k = 25:    %.3f", Accuracy(y_true=actual, y_pred=kknn_pred)))
print(sprintf("Precision of KNN for k = 25:   %.3f", Precision(y_true=actual, y_pred=kknn_pred)))
print(sprintf("F1 Score of KNN for k = 25:    %.3f", F1_Score(kknn_pred, actual)))
print(sprintf("Sensitivity of KNN for k = 25: %.3f", Sensitivity(y_true=actual, y_pred=kknn_pred)))
print(sprintf("Specificity of KNN for k = 25: %.3f", Specificity(y_true=kknn_pred, y_pred=actual)))

# #NN...91%
# install.packages("nnet")
# library(nnet)
# set.seed(123)
# 
# train_nn <- (train)
# test_nn <- test
# str(train_nn)
# 
# train_nn$status <- as.numeric(train_nn$status)
# train_nn$sector <- as.numeric(train_nn$sector)
# train_nn$country <- as.numeric(train_nn$country)
# train_nn$gender <- as.numeric(train_nn$gender)
# train_nn$countrygender <- as.numeric(as.factor(as.character(train_nn$countrygender)))
# train_nn$CountrySector <- as.numeric(as.factor(as.character(train_nn$CountrySector)))
# train_nn$nonpayment <- as.numeric (train_nn$nonpayment)
# 
# test_nn$status <- as.numeric(test_nn$status)
# test_nn$sector <- as.numeric(test_nn$sector)
# test_nn$country <- as.numeric(test_nn$country)
# test_nn$gender <- as.numeric(test_nn$gender)
# test_nn$countrygender <- as.numeric(as.factor(as.character(test_nn$countrygender)))
# test_nn$CountrySector <- as.numeric(as.factor(as.character(test_nn$CountrySector)))
# test_nn$nonpayment <- as.numeric (test_nn$nonpayment)
# str(train)
# ##start Neural Network analysis
# 
# my.grid <- expand.grid( .size = c(1,2,4),.decay = c(0.25,1,2)) # Tuning grid for Neural Net
# 
# model_NN <- train(status~ ., data = train_nn, method = "nnet", tuneGrid = my.grid, trace = TRUE, na.action =na.omit)
# 
# plot(model_NN) #Visualize the relationship between the number of layers, decay and accuracy
# NN_prediction<-predict(model_NN, test_nn) #Predict classification 
# print(sprintf("Accuracy of NN for k = 25:    %.3f", Accuracy(y_true=test_nn$status, y_pred=NN_prediction)))
# print(sprintf("Precision of NN for k = 25:   %.3f", Precision(y_true=test_nn$status, y_pred=NN_prediction)))
# print(sprintf("F1 Score of NN for k = 25:    %.3f", F1_Score(NN_prediction, test_nn$status)))
# print(sprintf("Sensitivity of NN for k = 25: %.3f", Sensitivity(y_true=test_nn$status, y_pred=NN_prediction)))
# print(sprintf("Specificity of NN for k = 25: %.3f", Specificity(y_true=NN_prediction, y_pred=test_nn$status)))

### removing topics from this part
topics <- cbind(topics,df[,14:38])
### removing topics from this part
df[,14:38] <- NULL

##### Topic Modelling with Sensitive Study ###############

#### With k = 30 ##########

##### fitting an LDA Model #####

thirty_model <- FitLdaModel(dtm = dtm_3_3, 
                            k = 30, 
                            iterations = 500, # recommend a larger value, 500 or more
                            burnin = 180,
                            optimize_alpha = TRUE,
                            calc_likelihood = TRUE,
                            calc_coherence = TRUE,
                            calc_r2 = TRUE,
                            alpha = 0.1, # this is the default value
                            beta = 0.05, # this is the default value
                            cpus = 21) # Note, this is for a big machine  

plot(thirty_model$log_likelihood, type = "l")


thirty_model$top_terms <- GetTopTerms(phi = thirty_model$phi, M = 8)
head(thirty_model$top_terms) 

# probabilistic coherence, a measure of topic quality
# this measure can be used with any topic model, not just probabilistic ones
thirty_model$coherence <- CalcProbCoherence(phi = thirty_model$phi, dtm = dtm_3_3, M = 5)

# Get the prevalence of each topic
# You can make this discrete by applying a threshold, say 0.05, for
# topics in/out of docuemnts. 
thirty_model$prevalence <- colSums(thirty_model$theta) / sum(thirty_model$theta) * 100


# textmineR has a naive topic labeling tool based on probable bigrams
thirty_model$labels <- LabelTopics(assignments = thirty_model$theta > 0.20, 
                                   dtm = dtm_3_3,
                                   M = 2)

head(thirty_model$labels)


# put them together, with coherence into a summary table
thirty_model$summary <- data.frame(topic = rownames(thirty_model$phi),
                                   label = thirty_model$labels,
                                   coherence = round(thirty_model$coherence, 2),
                                   prevalence = round(thirty_model$prevalence,1),
                                   top_terms = apply(thirty_model$top_terms, 2, function(x){
                                     paste(x, collapse = ", ")
                                   }),
                                   stringsAsFactors = FALSE)

thirty_model$summary %>%
  arrange(-prevalence)


write.csv(thirty_model$top_terms, file = "3.3_30_Topics_kiva.csv")

#### Adding topics to data frame #####

df = cbind(df, round(thirty_model$theta, digits=2))
head(df, n=10)

####################################################################
########## Model Building for k = 30 ###################

# Splitting the Data into train and test

set.seed(123) # Set the seed to make it reproducible

train.index <- createDataPartition(df$status, p = .8, list = FALSE)
train <- df[ train.index,]
test  <- df[-train.index,]

actual <- test$status
formula <- status ~ .
positive <- "yes"

# Random Forest with great results
set.seed(123)
model_forest <- randomForest(status~ ., data=train, 
                             importance=TRUE,proximity=TRUE,
                             cutoff = c(0.5, 0.5),type="classification", na.action=na.exclude)
print(model_forest)   
plot(model_forest)
importance(model_forest)

varImpPlot(model_forest)#Shows the decrease in accuracy with the removal of a variables so non Annual removal the accuray can go down by 30% for example

predicted <- predict(model_forest, test, type="class") 

print(sprintf("Accuracy of Random Forest for k = 30:    %.3f", Accuracy(y_true=actual, y_pred=predicted)))
print(sprintf("AUC of Random Forest for k = 30:     %.3f", AUC(y_pred=predicted, y_true=actual)))
print(sprintf("Precision of Random Forest for k = 30:   %.3f", Precision(y_true=actual, y_pred=predicted)))
print(sprintf("Recall of Random Forest for k = 30:      %.3f", Recall(y_true=actual, y_pred=predicted)))
print(sprintf("F1 Score of Random Forest for k = 30:    %.3f", F1_Score(predicted, actual)))
print(sprintf("Sensitivity of Random Forest for k = 30: %.3f", Sensitivity(y_true=actual, y_pred=predicted)))
print(sprintf("Specificity of Random Forest for k = 30: %.3f", Specificity(y_true=predicted, y_pred=actual)))

# Naive Bayes...Accuracy of 0.73 not very good
set.seed(123)

nb_fit = naiveBayes(status ~ ., data=train)
nb_fit

NB_pred = predict(nb_fit, test, type="class") 
print(sprintf("Accuracy of NB for k = 30:    %.3f", Accuracy(y_true=actual, y_pred=NB_pred)))
print(sprintf("Precision of NB for k = 30:   %.3f", Precision(y_true=actual, y_pred=NB_pred)))
print(sprintf("F1 Score of NB for k = 30:    %.3f", F1_Score(NB_pred, actual)))
print(sprintf("Sensitivityof NB for k = 30: %.3f", Sensitivity(y_true=actual, y_pred=NB_pred)))
print(sprintf("Specificity of NB for k = 30: %.3f", Specificity(y_true=NB_pred, y_pred=actual)))

# Decision Tree
set.seed(123)

tree <- rpart(status ~ ., method="class", data=train)
tree_pred = predict(tree, test, type="class") 
print(sprintf("Accuracy of DT for k = 30:    %.3f", Accuracy(y_true=actual, y_pred=tree_pred)))
print(sprintf("Precision of DT for k = 30:   %.3f", Precision(y_true=actual, y_pred=tree_pred)))
print(sprintf("F1 Score of DT for k = 30:    %.3f", F1_Score(tree_pred, actual)))
print(sprintf("Sensitivity of DT for k = 30: %.3f", Sensitivity(y_true=actual, y_pred=tree_pred)))
print(sprintf("Specificity of DT for k = 30: %.3f", Specificity(y_true=tree_pred, y_pred=actual)))

#SVM...this is the second best
set.seed(123)
svm_fit <- svm(status ~ ., data = train)
summary(svm_fit)

svm_pred <- predict(svm_fit, test) 

print(sprintf("Accuracy of SVM for k = 30:    %.3f", Accuracy(y_true=actual, y_pred=svm_pred)))
print(sprintf("Precision of SVM for k = 30:   %.3f", Precision(y_true=actual, y_pred=svm_pred)))
print(sprintf("F1 Score of SVM for k = 30:    %.3f", F1_Score(svm_pred, actual)))
print(sprintf("Sensitivity of SVM for k = 30: %.3f", Sensitivity(y_true=actual, y_pred=svm_pred)))
print(sprintf("Specificity of SVM for k = 30: %.3f", Specificity(y_true=svm_pred, y_pred=actual)))


#kknn....89%
set.seed(123)

kknn_fit = train.kknn(status ~ ., train, kmax=7)
summary(kknn_fit)

kknn_pred = predict(kknn_fit, test)

print(sprintf("Accuracy of KNN for k = 30:    %.3f", Accuracy(y_true=actual, y_pred=kknn_pred)))
print(sprintf("Precision of KNN for k = 30:   %.3f", Precision(y_true=actual, y_pred=kknn_pred)))
print(sprintf("F1 Score of KNN for k = 30:    %.3f", F1_Score(kknn_pred, actual)))
print(sprintf("Sensitivity of KNN for k = 30: %.3f", Sensitivity(y_true=actual, y_pred=kknn_pred)))
print(sprintf("Specificity of KNN for k = 30: %.3f", Specificity(y_true=kknn_pred, y_pred=actual)))

# #NN...91%
# install.packages("nnet")
# library(nnet)
# set.seed(123)
# 
# train_nn <- (train)
# test_nn <- test
# str(train_nn)
# 
# train_nn$status <- as.numeric(train_nn$status)
# train_nn$sector <- as.numeric(train_nn$sector)
# train_nn$country <- as.numeric(train_nn$country)
# train_nn$gender <- as.numeric(train_nn$gender)
# train_nn$countrygender <- as.numeric(as.factor(as.character(train_nn$countrygender)))
# train_nn$CountrySector <- as.numeric(as.factor(as.character(train_nn$CountrySector)))
# train_nn$nonpayment <- as.numeric (train_nn$nonpayment)
# 
# test_nn$status <- as.numeric(test_nn$status)
# test_nn$sector <- as.numeric(test_nn$sector)
# test_nn$country <- as.numeric(test_nn$country)
# test_nn$gender <- as.numeric(test_nn$gender)
# test_nn$countrygender <- as.numeric(as.factor(as.character(test_nn$countrygender)))
# test_nn$CountrySector <- as.numeric(as.factor(as.character(test_nn$CountrySector)))
# test_nn$nonpayment <- as.numeric (test_nn$nonpayment)
# str(train)
# ##start Neural Network analysis
# 
# my.grid <- expand.grid( .size = c(1,2,4),.decay = c(0.25,1,2)) # Tuning grid for Neural Net
# 
# model_NN <- train(status~ ., data = train_nn, method = "nnet", tuneGrid = my.grid, trace = TRUE, na.action =na.omit)
# 
# plot(model_NN) #Visualize the relationship between the number of layers, decay and accuracy
# NN_prediction<-predict(model_NN, test_nn) #Predict classification 
# print(sprintf("Accuracy of NN for k = 30:    %.3f", Accuracy(y_true=test_nn$status, y_pred=NN_prediction)))
# print(sprintf("Precision of NN for k = 30:   %.3f", Precision(y_true=test_nn$status, y_pred=NN_prediction)))
# print(sprintf("F1 Score of NN for k = 30:    %.3f", F1_Score(NN_prediction, test_nn$status)))
# print(sprintf("Sensitivity of NN for k = 30: %.3f", Sensitivity(y_true=test_nn$status, y_pred=NN_prediction)))
# print(sprintf("Specificity of NN for k = 30: %.3f", Specificity(y_true=NN_prediction, y_pred=test_nn$status)))

### removing topics from this part
topics <- cbind(topics, df[,14:43])
### removing topics from this part
df[,14:43] <- NULL

################################################################################################################################################################################

df <- read.csv("D:\\kiva_data.csv", stringsAsFactors=FALSE)

########## Renaming and type setting of attributes within the dataset ###########

df = df %>%
  rename(story = en)
str(df)
df$id = 1:nrow(df)
df$status = as.factor(df$status)
df$sector = as.factor(df$sector)
df$country = as.factor(df$country)
df$gender = as.factor(df$gender)
df$nonpayment = as.factor(df$nonpayment)

########## Remove HTML Tags ###################
df = df %>% 
  mutate(story = gsub("<.*?>", "", story))

########## Retain distinct entries ################

dim(df)
df = distinct(df, story, .keep_all = TRUE)
dim(df)

############## Clustering the applicants based on all attributes but the story #######

cl <- df
cl$X <- NULL
cl$loan_nonpayment <- NULL
cl$id <- NULL
str(cl)
cl$status <- as.numeric(cl$status)
cl$sector <- as.numeric(cl$sector)
cl$story <- NULL
cl$country <- as.numeric(cl$country)
cl$gender <- as.numeric(cl$gender)
cl$countrygender <- as.numeric(as.factor(as.character(cl$countrygender)))
cl$CountrySector <- as.numeric(as.factor(as.character(cl$CountrySector)))
cl$loan_amount <- as.numeric (cl$loan_amount)
cl$nonpayment <- as.numeric (cl$nonpayment)

## Elbow plot
wss <- (nrow(cl)-1)*sum(apply(cl,2,var))

for (i in 1:30) wss[i] <- sum(kmeans(cl, 
                                     centers=i)$withinss)
plot(1:30, wss, type="b", xlab="Number of Clusters",
     ylab="Within groups sum of squares")

## Choose 4 clusters 

clust <- kmeans(cl, centers = 4, n = 4)

head(clust$cluster)
df <- cbind(df, cluster_name = clust$cluster)

#################### Sentiment Detection ######################

Sentiment <- sentiment_by(df$story)
summary(Sentiment$ave_sentiment)

qplot(Sentiment$ave_sentiment, geom = "histogram", binwidth = 0.1, main = "Review Sentiment Histogram")
head(sentiment)
df$ave_sentiment <- Sentiment$ave_sentiment
df$sd_sentiment <- Sentiment$sd
df$word_count <- Sentiment$word_count

rm(Sentiment, cl, clust)
########## Removing text that is not neccessary for topic identification or further analysis #############
dim(df)
df = distinct(df, story, .keep_all = TRUE)
dim(df)

idx = grep("Translated from Spanish", df$story)
df$story[idx[1]]
df$story = gsub("Translated from Spanish.*Kiva volunteer", " ", df$story, ignore.case = TRUE)
grep("Translated from Spanish", df$story)

grep("Mifex offers", df$story)
df$story = gsub("Mifex offers.*www.mifex.org", " ", df$story, ignore.case = TRUE)
grep("Mifex offers", df$story)

df$story = gsub("About KADET:.*within those communities", " ", df$story, ignore.case = TRUE)
df$story = gsub("Disclaimer: Due to recent events in.*making their loan.", " ", df$story, ignore.case = TRUE)

########## Topic Modelling ######################################
################### LDA #######################

####### Create Document Term Matrix ###########
# setting n-gram window to be exactly 1 -- unigrams are necessary for our analysis ########## Eliminating stopwords from our text based on english, french, spanish and SMART distionaries ##### converting all uppercase letters to lowercase ##### removing both puctuation and numbers from our text #### using Textstem library, we are lemmatizing words to their root forms instead of stemming #########

dtm_1_2 <- CreateDtm(doc_vec = df$story, # character vector of documents
                     doc_names = df$id, # document names
                     ngram_window = c(1,2), # minimum and maximum n-gram length
                     stopword_vec = c(tm::stopwords("english"), # stopwords from tm
                                      tm::stopwords("french"), # stopwords from tm
                                      tm::stopwords("spanish"), # stopwords from tm
                                      tm::stopwords("SMART")), # this is the default value
                     lower = TRUE, # lowercase - this is the default value
                     remove_punctuation = TRUE, # punctuation - this is the default
                     remove_numbers = TRUE, # numbers - this is the default
                     verbose = FALSE, # Turn off status bar for this demo,
                     stem_lemma_function = function(x) textstem::lemmatize_words(x),
                     cpus = 20) # default is all available cpus on the system


# Filter rare words
dim(dtm_1_2)
dtm_1_1 <- dtm_1_2[ , colSums(dtm_1_2 > 0) > 50 ]
dim(dtm_1_2)

max_num = nrow(dtm_1_2) * 0.3
max_num
dtm_1_2 <- dtm_1_2[ , colSums(dtm_1_2 > 0) <= max_num ]
dim(dtm_1_2)

# Capture how long each document is
df$doc_lengths = rowSums(dtm_1_2)

### identify frequent terms and highest 50 IDFs ###
tf_mat <- TermDocFreq(dtm = dtm_1_2)
head(tf_mat[ order(tf_mat$term_freq, decreasing = TRUE) , ], 50)
# write.csv(tf_mat, "term_frequency.csv")
head(tf_mat[ order(tf_mat$idf, decreasing = TRUE) , ], 50)

# look at the most frequent bigrams
tf_bigrams <- tf_mat[ stringr::str_detect(tf_mat$term, "_") , ]
head(tf_bigrams[ order(tf_bigrams$term_freq, decreasing = TRUE) , ], 50)
write.csv(head(tf_bigrams[ order(tf_bigrams$term_freq, decreasing = TRUE) , ], 50), "1.2_kiva_bi_grams.csv")

# look at the most frequent trigrams
tf_trigrams <- tf_mat[ stringr::str_detect(tf_mat$term, "_") , ]
head(tf_trigrams[ order(tf_trigrams$term_freq, decreasing = TRUE) , ], 50)
write.csv(head(tf_trigrams[ order(tf_trigrams$term_freq, decreasing = TRUE) , ], 50), "1.2_kiva_tri_grams.csv")


##### fitting an LDA Model #####

twenty_model <- FitLdaModel(dtm = dtm_1_2, 
                            k = 20, 
                            iterations = 500, # recommend a larger value, 500 or more
                            burnin = 180,
                            optimize_alpha = TRUE,
                            calc_likelihood = TRUE,
                            calc_coherence = TRUE,
                            calc_r2 = TRUE,
                            alpha = 0.1, # this is the default value
                            beta = 0.05, # this is the default value
                            cpus = 21) # Note, this is for a big machine  

plot(twenty_model$log_likelihood, type = "l")


twenty_model$top_terms <- GetTopTerms(phi = twenty_model$phi, M = 8)
head(twenty_model$top_terms) 

# probabilistic coherence, a measure of topic quality
# this measure can be used with any topic model, not just probabilistic ones
twenty_model$coherence <- CalcProbCoherence(phi = twenty_model$phi, dtm = dtm_1_2, M = 5)

# Get the prevalence of each topic
# You can make this discrete by applying a threshold, say 0.05, for
# topics in/out of docuemnts. 
twenty_model$prevalence <- colSums(twenty_model$theta) / sum(twenty_model$theta) * 100


# textmineR has a naive topic labeling tool based on probable bigrams
twenty_model$labels <- LabelTopics(assignments = twenty_model$theta > 0.20, 
                                   dtm = dtm_1_2,
                                   M = 2)

head(twenty_model$labels)


# put them together, with coherence into a summary table
twenty_model$summary <- data.frame(topic = rownames(twenty_model$phi),
                                   label = twenty_model$labels,
                                   coherence = round(twenty_model$coherence, 2),
                                   prevalence = round(twenty_model$prevalence,1),
                                   top_terms = apply(twenty_model$top_terms, 2, function(x){
                                     paste(x, collapse = ", ")
                                   }),
                                   stringsAsFactors = FALSE)

twenty_model$summary %>%
  arrange(-prevalence)


write.csv(twenty_model$top_terms, file = "1.1_20_Topics_kiva.csv")

#### Adding topics to data frame #####

df = cbind(df, round(twenty_model$theta, digits=2))
head(df, n=10)


##### Model Building for Classification ######
# Helper function to print the confusion matrix and other performance metrics of the models.
printPerformance = function(pred, actual, positive="Yes") {
  print(caret::confusionMatrix(data=pred, reference=actual, positive=positive, dnn=c("Predicted", "Actual")))
}

#### Data Type casting ####

str(df$status)
str(df)
df$id <- NULL
df <- as.data.frame(unclass(df))#converting all characters variables to factors
str(df)
df$story <- NULL

# Splitting the Data into train and test

set.seed(123) # Set the seed to make it reproducible

train.index <- createDataPartition(df$status, p = .8, list = FALSE)
train <- df[ train.index,]
test  <- df[-train.index,]

actual <- test$status
formula <- status ~ .
positive <- "yes"

# Random Forest with great results
set.seed(123)
model_forest <- randomForest(status~ ., data=train, 
                             importance=TRUE,proximity=TRUE,
                             cutoff = c(0.5, 0.5),type="classification", na.action=na.exclude)
print(model_forest)   
plot(model_forest)
importance(model_forest)

varImpPlot(model_forest)#Shows the decrease in accuracy with the removal of a variables so non Annual removal the accuray can go down by 30% for example

predicted <- predict(model_forest, test, type="class") 

print(sprintf("Accuracy of Random Forest for k = 20:    %.3f", Accuracy(y_true=actual, y_pred=predicted)))
print(sprintf("Precision of Random Forest for k = 20:   %.3f", Precision(y_true=actual, y_pred=predicted)))
print(sprintf("Recall of Random Forest for k = 20:      %.3f", Recall(y_true=actual, y_pred=predicted)))
print(sprintf("F1 Score of Random Forest for k = 20:    %.3f", F1_Score(predicted, actual)))
print(sprintf("Sensitivity of Random Forest for k = 20: %.3f", Sensitivity(y_true=actual, y_pred=predicted)))
print(sprintf("Specificity of Random Forest for k = 20: %.3f", Specificity(y_true=predicted, y_pred=actual)))

# Naive Bayes...Accuracy of 0.73 not very good
set.seed(123)

nb_fit = naiveBayes(status ~ ., data=train)
nb_fit

NB_pred = predict(nb_fit, test, type="class") 
print(sprintf("Accuracy of NB for k = 20:    %.3f", Accuracy(y_true=actual, y_pred=NB_pred)))
print(sprintf("Precision of NB for k = 20:   %.3f", Precision(y_true=actual, y_pred=NB_pred)))
print(sprintf("F1 Score of NB for k = 20:    %.3f", F1_Score(NB_pred, actual)))
print(sprintf("Sensitivityof NB for k = 20: %.3f", Sensitivity(y_true=actual, y_pred=NB_pred)))
print(sprintf("Specificity of NB for k = 20: %.3f", Specificity(y_true=NB_pred, y_pred=actual)))

# Decision Tree
set.seed(123)

tree <- rpart(status ~ ., method="class", data=train)
tree_pred = predict(tree, test, type="class") 
print(sprintf("Accuracy of DT for k = 20:    %.3f", Accuracy(y_true=actual, y_pred=tree_pred)))
print(sprintf("Precision of DT for k = 20:   %.3f", Precision(y_true=actual, y_pred=tree_pred)))
print(sprintf("F1 Score of DT for k = 20:    %.3f", F1_Score(tree_pred, actual)))
print(sprintf("Sensitivity of DT for k = 20: %.3f", Sensitivity(y_true=actual, y_pred=tree_pred)))
print(sprintf("Specificity of DT for k = 20: %.3f", Specificity(y_true=tree_pred, y_pred=actual)))

#SVM...this is the second best
set.seed(123)
svm_fit <- svm(status ~ ., data = train)
summary(svm_fit)

svm_pred <- predict(svm_fit, test) 

print(sprintf("Accuracy of SVM for k = 20:    %.3f", Accuracy(y_true=actual, y_pred=svm_pred)))
print(sprintf("Precision of SVM for k = 20:   %.3f", Precision(y_true=actual, y_pred=svm_pred)))
print(sprintf("F1 Score of SVM for k = 20:    %.3f", F1_Score(svm_pred, actual)))
print(sprintf("Sensitivity of SVM for k = 20: %.3f", Sensitivity(y_true=actual, y_pred=svm_pred)))
print(sprintf("Specificity of SVM for k = 20: %.3f", Specificity(y_true=svm_pred, y_pred=actual)))


#kknn....
set.seed(123)

kknn_fit = train.kknn(status ~ ., train, kmax=7)
summary(kknn_fit)

kknn_pred = predict(kknn_fit, test)

print(sprintf("Accuracy of KNN for k = 20:    %.3f", Accuracy(y_true=actual, y_pred=kknn_pred)))
print(sprintf("Precision of KNN for k = 20:   %.3f", Precision(y_true=actual, y_pred=kknn_pred)))
print(sprintf("F1 Score of KNN for k = 20:    %.3f", F1_Score(kknn_pred, actual)))
print(sprintf("Sensitivity of KNN for k = 20: %.3f", Sensitivity(y_true=actual, y_pred=kknn_pred)))
print(sprintf("Specificity of KNN for k = 20: %.3f", Specificity(y_true=kknn_pred, y_pred=actual)))

# #NN...91%
# install.packages("nnet")
# library(nnet)
# set.seed(123)
# 
# train_nn <- (train)
# test_nn <- test
# str(train_nn)
# 
# train_nn$status <- as.numeric(train_nn$status)
# train_nn$sector <- as.numeric(train_nn$sector)
# train_nn$country <- as.numeric(train_nn$country)
# train_nn$gender <- as.numeric(train_nn$gender)
# train_nn$countrygender <- as.numeric(as.factor(as.character(train_nn$countrygender)))
# train_nn$CountrySector <- as.numeric(as.factor(as.character(train_nn$CountrySector)))
# train_nn$nonpayment <- as.numeric (train_nn$nonpayment)
# 
# test_nn$status <- as.numeric(test_nn$status)
# test_nn$sector <- as.numeric(test_nn$sector)
# test_nn$country <- as.numeric(test_nn$country)
# test_nn$gender <- as.numeric(test_nn$gender)
# test_nn$countrygender <- as.numeric(as.factor(as.character(test_nn$countrygender)))
# test_nn$CountrySector <- as.numeric(as.factor(as.character(test_nn$CountrySector)))
# test_nn$nonpayment <- as.numeric (test_nn$nonpayment)
# str(train)
# ##start Neural Network analysis
# 
# my.grid <- expand.grid( .size = c(1,2,4),.decay = c(0.25,1,2)) # Tuning grid for Neural Net
# 
# model_NN <- train(status~ ., data = train_nn, method = "nnet", tuneGrid = my.grid, trace = TRUE, na.action =na.omit)
# 
# plot(model_NN) #Visualize the relationship between the number of layers, decay and accuracy
# NN_prediction<-predict(model_NN, test_nn) #Predict classification 
# print(sprintf("Accuracy of NN for k = 20:    %.3f", Accuracy(y_true=test_nn$status, y_pred=NN_prediction)))
# print(sprintf("Precision of NN for k = 20:   %.3f", Precision(y_true=test_nn$status, y_pred=NN_prediction)))
# print(sprintf("F1 Score of NN for k = 20:    %.3f", F1_Score(NN_prediction, test_nn$status)))
# print(sprintf("Sensitivity of NN for k = 20: %.3f", Sensitivity(y_true=test_nn$status, y_pred=NN_prediction)))
# print(sprintf("Specificity of NN for k = 20: %.3f", Specificity(y_true=NN_prediction, y_pred=test_nn$status)))

### removing topics from this part
topics <- df[,14:33]
df[,14:33] <- NULL


##### Topic Modelling with Sensitive Study ###############

#### With k = 10 ##########
##### fitting an LDA Model #####

ten_model <- FitLdaModel(dtm = dtm_1_2, 
                         k = 10, 
                         iterations = 500, # recommend a larger value, 500 or more
                         burnin = 180,
                         optimize_alpha = TRUE,
                         calc_likelihood = TRUE,
                         calc_coherence = TRUE,
                         calc_r2 = TRUE,
                         alpha = 0.1, # this is the default value
                         beta = 0.05, # this is the default value
                         cpus = 21) # Note, this is for a big machine  

plot(ten_model$log_likelihood, type = "l")


ten_model$top_terms <- GetTopTerms(phi = ten_model$phi, M = 8)
head(ten_model$top_terms) 

# probabilistic coherence, a measure of topic quality
# this measure can be used with any topic model, not just probabilistic ones
ten_model$coherence <- CalcProbCoherence(phi = ten_model$phi, dtm = dtm_1_2, M = 5)

# Get the prevalence of each topic
# You can make this discrete by applying a threshold, say 0.05, for
# topics in/out of docuemnts. 
ten_model$prevalence <- colSums(ten_model$theta) / sum(ten_model$theta) * 100


# textmineR has a naive topic labeling tool based on probable bigrams
ten_model$labels <- LabelTopics(assignments = ten_model$theta > 0.20, 
                                dtm = dtm_1_2,
                                M = 2)

head(ten_model$labels)


# put them together, with coherence into a summary table
ten_model$summary <- data.frame(topic = rownames(ten_model$phi),
                                label = ten_model$labels,
                                coherence = round(ten_model$coherence, 2),
                                prevalence = round(ten_model$prevalence,1),
                                top_terms = apply(ten_model$top_terms, 2, function(x){
                                  paste(x, collapse = ", ")
                                }),
                                stringsAsFactors = FALSE)

ten_model$summary %>%
  arrange(-prevalence)


write.csv(ten_model$top_terms, file = "1.1_10_Topics_kiva.csv")

#### Adding topics to data frame #####

df = cbind(df, round(ten_model$theta, digits=2))
head(df, n=10)

####################################################################
########## Model Building for k = 10 ###################

# Splitting the Data into train and test

set.seed(123) # Set the seed to make it reproducible

train.index <- createDataPartition(df$status, p = .8, list = FALSE)
train <- df[ train.index,]
test  <- df[-train.index,]

actual <- test$status
formula <- status ~ .
positive <- "yes"

# Random Forest with great results
set.seed(123)
model_forest <- randomForest(status~ ., data=train, 
                             importance=TRUE,proximity=TRUE,
                             cutoff = c(0.5, 0.5),type="classification", na.action=na.exclude)
print(model_forest)   
plot(model_forest)
importance(model_forest)

varImpPlot(model_forest)#Shows the decrease in accuracy with the removal of a variables so non Annual removal the accuray can go down by 30% for example

predicted <- predict(model_forest, test, type="class") 

print(sprintf("Accuracy of Random Forest for k = 10:    %.3f", Accuracy(y_true=actual, y_pred=predicted)))
print(sprintf("Precision of Random Forest for k = 10:   %.3f", Precision(y_true=actual, y_pred=predicted)))
print(sprintf("Recall of Random Forest for k = 10:      %.3f", Recall(y_true=actual, y_pred=predicted)))
print(sprintf("F1 Score of Random Forest for k = 10:    %.3f", F1_Score(predicted, actual)))
print(sprintf("Sensitivity of Random Forest for k = 10: %.3f", Sensitivity(y_true=actual, y_pred=predicted)))
print(sprintf("Specificity of Random Forest for k = 10: %.3f", Specificity(y_true=predicted, y_pred=actual)))

# Naive Bayes...Accuracy of 0.73 not very good
set.seed(123)

nb_fit = naiveBayes(status ~ ., data=train)
nb_fit

NB_pred = predict(nb_fit, test, type="class") 
print(sprintf("Accuracy of NB for k = 10:    %.3f", Accuracy(y_true=actual, y_pred=NB_pred)))
print(sprintf("Precision of NB for k = 10:   %.3f", Precision(y_true=actual, y_pred=NB_pred)))
print(sprintf("F1 Score of NB for k = 10:    %.3f", F1_Score(NB_pred, actual)))
print(sprintf("Sensitivityof NB for k = 10: %.3f", Sensitivity(y_true=actual, y_pred=NB_pred)))
print(sprintf("Specificity of NB for k = 10: %.3f", Specificity(y_true=NB_pred, y_pred=actual)))

# Decision Tree

set.seed(123)

tree <- rpart(status ~ ., method="class", data=train)
tree_pred = predict(tree, test, type="class") 
print(sprintf("Accuracy of DT for k = 10:    %.3f", Accuracy(y_true=actual, y_pred=tree_pred)))
print(sprintf("Precision of DT for k = 10:   %.3f", Precision(y_true=actual, y_pred=tree_pred)))
print(sprintf("F1 Score of DT for k = 10:    %.3f", F1_Score(tree_pred, actual)))
print(sprintf("Sensitivity of DT for k = 10: %.3f", Sensitivity(y_true=actual, y_pred=tree_pred)))
print(sprintf("Specificity of DT for k = 10: %.3f", Specificity(y_true=tree_pred, y_pred=actual)))

#SVM...this is the second best
set.seed(123)
svm_fit <- svm(status ~ ., data = train)
summary(svm_fit)

svm_pred <- predict(svm_fit, test) 

print(sprintf("Accuracy of SVM for k = 10:    %.3f", Accuracy(y_true=actual, y_pred=svm_pred)))
print(sprintf("Precision of SVM for k = 10:   %.3f", Precision(y_true=actual, y_pred=svm_pred)))
print(sprintf("F1 Score of SVM for k = 10:    %.3f", F1_Score(svm_pred, actual)))
print(sprintf("Sensitivity of SVM for k = 10: %.3f", Sensitivity(y_true=actual, y_pred=svm_pred)))
print(sprintf("Specificity of SVM for k = 10: %.3f", Specificity(y_true=svm_pred, y_pred=actual)))


#kknn....89%
set.seed(123)

kknn_fit = train.kknn(status ~ ., train, kmax=7)
summary(kknn_fit)

kknn_pred = predict(kknn_fit, test)

print(sprintf("Accuracy of KNN for k = 10:    %.3f", Accuracy(y_true=actual, y_pred=kknn_pred)))
print(sprintf("Precision of KNN for k = 10:   %.3f", Precision(y_true=actual, y_pred=kknn_pred)))
print(sprintf("F1 Score of KNN for k = 10:    %.3f", F1_Score(kknn_pred, actual)))
print(sprintf("Sensitivity of KNN for k = 10: %.3f", Sensitivity(y_true=actual, y_pred=kknn_pred)))
print(sprintf("Specificity of KNN for k = 10: %.3f", Specificity(y_true=kknn_pred, y_pred=actual)))

# #NN...91%
# install.packages("nnet")
# library(nnet)
# set.seed(123)
# 
# train_nn <- (train)
# test_nn <- test
# str(train_nn)
# 
# train_nn$status <- as.numeric(train_nn$status)
# train_nn$sector <- as.numeric(train_nn$sector)
# train_nn$country <- as.numeric(train_nn$country)
# train_nn$gender <- as.numeric(train_nn$gender)
# train_nn$countrygender <- as.numeric(as.factor(as.character(train_nn$countrygender)))
# train_nn$CountrySector <- as.numeric(as.factor(as.character(train_nn$CountrySector)))
# train_nn$nonpayment <- as.numeric (train_nn$nonpayment)
# 
# test_nn$status <- as.numeric(test_nn$status)
# test_nn$sector <- as.numeric(test_nn$sector)
# test_nn$country <- as.numeric(test_nn$country)
# test_nn$gender <- as.numeric(test_nn$gender)
# test_nn$countrygender <- as.numeric(as.factor(as.character(test_nn$countrygender)))
# test_nn$CountrySector <- as.numeric(as.factor(as.character(test_nn$CountrySector)))
# test_nn$nonpayment <- as.numeric (test_nn$nonpayment)
# str(train)
# ##start Neural Network analysis
# 
# my.grid <- expand.grid( .size = c(1,2,4),.decay = c(0.25,1,2)) # Tuning grid for Neural Net
# 
# model_NN <- train(status~ ., data = train_nn, method = "nnet", tuneGrid = my.grid, trace = TRUE, na.action =na.omit)
# 
# plot(model_NN) #Visualize the relationship between the number of layers, decay and accuracy
# NN_prediction<-predict(model_NN, test_nn) #Predict classification 
# print(sprintf("Accuracy of NN for k = 10:    %.3f", Accuracy(y_true=test_nn$status, y_pred=NN_prediction)))
# print(sprintf("Precision of NN for k = 10:   %.3f", Precision(y_true=test_nn$status, y_pred=NN_prediction)))
# print(sprintf("F1 Score of NN for k = 10:    %.3f", F1_Score(NN_prediction, test_nn$status)))
# print(sprintf("Sensitivity of NN for k = 10: %.3f", Sensitivity(y_true=test_nn$status, y_pred=NN_prediction)))
# print(sprintf("Specificity of NN for k = 10: %.3f", Specificity(y_true=NN_prediction, y_pred=test_nn$status)))

### removing topics from this part
topics <- cbind(df[,14:23])
### removing topics from this part
topics <- df[,14:23]
df[,14:23] <- NULL

##### Topic Modelling with Sensitive Study ###############

#### With k = 15 ##########

##### fitting an LDA Model #####

fifteen_model <- FitLdaModel(dtm = dtm_1_2, 
                             k = 15, 
                             iterations = 500, # recommend a larger value, 500 or more
                             burnin = 180,
                             optimize_alpha = TRUE,
                             calc_likelihood = TRUE,
                             calc_coherence = TRUE,
                             calc_r2 = TRUE,
                             alpha = 0.1, # this is the default value
                             beta = 0.05, # this is the default value
                             cpus = 21) # Note, this is for a big machine  

plot(fifteen_model$log_likelihood, type = "l")


fifteen_model$top_terms <- GetTopTerms(phi = fifteen_model$phi, M = 8)
head(fifteen_model$top_terms) 

# probabilistic coherence, a measure of topic quality
# this measure can be used with any topic model, not just probabilistic ones
fifteen_model$coherence <- CalcProbCoherence(phi = fifteen_model$phi, dtm = dtm_1_2, M = 5)

# Get the prevalence of each topic
# You can make this discrete by applying a threshold, say 0.05, for
# topics in/out of docuemnts. 
fifteen_model$prevalence <- colSums(fifteen_model$theta) / sum(fifteen_model$theta) * 100


# textmineR has a naive topic labeling tool based on probable bigrams
fifteen_model$labels <- LabelTopics(assignments = fifteen_model$theta > 0.20, 
                                    dtm = dtm_1_2,
                                    M = 2)

head(fifteen_model$labels)


# put them together, with coherence into a summary table
fifteen_model$summary <- data.frame(topic = rownames(fifteen_model$phi),
                                    label = fifteen_model$labels,
                                    coherence = round(fifteen_model$coherence, 2),
                                    prevalence = round(fifteen_model$prevalence,1),
                                    top_terms = apply(fifteen_model$top_terms, 2, function(x){
                                      paste(x, collapse = ", ")
                                    }),
                                    stringsAsFactors = FALSE)

fifteen_model$summary %>%
  arrange(-prevalence)


write.csv(fifteen_model$top_terms, file = "1.1_15_Topics_kiva.csv")

#### Adding topics to data frame #####

df = cbind(df, round(fifteen_model$theta, digits=2))
head(df, n=10)

####################################################################
########## Model Building for k = 15 ###################

# Splitting the Data into train and test

set.seed(123) # Set the seed to make it reproducible

train.index <- createDataPartition(df$status, p = .8, list = FALSE)
train <- df[ train.index,]
test  <- df[-train.index,]

actual <- test$status
formula <- status ~ .
positive <- "yes"

# Random Forest with great results
set.seed(123)
model_forest <- randomForest(status~ ., data=train, 
                             importance=TRUE,proximity=TRUE,
                             cutoff = c(0.5, 0.5),type="classification", na.action=na.exclude)
print(model_forest)   
plot(model_forest)
importance(model_forest)

varImpPlot(model_forest)#Shows the decrease in accuracy with the removal of a variables so non Annual removal the accuray can go down by 30% for example

predicted <- predict(model_forest, test, type="class") 

print(sprintf("Accuracy of Random Forest for k = 15:    %.3f", Accuracy(y_true=actual, y_pred=predicted)))
print(sprintf("AUC of Random Forest for k = 15:     %.3f", AUC(y_pred=predicted, y_true=actual)))
print(sprintf("Precision of Random Forest for k = 15:   %.3f", Precision(y_true=actual, y_pred=predicted)))
print(sprintf("Recall of Random Forest for k = 15:      %.3f", Recall(y_true=actual, y_pred=predicted)))
print(sprintf("F1 Score of Random Forest for k = 15:    %.3f", F1_Score(predicted, actual)))
print(sprintf("Sensitivity of Random Forest for k = 15: %.3f", Sensitivity(y_true=actual, y_pred=predicted)))
print(sprintf("Specificity of Random Forest for k = 15: %.3f", Specificity(y_true=predicted, y_pred=actual)))

# Naive Bayes...Accuracy of 0.73 not very good
set.seed(123)

nb_fit = naiveBayes(status ~ ., data=train)
nb_fit

NB_pred = predict(nb_fit, test, type="class") 
print(sprintf("Accuracy of NB for k = 15:    %.3f", Accuracy(y_true=actual, y_pred=NB_pred)))
print(sprintf("Precision of NB for k = 15:   %.3f", Precision(y_true=actual, y_pred=NB_pred)))
print(sprintf("F1 Score of NB for k = 15:    %.3f", F1_Score(NB_pred, actual)))
print(sprintf("Sensitivityof NB for k = 15: %.3f", Sensitivity(y_true=actual, y_pred=NB_pred)))
print(sprintf("Specificity of NB for k = 15: %.3f", Specificity(y_true=NB_pred, y_pred=actual)))

# Decision Tree

set.seed(123)

tree <- rpart(status ~ ., method="class", data=train)
tree_pred = predict(tree, test, type="class") 
print(sprintf("Accuracy of DT for k = 15:    %.3f", Accuracy(y_true=actual, y_pred=tree_pred)))
print(sprintf("Precision of DT for k = 15:   %.3f", Precision(y_true=actual, y_pred=tree_pred)))
print(sprintf("F1 Score of DT for k = 15:    %.3f", F1_Score(tree_pred, actual)))
print(sprintf("Sensitivity of DT for k = 15: %.3f", Sensitivity(y_true=actual, y_pred=tree_pred)))
print(sprintf("Specificity of DT for k = 15: %.3f", Specificity(y_true=tree_pred, y_pred=actual)))

#SVM...this is the second best

svm_fit <- svm(status ~ ., data = train)
summary(svm_fit)

svm_pred <- predict(svm_fit, test) 

print(sprintf("Accuracy of SVM for k = 15:    %.3f", Accuracy(y_true=actual, y_pred=svm_pred)))
print(sprintf("Precision of SVM for k = 15:   %.3f", Precision(y_true=actual, y_pred=svm_pred)))
print(sprintf("F1 Score of SVM for k = 15:    %.3f", F1_Score(svm_pred, actual)))
print(sprintf("Sensitivity of SVM for k = 15: %.3f", Sensitivity(y_true=actual, y_pred=svm_pred)))
print(sprintf("Specificity of SVM for k = 15: %.3f", Specificity(y_true=svm_pred, y_pred=actual)))


#kknn....89%
set.seed(123)
kknn_fit = train.kknn(status ~ ., train, kmax=7)
summary(kknn_fit)

kknn_pred = predict(kknn_fit, test)

print(sprintf("Accuracy of KNN for k = 15:    %.3f", Accuracy(y_true=actual, y_pred=kknn_pred)))
print(sprintf("Precision of KNN for k = 15:   %.3f", Precision(y_true=actual, y_pred=kknn_pred)))
print(sprintf("F1 Score of KNN for k = 15:    %.3f", F1_Score(kknn_pred, actual)))
print(sprintf("Sensitivity of KNN for k = 15: %.3f", Sensitivity(y_true=actual, y_pred=kknn_pred)))
print(sprintf("Specificity of KNN for k = 15: %.3f", Specificity(y_true=kknn_pred, y_pred=actual)))

# #NN...91%
# install.packages("nnet")
# library(nnet)
# set.seed(123)
# 
# train_nn <- (train)
# test_nn <- test
# str(train_nn)
# 
# train_nn$status <- as.numeric(train_nn$status)
# train_nn$sector <- as.numeric(train_nn$sector)
# train_nn$country <- as.numeric(train_nn$country)
# train_nn$gender <- as.numeric(train_nn$gender)
# train_nn$countrygender <- as.numeric(as.factor(as.character(train_nn$countrygender)))
# train_nn$CountrySector <- as.numeric(as.factor(as.character(train_nn$CountrySector)))
# train_nn$nonpayment <- as.numeric (train_nn$nonpayment)
# 
# test_nn$status <- as.numeric(test_nn$status)
# test_nn$sector <- as.numeric(test_nn$sector)
# test_nn$country <- as.numeric(test_nn$country)
# test_nn$gender <- as.numeric(test_nn$gender)
# test_nn$countrygender <- as.numeric(as.factor(as.character(test_nn$countrygender)))
# test_nn$CountrySector <- as.numeric(as.factor(as.character(test_nn$CountrySector)))
# test_nn$nonpayment <- as.numeric (test_nn$nonpayment)
# str(train)
# ##start Neural Network analysis
# 
# my.grid <- expand.grid( .size = c(1,2,4),.decay = c(0.25,1,2)) # Tuning grid for Neural Net
# 
# model_NN <- train(status~ ., data = train_nn, method = "nnet", tuneGrid = my.grid, trace = TRUE, na.action =na.omit)
# 
# plot(model_NN) #Visualize the relationship between the number of layers, decay and accuracy
# NN_prediction<-predict(model_NN, test_nn) #Predict classification 
# print(sprintf("Accuracy of NN for k = 15:    %.3f", Accuracy(y_true=test_nn$status, y_pred=NN_prediction)))
# print(sprintf("Precision of NN for k = 15:   %.3f", Precision(y_true=test_nn$status, y_pred=NN_prediction)))
# print(sprintf("F1 Score of NN for k = 15:    %.3f", F1_Score(NN_prediction, test_nn$status)))
# print(sprintf("Sensitivity of NN for k = 15: %.3f", Sensitivity(y_true=test_nn$status, y_pred=NN_prediction)))
# print(sprintf("Specificity of NN for k = 15: %.3f", Specificity(y_true=NN_prediction, y_pred=test_nn$status)))

### removing topics from this part
topics <- cbind(df[,14:28])
### removing topics from this part
topics <- df[,14:28]
df[,14:28] <- NULL

##### Topic Modelling with Sensitive Study ###############

#### With k = 25 ##########

##### fitting an LDA Model #####

twentyfive_model <- FitLdaModel(dtm = dtm_1_2, 
                                k = 25, 
                                iterations = 500, # recommend a larger value, 500 or more
                                burnin = 180,
                                optimize_alpha = TRUE,
                                calc_likelihood = TRUE,
                                calc_coherence = TRUE,
                                calc_r2 = TRUE,
                                alpha = 0.1, # this is the default value
                                beta = 0.05, # this is the default value
                                cpus = 21) # Note, this is for a big machine  

plot(twentyfive_model$log_likelihood, type = "l")


twentyfive_model$top_terms <- GetTopTerms(phi = twentyfive_model$phi, M = 8)
head(twentyfive_model$top_terms) 

# probabilistic coherence, a measure of topic quality
# this measure can be used with any topic model, not just probabilistic ones
twentyfive_model$coherence <- CalcProbCoherence(phi = twentyfive_model$phi, dtm = dtm_1_2, M = 5)

# Get the prevalence of each topic
# You can make this discrete by applying a threshold, say 0.05, for
# topics in/out of docuemnts. 
twentyfive_model$prevalence <- colSums(twentyfive_model$theta) / sum(twentyfive_model$theta) * 100


# textmineR has a naive topic labeling tool based on probable bigrams
twentyfive_model$labels <- LabelTopics(assignments = twentyfive_model$theta > 0.20, 
                                       dtm = dtm_1_2,
                                       M = 2)

head(twentyfive_model$labels)


# put them together, with coherence into a summary table
twentyfive_model$summary <- data.frame(topic = rownames(twentyfive_model$phi),
                                       label = twentyfive_model$labels,
                                       coherence = round(twentyfive_model$coherence, 2),
                                       prevalence = round(twentyfive_model$prevalence,1),
                                       top_terms = apply(twentyfive_model$top_terms, 2, function(x){
                                         paste(x, collapse = ", ")
                                       }),
                                       stringsAsFactors = FALSE)

twentyfive_model$summary %>%
  arrange(-prevalence)


write.csv(twentyfive_model$top_terms, file = "1.1_25_Topics_kiva.csv")

#### Adding topics to data frame #####

df = cbind(df, round(twentyfive_model$theta, digits=2))
head(df, n=10)

####################################################################
########## Model Building for k = 25 ###################

# Splitting the Data into train and test

set.seed(123) # Set the seed to make it reproducible

train.index <- createDataPartition(df$status, p = .8, list = FALSE)
train <- df[ train.index,]
test  <- df[-train.index,]

actual <- test$status
formula <- status ~ .
positive <- "yes"

# Random Forest with great results
set.seed(123)
model_forest <- randomForest(status~ ., data=train, 
                             importance=TRUE,proximity=TRUE,
                             cutoff = c(0.5, 0.5),type="classification", na.action=na.exclude)
print(model_forest)   
plot(model_forest)
importance(model_forest)

varImpPlot(model_forest)#Shows the decrease in accuracy with the removal of a variables so non Annual removal the accuray can go down by 30% for example

predicted <- predict(model_forest, test, type="class") 

print(sprintf("Accuracy of Random Forest for k = 25:    %.3f", Accuracy(y_true=actual, y_pred=predicted)))
print(sprintf("AUC of Random Forest for k = 25:     %.3f", AUC(y_pred=predicted, y_true=actual)))
print(sprintf("Precision of Random Forest for k = 25:   %.3f", Precision(y_true=actual, y_pred=predicted)))
print(sprintf("Recall of Random Forest for k = 25:      %.3f", Recall(y_true=actual, y_pred=predicted)))
print(sprintf("F1 Score of Random Forest for k = 25:    %.3f", F1_Score(predicted, actual)))
print(sprintf("Sensitivity of Random Forest for k = 25: %.3f", Sensitivity(y_true=actual, y_pred=predicted)))
print(sprintf("Specificity of Random Forest for k = 25: %.3f", Specificity(y_true=predicted, y_pred=actual)))

# Naive Bayes...Accuracy of 0.73 not very good
set.seed(123)

nb_fit = naiveBayes(status ~ ., data=train)
nb_fit

NB_pred = predict(nb_fit, test, type="class") 
print(sprintf("Accuracy of NB for k = 25:    %.3f", Accuracy(y_true=actual, y_pred=NB_pred)))
print(sprintf("Precision of NB for k = 25:   %.3f", Precision(y_true=actual, y_pred=NB_pred)))
print(sprintf("F1 Score of NB for k = 25:    %.3f", F1_Score(NB_pred, actual)))
print(sprintf("Sensitivityof NB for k = 25: %.3f", Sensitivity(y_true=actual, y_pred=NB_pred)))
print(sprintf("Specificity of NB for k = 25: %.3f", Specificity(y_true=NB_pred, y_pred=actual)))
# Decision Tree

set.seed(123)

tree <- rpart(status ~ ., method="class", data=train)
tree_pred = predict(tree, test, type="class") 
print(sprintf("Accuracy of DT for k = 25:    %.3f", Accuracy(y_true=actual, y_pred=tree_pred)))
print(sprintf("Precision of DT for k = 25:   %.3f", Precision(y_true=actual, y_pred=tree_pred)))
print(sprintf("F1 Score of DT for k = 25:    %.3f", F1_Score(tree_pred, actual)))
print(sprintf("Sensitivity of DT for k = 25: %.3f", Sensitivity(y_true=actual, y_pred=tree_pred)))
print(sprintf("Specificity of DT for k = 25: %.3f", Specificity(y_true=tree_pred, y_pred=actual)))

#SVM...this is the second best

svm_fit <- svm(status ~ ., data = train)
summary(svm_fit)

svm_pred <- predict(svm_fit, test) 

print(sprintf("Accuracy of SVM for k = 25:    %.3f", Accuracy(y_true=actual, y_pred=svm_pred)))
print(sprintf("Precision of SVM for k = 25:   %.3f", Precision(y_true=actual, y_pred=svm_pred)))
print(sprintf("F1 Score of SVM for k = 25:    %.3f", F1_Score(svm_pred, actual)))
print(sprintf("Sensitivity of SVM for k = 25: %.3f", Sensitivity(y_true=actual, y_pred=svm_pred)))
print(sprintf("Specificity of SVM for k = 25: %.3f", Specificity(y_true=svm_pred, y_pred=actual)))


#kknn....89%
set.seed(123)
kknn_fit = train.kknn(status ~ ., train, kmax=7)
summary(kknn_fit)

kknn_pred = predict(kknn_fit, test)

print(sprintf("Accuracy of KNN for k = 25:    %.3f", Accuracy(y_true=actual, y_pred=kknn_pred)))
print(sprintf("Precision of KNN for k = 25:   %.3f", Precision(y_true=actual, y_pred=kknn_pred)))
print(sprintf("F1 Score of KNN for k = 25:    %.3f", F1_Score(kknn_pred, actual)))
print(sprintf("Sensitivity of KNN for k = 25: %.3f", Sensitivity(y_true=actual, y_pred=kknn_pred)))
print(sprintf("Specificity of KNN for k = 25: %.3f", Specificity(y_true=kknn_pred, y_pred=actual)))

# #NN...91%
# install.packages("nnet")
# library(nnet)
# set.seed(123)
# 
# train_nn <- (train)
# test_nn <- test
# str(train_nn)
# 
# train_nn$status <- as.numeric(train_nn$status)
# train_nn$sector <- as.numeric(train_nn$sector)
# train_nn$country <- as.numeric(train_nn$country)
# train_nn$gender <- as.numeric(train_nn$gender)
# train_nn$countrygender <- as.numeric(as.factor(as.character(train_nn$countrygender)))
# train_nn$CountrySector <- as.numeric(as.factor(as.character(train_nn$CountrySector)))
# train_nn$nonpayment <- as.numeric (train_nn$nonpayment)
# 
# test_nn$status <- as.numeric(test_nn$status)
# test_nn$sector <- as.numeric(test_nn$sector)
# test_nn$country <- as.numeric(test_nn$country)
# test_nn$gender <- as.numeric(test_nn$gender)
# test_nn$countrygender <- as.numeric(as.factor(as.character(test_nn$countrygender)))
# test_nn$CountrySector <- as.numeric(as.factor(as.character(test_nn$CountrySector)))
# test_nn$nonpayment <- as.numeric (test_nn$nonpayment)
# str(train)
# ##start Neural Network analysis
# 
# my.grid <- expand.grid( .size = c(1,2,4),.decay = c(0.25,1,2)) # Tuning grid for Neural Net
# 
# model_NN <- train(status~ ., data = train_nn, method = "nnet", tuneGrid = my.grid, trace = TRUE, na.action =na.omit)
# 
# plot(model_NN) #Visualize the relationship between the number of layers, decay and accuracy
# NN_prediction<-predict(model_NN, test_nn) #Predict classification 
# print(sprintf("Accuracy of NN for k = 25:    %.3f", Accuracy(y_true=test_nn$status, y_pred=NN_prediction)))
# print(sprintf("Precision of NN for k = 25:   %.3f", Precision(y_true=test_nn$status, y_pred=NN_prediction)))
# print(sprintf("F1 Score of NN for k = 25:    %.3f", F1_Score(NN_prediction, test_nn$status)))
# print(sprintf("Sensitivity of NN for k = 25: %.3f", Sensitivity(y_true=test_nn$status, y_pred=NN_prediction)))
# print(sprintf("Specificity of NN for k = 25: %.3f", Specificity(y_true=NN_prediction, y_pred=test_nn$status)))


### removing topics from this part
topics <- cbind(df[,14:38])
### removing topics from this part
df[,14:38] <- NULL

##### Topic Modelling with Sensitive Study ###############

#### With k = 30 ##########

##### fitting an LDA Model #####

thirty_model <- FitLdaModel(dtm = dtm_1_2, 
                            k = 30, 
                            iterations = 500, # recommend a larger value, 500 or more
                            burnin = 180,
                            optimize_alpha = TRUE,
                            calc_likelihood = TRUE,
                            calc_coherence = TRUE,
                            calc_r2 = TRUE,
                            alpha = 0.1, # this is the default value
                            beta = 0.05, # this is the default value
                            cpus = 21) # Note, this is for a big machine  

plot(thirty_model$log_likelihood, type = "l")


thirty_model$top_terms <- GetTopTerms(phi = thirty_model$phi, M = 8)
head(thirty_model$top_terms) 

# probabilistic coherence, a measure of topic quality
# this measure can be used with any topic model, not just probabilistic ones
thirty_model$coherence <- CalcProbCoherence(phi = thirty_model$phi, dtm = dtm_1_2, M = 5)

# Get the prevalence of each topic
# You can make this discrete by applying a threshold, say 0.05, for
# topics in/out of docuemnts. 
thirty_model$prevalence <- colSums(thirty_model$theta) / sum(thirty_model$theta) * 100


# textmineR has a naive topic labeling tool based on probable bigrams
thirty_model$labels <- LabelTopics(assignments = thirty_model$theta > 0.20, 
                                   dtm = dtm_1_2,
                                   M = 2)

head(thirty_model$labels)


# put them together, with coherence into a summary table
thirty_model$summary <- data.frame(topic = rownames(thirty_model$phi),
                                   label = thirty_model$labels,
                                   coherence = round(thirty_model$coherence, 2),
                                   prevalence = round(thirty_model$prevalence,1),
                                   top_terms = apply(thirty_model$top_terms, 2, function(x){
                                     paste(x, collapse = ", ")
                                   }),
                                   stringsAsFactors = FALSE)

thirty_model$summary %>%
  arrange(-prevalence)


write.csv(thirty_model$top_terms, file = "1.1_30_Topics_kiva.csv")

#### Adding topics to data frame #####

df = cbind(df, round(thirty_model$theta, digits=2))
head(df, n=10)

####################################################################
########## Model Building for k = 15 ###################

# Splitting the Data into train and test

set.seed(123) # Set the seed to make it reproducible

train.index <- createDataPartition(df$status, p = .8, list = FALSE)
train <- df[ train.index,]
test  <- df[-train.index,]

actual <- test$status
formula <- status ~ .
positive <- "yes"

# Random Forest with great results
set.seed(123)
model_forest <- randomForest(status~ ., data=train, 
                             importance=TRUE,proximity=TRUE,
                             cutoff = c(0.5, 0.5),type="classification", na.action=na.exclude)
print(model_forest)   
plot(model_forest)
importance(model_forest)

varImpPlot(model_forest)#Shows the decrease in accuracy with the removal of a variables so non Annual removal the accuray can go down by 30% for example

predicted <- predict(model_forest, test, type="class") 

print(sprintf("Accuracy of Random Forest for k = 30:    %.3f", Accuracy(y_true=actual, y_pred=predicted)))
print(sprintf("AUC of Random Forest for k = 30:     %.3f", AUC(y_pred=predicted, y_true=actual)))
print(sprintf("Precision of Random Forest for k = 30:   %.3f", Precision(y_true=actual, y_pred=predicted)))
print(sprintf("Recall of Random Forest for k = 30:      %.3f", Recall(y_true=actual, y_pred=predicted)))
print(sprintf("F1 Score of Random Forest for k = 30:    %.3f", F1_Score(predicted, actual)))
print(sprintf("Sensitivity of Random Forest for k = 30: %.3f", Sensitivity(y_true=actual, y_pred=predicted)))
print(sprintf("Specificity of Random Forest for k = 30: %.3f", Specificity(y_true=predicted, y_pred=actual)))

# Naive Bayes...Accuracy of 0.73 not very good
set.seed(123)

nb_fit = naiveBayes(status ~ ., data=train)
nb_fit

NB_pred = predict(nb_fit, test, type="class") 
print(sprintf("Accuracy of NB for k = 30:    %.3f", Accuracy(y_true=actual, y_pred=NB_pred)))
print(sprintf("Precision of NB for k = 30:   %.3f", Precision(y_true=actual, y_pred=NB_pred)))
print(sprintf("F1 Score of NB for k = 30:    %.3f", F1_Score(NB_pred, actual)))
print(sprintf("Sensitivityof NB for k = 30: %.3f", Sensitivity(y_true=actual, y_pred=NB_pred)))
print(sprintf("Specificity of NB for k = 30: %.3f", Specificity(y_true=NB_pred, y_pred=actual)))


# Decision Tree
set.seed(123)

tree <- rpart(status ~ ., method="class", data=train)
tree_pred = predict(tree, test, type="class") 
print(sprintf("Accuracy of DT for k = 30:    %.3f", Accuracy(y_true=actual, y_pred=tree_pred)))
print(sprintf("Precision of DT for k = 30:   %.3f", Precision(y_true=actual, y_pred=tree_pred)))
print(sprintf("F1 Score of DT for k = 30:    %.3f", F1_Score(tree_pred, actual)))
print(sprintf("Sensitivity of DT for k = 30: %.3f", Sensitivity(y_true=actual, y_pred=tree_pred)))
print(sprintf("Specificity of DT for k = 30: %.3f", Specificity(y_true=tree_pred, y_pred=actual)))

#SVM...this is the second best
set.seed(123)
svm_fit <- svm(status ~ ., data = train)
summary(svm_fit)

svm_pred <- predict(svm_fit, test) 

print(sprintf("Accuracy of SVM for k = 30:    %.3f", Accuracy(y_true=actual, y_pred=svm_pred)))
print(sprintf("Precision of SVM for k = 30:   %.3f", Precision(y_true=actual, y_pred=svm_pred)))
print(sprintf("F1 Score of SVM for k = 30:    %.3f", F1_Score(svm_pred, actual)))
print(sprintf("Sensitivity of SVM for k = 30: %.3f", Sensitivity(y_true=actual, y_pred=svm_pred)))
print(sprintf("Specificity of SVM for k = 30: %.3f", Specificity(y_true=svm_pred, y_pred=actual)))


#kknn....89%
set.seed(123)
kknn_fit = train.kknn(status ~ ., train, kmax=7)
summary(kknn_fit)

kknn_pred = predict(kknn_fit, test)

print(sprintf("Accuracy of KNN for k = 30:    %.3f", Accuracy(y_true=actual, y_pred=kknn_pred)))
print(sprintf("Precision of KNN for k = 30:   %.3f", Precision(y_true=actual, y_pred=kknn_pred)))
print(sprintf("F1 Score of KNN for k = 30:    %.3f", F1_Score(kknn_pred, actual)))
print(sprintf("Sensitivity of KNN for k = 30: %.3f", Sensitivity(y_true=actual, y_pred=kknn_pred)))
print(sprintf("Specificity of KNN for k = 30: %.3f", Specificity(y_true=kknn_pred, y_pred=actual)))

# #NN...91%
# install.packages("nnet")
# library(nnet)
# set.seed(123)
# 
# train_nn <- (train)
# test_nn <- test
# str(train_nn)
# 
# train_nn$status <- as.numeric(train_nn$status)
# train_nn$sector <- as.numeric(train_nn$sector)
# train_nn$country <- as.numeric(train_nn$country)
# train_nn$gender <- as.numeric(train_nn$gender)
# train_nn$countrygender <- as.numeric(as.factor(as.character(train_nn$countrygender)))
# train_nn$CountrySector <- as.numeric(as.factor(as.character(train_nn$CountrySector)))
# train_nn$nonpayment <- as.numeric (train_nn$nonpayment)
# 
# test_nn$status <- as.numeric(test_nn$status)
# test_nn$sector <- as.numeric(test_nn$sector)
# test_nn$country <- as.numeric(test_nn$country)
# test_nn$gender <- as.numeric(test_nn$gender)
# test_nn$countrygender <- as.numeric(as.factor(as.character(test_nn$countrygender)))
# test_nn$CountrySector <- as.numeric(as.factor(as.character(test_nn$CountrySector)))
# test_nn$nonpayment <- as.numeric (test_nn$nonpayment)
# str(train)
# ##start Neural Network analysis
# 
# my.grid <- expand.grid( .size = c(1,2,4),.decay = c(0.25,1,2)) # Tuning grid for Neural Net
# 
# model_NN <- train(status~ ., data = train_nn, method = "nnet", tuneGrid = my.grid, trace = TRUE, na.action =na.omit)
# 
# plot(model_NN) #Visualize the relationship between the number of layers, decay and accuracy
# NN_prediction<-predict(model_NN, test_nn) #Predict classification 
# print(sprintf("Accuracy of NN for k = 30:    %.3f", Accuracy(y_true=test_nn$status, y_pred=NN_prediction)))
# print(sprintf("Precision of NN for k = 30:   %.3f", Precision(y_true=test_nn$status, y_pred=NN_prediction)))
# print(sprintf("F1 Score of NN for k = 30:    %.3f", F1_Score(NN_prediction, test_nn$status)))
# print(sprintf("Sensitivity of NN for k = 30: %.3f", Sensitivity(y_true=test_nn$status, y_pred=NN_prediction)))
# print(sprintf("Specificity of NN for k = 30: %.3f", Specificity(y_true=NN_prediction, y_pred=test_nn$status)))

### removing topics from this part
topics <- cbind(df[,14:43])
### removing topics from this part
df[,14:43] <- NULL


##########################################################################
######### Topic Modelling ################################################
################### LDA ####################### 1,3

df <- read.csv("D:\\kiva_data.csv", stringsAsFactors=FALSE)

########## Renaming and type setting of attributes within the dataset ###########

df = df %>%
  rename(story = en)
str(df)
df$id = 1:nrow(df)
df$status = as.factor(df$status)
df$sector = as.factor(df$sector)
df$country = as.factor(df$country)
df$gender = as.factor(df$gender)
df$nonpayment = as.factor(df$nonpayment)

########## Remove HTML Tags ###################
df = df %>% 
  mutate(story = gsub("<.*?>", "", story))

########## Retain distinct entries ################

dim(df)
df = distinct(df, story, .keep_all = TRUE)
dim(df)

############## Clustering the applicants based on all attributes but the story #######

cl <- df
cl$X <- NULL
cl$loan_nonpayment <- NULL
cl$id <- NULL
str(cl)
cl$status <- as.numeric(cl$status)
cl$sector <- as.numeric(cl$sector)
cl$story <- NULL
cl$country <- as.numeric(cl$country)
cl$gender <- as.numeric(cl$gender)
cl$countrygender <- as.numeric(as.factor(as.character(cl$countrygender)))
cl$CountrySector <- as.numeric(as.factor(as.character(cl$CountrySector)))
cl$loan_amount <- as.numeric (cl$loan_amount)
cl$nonpayment <- as.numeric (cl$nonpayment)

## Elbow plot
wss <- (nrow(cl)-1)*sum(apply(cl,2,var))

for (i in 1:30) wss[i] <- sum(kmeans(cl, 
                                     centers=i)$withinss)
plot(1:30, wss, type="b", xlab="Number of Clusters",
     ylab="Within groups sum of squares")

## Choose 4 clusters 

clust <- kmeans(cl, centers = 4, n = 4)

head(clust$cluster)
df <- cbind(df, cluster_name = clust$cluster)

#################### Sentiment Detection ######################

Sentiment <- sentiment_by(df$story)
summary(Sentiment$ave_sentiment)

qplot(Sentiment$ave_sentiment, geom = "histogram", binwidth = 0.1, main = "Review Sentiment Histogram")
head(sentiment)
df$ave_sentiment <- Sentiment$ave_sentiment
df$sd_sentiment <- Sentiment$sd
df$word_count <- Sentiment$word_count

rm(Sentiment, cl, clust)
########## Removing text that is not neccessary for topic identification or further analysis #############
dim(df)
df = distinct(df, story, .keep_all = TRUE)
dim(df)

idx = grep("Translated from Spanish", df$story)
df$story[idx[1]]
df$story = gsub("Translated from Spanish.*Kiva volunteer", " ", df$story, ignore.case = TRUE)
grep("Translated from Spanish", df$story)

grep("Mifex offers", df$story)
df$story = gsub("Mifex offers.*www.mifex.org", " ", df$story, ignore.case = TRUE)
grep("Mifex offers", df$story)

df$story = gsub("About KADET:.*within those communities", " ", df$story, ignore.case = TRUE)
df$story = gsub("Disclaimer: Due to recent events in.*making their loan.", " ", df$story, ignore.case = TRUE)



####### Create Document Term Matrix ###########
# setting n-gram window to be exactly 1 -- unigrams are necessary for our analysis ########## Eliminating stopwords from our text based on english, french, spanish and SMART distionaries ##### converting all uppercase letters to lowercase ##### removing both puctuation and numbers from our text #### using Textstem library, we are lemmatizing words to their root forms instead of stemming #########

dtm_1_3 <- CreateDtm(doc_vec = df$story, # character vector of documents
                     doc_names = df$id, # document names
                     ngram_window = c(1,3), # minimum and maximum n-gram length
                     stopword_vec = c(tm::stopwords("english"), # stopwords from tm
                                      tm::stopwords("french"), # stopwords from tm
                                      tm::stopwords("spanish"), # stopwords from tm
                                      tm::stopwords("SMART")), # this is the default value
                     lower = TRUE, # lowercase - this is the default value
                     remove_punctuation = TRUE, # punctuation - this is the default
                     remove_numbers = TRUE, # numbers - this is the default
                     verbose = FALSE, # Turn off status bar for this demo,
                     stem_lemma_function = function(x) textstem::lemmatize_words(x),
                     cpus = 20) # default is all available cpus on the system


# Filter rare words
dim(dtm_1_3)
dtm_1_3 <- dtm_1_3[ , colSums(dtm_1_3 > 0) > 50 ]
dim(dtm_1_3)

max_num = nrow(dtm_1_3) * 0.3
max_num
dtm_1_3 <- dtm_1_3[ , colSums(dtm_1_3 > 0) <= max_num ]
dim(dtm_1_3)

# Capture how long each document is
df$doc_lengths = rowSums(dtm_1_3)

### identify frequent terms and highest 50 IDFs ###
tf_mat <- TermDocFreq(dtm = dtm_1_3)
head(tf_mat[ order(tf_mat$term_freq, decreasing = TRUE) , ], 50)
# write.csv(tf_mat, "term_frequency.csv")
head(tf_mat[ order(tf_mat$idf, decreasing = TRUE) , ], 50)

# look at the most frequent bigrams
tf_bigrams <- tf_mat[ stringr::str_detect(tf_mat$term, "_") , ]
head(tf_bigrams[ order(tf_bigrams$term_freq, decreasing = TRUE) , ], 50)
write.csv(head(tf_bigrams[ order(tf_bigrams$term_freq, decreasing = TRUE) , ], 50), "1.3_kiva_bi_grams.csv")

# look at the most frequent trigrams
tf_trigrams <- tf_mat[ stringr::str_detect(tf_mat$term, "_") , ]
head(tf_trigrams[ order(tf_trigrams$term_freq, decreasing = TRUE) , ], 50)
write.csv(head(tf_trigrams[ order(tf_trigrams$term_freq, decreasing = TRUE) , ], 50), "1.3_kiva_tri_grams.csv")


##### fitting an LDA Model #####

twenty_model <- FitLdaModel(dtm = dtm_1_3, 
                            k = 20, 
                            iterations = 500, # recommend a larger value, 500 or more
                            burnin = 180,
                            optimize_alpha = TRUE,
                            calc_likelihood = TRUE,
                            calc_coherence = TRUE,
                            calc_r2 = TRUE,
                            alpha = 0.1, # this is the default value
                            beta = 0.05, # this is the default value
                            cpus = 21) # Note, this is for a big machine  

plot(twenty_model$log_likelihood, type = "l")


twenty_model$top_terms <- GetTopTerms(phi = twenty_model$phi, M = 8)
head(twenty_model$top_terms) 

# probabilistic coherence, a measure of topic quality
# this measure can be used with any topic model, not just probabilistic ones
twenty_model$coherence <- CalcProbCoherence(phi = twenty_model$phi, dtm = dtm_1_3, M = 5)

# Get the prevalence of each topic
# You can make this discrete by applying a threshold, say 0.05, for
# topics in/out of docuemnts. 
twenty_model$prevalence <- colSums(twenty_model$theta) / sum(twenty_model$theta) * 100


# textmineR has a naive topic labeling tool based on probable bigrams
twenty_model$labels <- LabelTopics(assignments = twenty_model$theta > 0.20, 
                                   dtm = dtm_1_3,
                                   M = 2)

head(twenty_model$labels)


# put them together, with coherence into a summary table
twenty_model$summary <- data.frame(topic = rownames(twenty_model$phi),
                                   label = twenty_model$labels,
                                   coherence = round(twenty_model$coherence, 2),
                                   prevalence = round(twenty_model$prevalence,1),
                                   top_terms = apply(twenty_model$top_terms, 2, function(x){
                                     paste(x, collapse = ", ")
                                   }),
                                   stringsAsFactors = FALSE)

twenty_model$summary %>%
  arrange(-prevalence)

write.csv(twenty_model$top_terms, file = "1.3_20_Topics_kiva.csv")

#### Adding topics to data frame #####

df = cbind(df, round(twenty_model$theta, digits=2))
head(df, n=10)


##### Model Building for Classification ######
# Helper function to print the confusion matrix and other performance metrics of the models.
printPerformance = function(pred, actual, positive="Yes") {
  print(caret::confusionMatrix(data=pred, reference=actual, positive=positive, dnn=c("Predicted", "Actual")))
}

#### Data Type casting ####

str(df$status)
str(df)
df$id <- NULL
df <- as.data.frame(unclass(df))#converting all characters variables to factors
str(df)
df$story <- NULL

# Splitting the Data into train and test

set.seed(123) # Set the seed to make it reproducible

train.index <- createDataPartition(df$status, p = .8, list = FALSE)
train <- df[ train.index,]
test  <- df[-train.index,]

actual <- test$status
formula <- status ~ .
positive <- "yes"

# Random Forest with great results
set.seed(123)
model_forest <- randomForest(status~ ., data=train, 
                             importance=TRUE,proximity=TRUE,
                             cutoff = c(0.5, 0.5),type="classification", na.action=na.exclude)
print(model_forest)   
plot(model_forest)
importance(model_forest)

varImpPlot(model_forest)#Shows the decrease in accuracy with the removal of a variables so non Annual removal the accuray can go down by 30% for example

predicted <- predict(model_forest, test, type="class") 

print(sprintf("Accuracy of Random Forest for k = 30:    %.3f", Accuracy(y_true=actual, y_pred=predicted)))
print(sprintf("AUC of Random Forest for k = 30:     %.3f", AUC(y_pred=predicted, y_true=actual)))
print(sprintf("Precision of Random Forest for k = 30:   %.3f", Precision(y_true=actual, y_pred=predicted)))
print(sprintf("Recall of Random Forest for k = 30:      %.3f", Recall(y_true=actual, y_pred=predicted)))
print(sprintf("F1 Score of Random Forest for k = 30:    %.3f", F1_Score(predicted, actual)))
print(sprintf("Sensitivity of Random Forest for k = 30: %.3f", Sensitivity(y_true=actual, y_pred=predicted)))
print(sprintf("Specificity of Random Forest for k = 30: %.3f", Specificity(y_true=predicted, y_pred=actual)))

# Naive Bayes...Accuracy of 0.73 not very good
set.seed(123)

nb_fit = naiveBayes(status ~ ., data=train)
nb_fit

NB_pred = predict(nb_fit, test, type="class") 
print(sprintf("Accuracy of NB for k = 30:    %.3f", Accuracy(y_true=actual, y_pred=NB_pred)))
print(sprintf("Precision of NB for k = 30:   %.3f", Precision(y_true=actual, y_pred=NB_pred)))
print(sprintf("F1 Score of NB for k = 30:    %.3f", F1_Score(NB_pred, actual)))
print(sprintf("Sensitivityof NB for k = 30: %.3f", Sensitivity(y_true=actual, y_pred=NB_pred)))
print(sprintf("Specificity of NB for k = 30: %.3f", Specificity(y_true=NB_pred, y_pred=actual)))

# Decision Tree
set.seed(123)

tree <- rpart(status ~ ., method="class", data=train)
tree_pred = predict(tree, test, type="class") 
print(sprintf("Accuracy of DT for k = 30:    %.3f", Accuracy(y_true=actual, y_pred=tree_pred)))
print(sprintf("Precision of DT for k = 30:   %.3f", Precision(y_true=actual, y_pred=tree_pred)))
print(sprintf("F1 Score of DT for k = 30:    %.3f", F1_Score(tree_pred, actual)))
print(sprintf("Sensitivity of DT for k = 30: %.3f", Sensitivity(y_true=actual, y_pred=tree_pred)))
print(sprintf("Specificity of DT for k = 30: %.3f", Specificity(y_true=tree_pred, y_pred=actual)))

#SVM...this is the second best
set.seed(123)
svm_fit <- svm(status ~ ., data = train)
summary(svm_fit)

svm_pred <- predict(svm_fit, test) 

print(sprintf("Accuracy of SVM for k = 30:    %.3f", Accuracy(y_true=actual, y_pred=svm_pred)))
print(sprintf("Precision of SVM for k = 30:   %.3f", Precision(y_true=actual, y_pred=svm_pred)))
print(sprintf("F1 Score of SVM for k = 30:    %.3f", F1_Score(svm_pred, actual)))
print(sprintf("Sensitivity of SVM for k = 30: %.3f", Sensitivity(y_true=actual, y_pred=svm_pred)))
print(sprintf("Specificity of SVM for k = 30: %.3f", Specificity(y_true=svm_pred, y_pred=actual)))


#kknn
set.seed(123)
kknn_fit = train.kknn(status ~ ., train, kmax=7)
summary(kknn_fit)

kknn_pred = predict(kknn_fit, test)

print(sprintf("Accuracy of KNN for k = 30:    %.3f", Accuracy(y_true=actual, y_pred=kknn_pred)))
print(sprintf("Precision of KNN for k = 30:   %.3f", Precision(y_true=actual, y_pred=kknn_pred)))
print(sprintf("F1 Score of KNN for k = 30:    %.3f", F1_Score(kknn_pred, actual)))
print(sprintf("Sensitivity of KNN for k = 30: %.3f", Sensitivity(y_true=actual, y_pred=kknn_pred)))
print(sprintf("Specificity of KNN for k = 30: %.3f", Specificity(y_true=kknn_pred, y_pred=actual)))

# #NN
# install.packages("nnet")
# library(nnet)
# set.seed(123)
# 
# train_nn <- (train)
# test_nn <- test
# str(train_nn)
# 
# train_nn$status <- as.numeric(train_nn$status)
# train_nn$sector <- as.numeric(train_nn$sector)
# train_nn$country <- as.numeric(train_nn$country)
# train_nn$gender <- as.numeric(train_nn$gender)
# train_nn$countrygender <- as.numeric(as.factor(as.character(train_nn$countrygender)))
# train_nn$CountrySector <- as.numeric(as.factor(as.character(train_nn$CountrySector)))
# train_nn$nonpayment <- as.numeric (train_nn$nonpayment)
# 
# test_nn$status <- as.numeric(test_nn$status)
# test_nn$sector <- as.numeric(test_nn$sector)
# test_nn$country <- as.numeric(test_nn$country)
# test_nn$gender <- as.numeric(test_nn$gender)
# test_nn$countrygender <- as.numeric(as.factor(as.character(test_nn$countrygender)))
# test_nn$CountrySector <- as.numeric(as.factor(as.character(test_nn$CountrySector)))
# test_nn$nonpayment <- as.numeric (test_nn$nonpayment)
# str(train)
# ##start Neural Network analysis
# 
# my.grid <- expand.grid( .size = c(1,2,4),.decay = c(0.25,1,2)) # Tuning grid for Neural Net
# 
# model_NN <- train(status~ ., data = train_nn, method = "nnet", tuneGrid = my.grid, trace = TRUE, na.action =na.omit)
# 
# plot(model_NN) #Visualize the relationship between the number of layers, decay and accuracy
# NN_prediction<-predict(model_NN, test_nn) #Predict classification 
# print(sprintf("Accuracy of NN for k = 30:    %.3f", Accuracy(y_true=test_nn$status, y_pred=NN_prediction)))
# print(sprintf("Precision of NN for k = 30:   %.3f", Precision(y_true=test_nn$status, y_pred=NN_prediction)))
# print(sprintf("F1 Score of NN for k = 30:    %.3f", F1_Score(NN_prediction, test_nn$status)))
# print(sprintf("Sensitivity of NN for k = 30: %.3f", Sensitivity(y_true=test_nn$status, y_pred=NN_prediction)))
# print(sprintf("Specificity of NN for k = 30: %.3f", Specificity(y_true=NN_prediction, y_pred=test_nn$status)))


### removing topics from this part
topics <- cbind(topics, df[,14:33])
df[,14:33] <- NULL


##### Topic Modelling with Sensitive Study ###############

#### With k = 40 ##########
##### fitting an LDA Model #####

forty_model <- FitLdaModel(dtm = dtm_1_3, 
                         k = 40, 
                         iterations = 500, # recommend a larger value, 500 or more
                         burnin = 180,
                         optimize_alpha = TRUE,
                         calc_likelihood = TRUE,
                         calc_coherence = TRUE,
                         calc_r2 = TRUE,
                         alpha = 0.1, # this is the default value
                         beta = 0.05, # this is the default value
                         cpus = 21) # Note, this is for a big machine  

plot(forty_model$log_likelihood, type = "l")


forty_model$top_terms <- GetTopTerms(phi = forty_model$phi, M = 8)
head(forty_model$top_terms) 

# probabilistic coherence, a measure of topic quality
# this measure can be used with any topic model, not just probabilistic ones
forty_model$coherence <- CalcProbCoherence(phi = forty_model$phi, dtm = dtm_1_3, M = 5)

# Get the prevalence of each topic
# You can make this discrete by applying a threshold, say 0.05, for
# topics in/out of docuemnts. 
forty_model$prevalence <- colSums(forty_model$theta) / sum(forty_model$theta) * 100


# textmineR has a naive topic labeling tool based on probable bigrams
forty_model$labels <- LabelTopics(assignments = forty_model$theta > 0.20, 
                                dtm = dtm_1_3,
                                M = 2)

head(forty_model$labels)


# put them together, with coherence into a summary table
forty_model$summary <- data.frame(topic = rownames(forty_model$phi),
                                label = forty_model$labels,
                                coherence = round(forty_model$coherence, 2),
                                prevalence = round(forty_model$prevalence,1),
                                top_terms = apply(forty_model$top_terms, 2, function(x){
                                  paste(x, collapse = ", ")
                                }),
                                stringsAsFactors = FALSE)

forty_model$summary %>%
  arrange(-prevalence)

write.csv(forty_model$top_terms, file = "1.3_30_Topics_kiva.csv")

#### Adding topics to data frame #####

df = cbind(df, round(forty_model$theta, digits=2))
head(df, n=10)

####################################################################
########## Model Building for k = 10 ###################

# Splitting the Data into train and test

set.seed(123) # Set the seed to make it reproducible

train.index <- createDataPartition(df$status, p = .8, list = FALSE)
train <- df[ train.index,]
test  <- df[-train.index,]

actual <- test$status
formula <- status ~ .
positive <- "yes"

# Random Forest
set.seed(123)
model_forest <- randomForest(status~ ., data=train, 
                             importance=TRUE,proximity=TRUE,
                             cutoff = c(0.5, 0.5),type="classification", na.action=na.exclude)
print(model_forest)   
plot(model_forest)
importance(model_forest)

varImpPlot(model_forest)#Shows the decrease in accuracy with the removal of a variables so non Annual removal the accuray can go down by 30% for example

predicted <- predict(model_forest, test, type="class") 

print(sprintf("Accuracy of Random Forest for k = 40:    %.3f", Accuracy(y_true=actual, y_pred=predicted)))
print(sprintf("AUC of Random Forest for k = 40:     %.3f", AUC(y_pred=predicted, y_true=actual)))
print(sprintf("Precision of Random Forest for k = 40:   %.3f", Precision(y_true=actual, y_pred=predicted)))
print(sprintf("Recall of Random Forest for k = 40:      %.3f", Recall(y_true=actual, y_pred=predicted)))
print(sprintf("F1 Score of Random Forest for k = 40:    %.3f", F1_Score(predicted, actual)))
print(sprintf("Sensitivity of Random Forest for k = 40: %.3f", Sensitivity(y_true=actual, y_pred=predicted)))
print(sprintf("Specificity of Random Forest for k = 40: %.3f", Specificity(y_true=predicted, y_pred=actual)))

# Naive Bayes
set.seed(123)

nb_fit = naiveBayes(status ~ ., data=train)
nb_fit

NB_pred = predict(nb_fit, test, type="class") 
print(sprintf("Accuracy of NB for k = 40:    %.3f", Accuracy(y_true=actual, y_pred=NB_pred)))
print(sprintf("Precision of NB for k = 40:   %.3f", Precision(y_true=actual, y_pred=NB_pred)))
print(sprintf("F1 Score of NB for k = 40:    %.3f", F1_Score(NB_pred, actual)))
print(sprintf("Sensitivityof NB for k = 40: %.3f", Sensitivity(y_true=actual, y_pred=NB_pred)))
print(sprintf("Specificity of NB for k = 40: %.3f", Specificity(y_true=NB_pred, y_pred=actual)))

# Decision Tree
set.seed(123)

tree <- rpart(status ~ ., method="class", data=train)
tree_pred = predict(tree, test, type="class") 
print(sprintf("Accuracy of DT for k = 40:    %.3f", Accuracy(y_true=actual, y_pred=tree_pred)))
print(sprintf("Precision of DT for k = 40:   %.3f", Precision(y_true=actual, y_pred=tree_pred)))
print(sprintf("F1 Score of DT for k = 40:    %.3f", F1_Score(tree_pred, actual)))
print(sprintf("Sensitivity of DT for k = 40: %.3f", Sensitivity(y_true=actual, y_pred=tree_pred)))
print(sprintf("Specificity of DT for k = 40: %.3f", Specificity(y_true=tree_pred, y_pred=actual)))

#SVM
set.seed(123)
svm_fit <- svm(status ~ ., data = train)
summary(svm_fit)

svm_pred <- predict(svm_fit, test) 

print(sprintf("Accuracy of SVM for k = 40:    %.3f", Accuracy(y_true=actual, y_pred=svm_pred)))
print(sprintf("Precision of SVM for k = 40:   %.3f", Precision(y_true=actual, y_pred=svm_pred)))
print(sprintf("F1 Score of SVM for k = 40:    %.3f", F1_Score(svm_pred, actual)))
print(sprintf("Sensitivity of SVM for k = 40: %.3f", Sensitivity(y_true=actual, y_pred=svm_pred)))
print(sprintf("Specificity of SVM for k = 40: %.3f", Specificity(y_true=svm_pred, y_pred=actual)))


#kknn
set.seed(123)
kknn_fit = train.kknn(status ~ ., train, kmax=7)
summary(kknn_fit)

kknn_pred = predict(kknn_fit, test)

print(sprintf("Accuracy of KNN for k = 40:    %.3f", Accuracy(y_true=actual, y_pred=kknn_pred)))
print(sprintf("Precision of KNN for k = 40:   %.3f", Precision(y_true=actual, y_pred=kknn_pred)))
print(sprintf("F1 Score of KNN for k = 40:    %.3f", F1_Score(kknn_pred, actual)))
print(sprintf("Sensitivity of KNN for k = 40: %.3f", Sensitivity(y_true=actual, y_pred=kknn_pred)))
print(sprintf("Specificity of KNN for k = 40: %.3f", Specificity(y_true=kknn_pred, y_pred=actual)))

# #NN.
# install.packages("nnet")
# library(nnet)
# set.seed(123)
# 
# train_nn <- (train)
# test_nn <- test
# str(train_nn)
# 
# train_nn$status <- as.numeric(train_nn$status)
# train_nn$sector <- as.numeric(train_nn$sector)
# train_nn$country <- as.numeric(train_nn$country)
# train_nn$gender <- as.numeric(train_nn$gender)
# train_nn$countrygender <- as.numeric(as.factor(as.character(train_nn$countrygender)))
# train_nn$CountrySector <- as.numeric(as.factor(as.character(train_nn$CountrySector)))
# train_nn$nonpayment <- as.numeric (train_nn$nonpayment)
# 
# test_nn$status <- as.numeric(test_nn$status)
# test_nn$sector <- as.numeric(test_nn$sector)
# test_nn$country <- as.numeric(test_nn$country)
# test_nn$gender <- as.numeric(test_nn$gender)
# test_nn$countrygender <- as.numeric(as.factor(as.character(test_nn$countrygender)))
# test_nn$CountrySector <- as.numeric(as.factor(as.character(test_nn$CountrySector)))
# test_nn$nonpayment <- as.numeric (test_nn$nonpayment)
# str(train)
# ##start Neural Network analysis
# 
# my.grid <- expand.grid( .size = c(1,2,4),.decay = c(0.25,1,2)) # Tuning grid for Neural Net
# 
# model_NN <- train(status~ ., data = train_nn, method = "nnet", tuneGrid = my.grid, trace = TRUE, na.action =na.omit)
# 
# plot(model_NN) #Visualize the relationship between the number of layers, decay and accuracy
# NN_prediction<-predict(model_NN, test_nn) #Predict classification 
# print(sprintf("Accuracy of NN for k = 10:    %.3f", Accuracy(y_true=test_nn$status, y_pred=NN_prediction)))
# print(sprintf("Precision of NN for k = 10:   %.3f", Precision(y_true=test_nn$status, y_pred=NN_prediction)))
# print(sprintf("F1 Score of NN for k = 10:    %.3f", F1_Score(NN_prediction, test_nn$status)))
# print(sprintf("Sensitivity of NN for k = 10: %.3f", Sensitivity(y_true=test_nn$status, y_pred=NN_prediction)))
# print(sprintf("Specificity of NN for k = 10: %.3f", Specificity(y_true=NN_prediction, y_pred=test_nn$status)))

### removing topics from this part
topics <- cbind(topics, df[,14:23])
### removing topics from this part
df[,14:23] <- NULL


##### Topic Modelling with Sensitive Study ###############

#### With k = 15 ##########

##### fitting an LDA Model #####

fifteen_model <- FitLdaModel(dtm = dtm_1_3, 
                             k = 15, 
                             iterations = 500, # recommend a larger value, 500 or more
                             burnin = 180,
                             optimize_alpha = TRUE,
                             calc_likelihood = TRUE,
                             calc_coherence = TRUE,
                             calc_r2 = TRUE,
                             alpha = 0.1, # this is the default value
                             beta = 0.05, # this is the default value
                             cpus = 21) # Note, this is for a big machine  

plot(fifteen_model$log_likelihood, type = "l")


fifteen_model$top_terms <- GetTopTerms(phi = fifteen_model$phi, M = 8)
head(fifteen_model$top_terms) 

# probabilistic coherence, a measure of topic quality
# this measure can be used with any topic model, not just probabilistic ones
fifteen_model$coherence <- CalcProbCoherence(phi = fifteen_model$phi, dtm = dtm_1_3, M = 5)

# Get the prevalence of each topic
# You can make this discrete by applying a threshold, say 0.05, for
# topics in/out of docuemnts. 
fifteen_model$prevalence <- colSums(fifteen_model$theta) / sum(fifteen_model$theta) * 100


# textmineR has a naive topic labeling tool based on probable bigrams
fifteen_model$labels <- LabelTopics(assignments = fifteen_model$theta > 0.20, 
                                    dtm = dtm_1_3,
                                    M = 2)

head(fifteen_model$labels)


# put them together, with coherence into a summary table
fifteen_model$summary <- data.frame(topic = rownames(fifteen_model$phi),
                                    label = fifteen_model$labels,
                                    coherence = round(fifteen_model$coherence, 2),
                                    prevalence = round(fifteen_model$prevalence,1),
                                    top_terms = apply(fifteen_model$top_terms, 2, function(x){
                                      paste(x, collapse = ", ")
                                    }),
                                    stringsAsFactors = FALSE)

fifteen_model$summary %>%
  arrange(-prevalence)

write.csv(fifteen_model$top_terms, file = "2.2_15_Topics_kiva.csv")

#### Adding topics to data frame #####

df = cbind(df, round(fifteen_model$theta, digits=2))
head(df, n=10)

####################################################################
########## Model Building for k = 15 ###################

# Splitting the Data into train and test

set.seed(123) # Set the seed to make it reproducible

train.index <- createDataPartition(df$status, p = .8, list = FALSE)
train <- df[ train.index,]
test  <- df[-train.index,]

actual <- test$status
formula <- status ~ .
positive <- "yes"

# Random Forest with great results
set.seed(123)
model_forest <- randomForest(status~ ., data=train, 
                             importance=TRUE,proximity=TRUE,
                             cutoff = c(0.5, 0.5),type="classification", na.action=na.exclude)
print(model_forest)   
plot(model_forest)
importance(model_forest)

varImpPlot(model_forest)#Shows the decrease in accuracy with the removal of a variables so non Annual removal the accuray can go down by 30% for example

predicted <- predict(model_forest, test, type="class") 

print(sprintf("Accuracy of Random Forest for k = 15:    %.3f", Accuracy(y_true=actual, y_pred=predicted)))
print(sprintf("AUC of Random Forest for k = 15:     %.3f", AUC(y_pred=predicted, y_true=actual)))
print(sprintf("Precision of Random Forest for k = 15:   %.3f", Precision(y_true=actual, y_pred=predicted)))
print(sprintf("Recall of Random Forest for k = 15:      %.3f", Recall(y_true=actual, y_pred=predicted)))
print(sprintf("F1 Score of Random Forest for k = 15:    %.3f", F1_Score(predicted, actual)))
print(sprintf("Sensitivity of Random Forest for k = 15: %.3f", Sensitivity(y_true=actual, y_pred=predicted)))
print(sprintf("Specificity of Random Forest for k = 15: %.3f", Specificity(y_true=predicted, y_pred=actual)))

# Naive Bayes
set.seed(123)

nb_fit = naiveBayes(status ~ ., data=train)
nb_fit

NB_pred = predict(nb_fit, test, type="class") 
print(sprintf("Accuracy of NB for k = 15:    %.3f", Accuracy(y_true=actual, y_pred=NB_pred)))
print(sprintf("Precision of NB for k = 15:   %.3f", Precision(y_true=actual, y_pred=NB_pred)))
print(sprintf("F1 Score of NB for k = 15:    %.3f", F1_Score(NB_pred, actual)))
print(sprintf("Sensitivityof NB for k = 15: %.3f", Sensitivity(y_true=actual, y_pred=NB_pred)))
print(sprintf("Specificity of NB for k = 15: %.3f", Specificity(y_true=NB_pred, y_pred=actual)))

# Decision Tree
set.seed(123)

tree <- rpart(status ~ ., method="class", data=train)
tree_pred = predict(tree, test, type="class") 
print(sprintf("Accuracy of DT for k = 15:    %.3f", Accuracy(y_true=actual, y_pred=tree_pred)))
print(sprintf("Precision of DT for k = 15:   %.3f", Precision(y_true=actual, y_pred=tree_pred)))
print(sprintf("F1 Score of DT for k = 15:    %.3f", F1_Score(tree_pred, actual)))
print(sprintf("Sensitivity of DT for k = 15: %.3f", Sensitivity(y_true=actual, y_pred=tree_pred)))
print(sprintf("Specificity of DT for k = 15: %.3f", Specificity(y_true=tree_pred, y_pred=actual)))

#SVM.
set.seed(123)
svm_fit <- svm(status ~ ., data = train)
summary(svm_fit)

svm_pred <- predict(svm_fit, test) 

print(sprintf("Accuracy of SVM for k = 15:    %.3f", Accuracy(y_true=actual, y_pred=svm_pred)))
print(sprintf("Precision of SVM for k = 15:   %.3f", Precision(y_true=actual, y_pred=svm_pred)))
print(sprintf("F1 Score of SVM for k = 15:    %.3f", F1_Score(svm_pred, actual)))
print(sprintf("Sensitivity of SVM for k = 15: %.3f", Sensitivity(y_true=actual, y_pred=svm_pred)))
print(sprintf("Specificity of SVM for k = 15: %.3f", Specificity(y_true=svm_pred, y_pred=actual)))


#kknn.
set.seed(123)

kknn_fit = train.kknn(status ~ ., train, kmax=7)
summary(kknn_fit)

kknn_pred = predict(kknn_fit, test)

print(sprintf("Accuracy of KNN for k = 15:    %.3f", Accuracy(y_true=actual, y_pred=kknn_pred)))
print(sprintf("Precision of KNN for k = 15:   %.3f", Precision(y_true=actual, y_pred=kknn_pred)))
print(sprintf("F1 Score of KNN for k = 15:    %.3f", F1_Score(kknn_pred, actual)))
print(sprintf("Sensitivity of KNN for k = 15: %.3f", Sensitivity(y_true=actual, y_pred=kknn_pred)))
print(sprintf("Specificity of KNN for k = 15: %.3f", Specificity(y_true=kknn_pred, y_pred=actual)))

# #NN...
# install.packages("nnet")
# library(nnet)
# set.seed(123)
# 
# train_nn <- (train)
# test_nn <- test
# str(train_nn)
# 
# train_nn$status <- as.numeric(train_nn$status)
# train_nn$sector <- as.numeric(train_nn$sector)
# train_nn$country <- as.numeric(train_nn$country)
# train_nn$gender <- as.numeric(train_nn$gender)
# train_nn$countrygender <- as.numeric(as.factor(as.character(train_nn$countrygender)))
# train_nn$CountrySector <- as.numeric(as.factor(as.character(train_nn$CountrySector)))
# train_nn$nonpayment <- as.numeric (train_nn$nonpayment)
# 
# test_nn$status <- as.numeric(test_nn$status)
# test_nn$sector <- as.numeric(test_nn$sector)
# test_nn$country <- as.numeric(test_nn$country)
# test_nn$gender <- as.numeric(test_nn$gender)
# test_nn$countrygender <- as.numeric(as.factor(as.character(test_nn$countrygender)))
# test_nn$CountrySector <- as.numeric(as.factor(as.character(test_nn$CountrySector)))
# test_nn$nonpayment <- as.numeric (test_nn$nonpayment)
# str(train)
# ##start Neural Network analysis
# 
# my.grid <- expand.grid( .size = c(1,2,4),.decay = c(0.25,1,2)) # Tuning grid for Neural Net
# 
# model_NN <- train(status~ ., data = train_nn, method = "nnet", tuneGrid = my.grid, trace = TRUE, na.action =na.omit)
# 
# plot(model_NN) #Visualize the relationship between the number of layers, decay and accuracy
# NN_prediction<-predict(model_NN, test_nn) #Predict classification 
# print(sprintf("Accuracy of NN for k = 15:    %.3f", Accuracy(y_true=test_nn$status, y_pred=NN_prediction)))
# print(sprintf("Precision of NN for k = 15:   %.3f", Precision(y_true=test_nn$status, y_pred=NN_prediction)))
# print(sprintf("F1 Score of NN for k = 15:    %.3f", F1_Score(NN_prediction, test_nn$status)))
# print(sprintf("Sensitivity of NN for k = 15: %.3f", Sensitivity(y_true=test_nn$status, y_pred=NN_prediction)))
# print(sprintf("Specificity of NN for k = 15: %.3f", Specificity(y_true=NN_prediction, y_pred=test_nn$status)))

### removing topics from this part
topics <- cbind(topics, df[,14:28])
### removing topics from this part
df[,14:28] <- NULL


##### Topic Modelling with Sensitive Study ###############

#### With k = 25 ##########

##### fitting an LDA Model #####

twentyfive_model <- FitLdaModel(dtm = dtm_1_3, 
                                k = 25, 
                                iterations = 500, # recommend a larger value, 500 or more
                                burnin = 180,
                                optimize_alpha = TRUE,
                                calc_likelihood = TRUE,
                                calc_coherence = TRUE,
                                calc_r2 = TRUE,
                                alpha = 0.1, # this is the default value
                                beta = 0.05, # this is the default value
                                cpus = 21) # Note, this is for a big machine  

plot(twentyfive_model$log_likelihood, type = "l")

twentyfive_model$top_terms <- GetTopTerms(phi = twentyfive_model$phi, M = 8)
head(twentyfive_model$top_terms) 

# probabilistic coherence, a measure of topic quality
# this measure can be used with any topic model, not just probabilistic ones
twentyfive_model$coherence <- CalcProbCoherence(phi = twentyfive_model$phi, dtm = dtm_1_3, M = 5)

# Get the prevalence of each topic
# You can make this discrete by applying a threshold, say 0.05, for
# topics in/out of docuemnts. 
twentyfive_model$prevalence <- colSums(twentyfive_model$theta) / sum(twentyfive_model$theta) * 100


# textmineR has a naive topic labeling tool based on probable bigrams
twentyfive_model$labels <- LabelTopics(assignments = twentyfive_model$theta > 0.20, 
                                       dtm = dtm_1_3,
                                       M = 2)

head(twentyfive_model$labels)


# put them together, with coherence into a summary table
twentyfive_model$summary <- data.frame(topic = rownames(twentyfive_model$phi),
                                       label = twentyfive_model$labels,
                                       coherence = round(twentyfive_model$coherence, 2),
                                       prevalence = round(twentyfive_model$prevalence,1),
                                       top_terms = apply(twentyfive_model$top_terms, 2, function(x){
                                         paste(x, collapse = ", ")
                                       }),
                                       stringsAsFactors = FALSE)

twentyfive_model$summary %>%
  arrange(-prevalence)

write.csv(twentyfive_model$top_terms, file = "2.2_25_Topics_kiva.csv")

#### Adding topics to data frame #####

df = cbind(df, round(twentyfive_model$theta, digits=2))
head(df, n=10)

####################################################################
########## Model Building for k = 15 ###################

# Splitting the Data into train and test

set.seed(123) # Set the seed to make it reproducible

train.index <- createDataPartition(df$status, p = .8, list = FALSE)
train <- df[ train.index,]
test  <- df[-train.index,]

actual <- test$status
formula <- status ~ .
positive <- "yes"

# Random Forest with great results
set.seed(123)
model_forest <- randomForest(status~ ., data=train, 
                             importance=TRUE,proximity=TRUE,
                             cutoff = c(0.5, 0.5),type="classification", na.action=na.exclude)
print(model_forest)   
plot(model_forest)
importance(model_forest)

varImpPlot(model_forest)#Shows the decrease in accuracy with the removal of a variables so non Annual removal the accuray can go down by 30% for example

predicted <- predict(model_forest, test, type="class") 

print(sprintf("Accuracy of Random Forest for k = 25:    %.3f", Accuracy(y_true=actual, y_pred=predicted)))
print(sprintf("AUC of Random Forest for k = 25:     %.3f", AUC(y_pred=predicted, y_true=actual)))
print(sprintf("Precision of Random Forest for k = 25:   %.3f", Precision(y_true=actual, y_pred=predicted)))
print(sprintf("Recall of Random Forest for k = 25:      %.3f", Recall(y_true=actual, y_pred=predicted)))
print(sprintf("F1 Score of Random Forest for k = 25:    %.3f", F1_Score(predicted, actual)))
print(sprintf("Sensitivity of Random Forest for k = 25: %.3f", Sensitivity(y_true=actual, y_pred=predicted)))
print(sprintf("Specificity of Random Forest for k = 25: %.3f", Specificity(y_true=predicted, y_pred=actual)))

# Naive Bayes.
set.seed(123)

nb_fit = naiveBayes(status ~ ., data=train)
nb_fit

NB_pred = predict(nb_fit, test, type="class") 
print(sprintf("Accuracy of NB for k = 25:    %.3f", Accuracy(y_true=actual, y_pred=NB_pred)))
print(sprintf("Precision of NB for k = 25:   %.3f", Precision(y_true=actual, y_pred=NB_pred)))
print(sprintf("F1 Score of NB for k = 25:    %.3f", F1_Score(NB_pred, actual)))
print(sprintf("Sensitivityof NB for k = 25: %.3f", Sensitivity(y_true=actual, y_pred=NB_pred)))
print(sprintf("Specificity of NB for k = 25: %.3f", Specificity(y_true=NB_pred, y_pred=actual)))

# Decision Tree
set.seed(123)

tree <- rpart(status ~ ., method="class", data=train)
tree_pred = predict(tree, test, type="class") 
print(sprintf("Accuracy of DT for k = 25:    %.3f", Accuracy(y_true=actual, y_pred=tree_pred)))
print(sprintf("Precision of DT for k = 25:   %.3f", Precision(y_true=actual, y_pred=tree_pred)))
print(sprintf("F1 Score of DT for k = 25:    %.3f", F1_Score(tree_pred, actual)))
print(sprintf("Sensitivity of DT for k = 25: %.3f", Sensitivity(y_true=actual, y_pred=tree_pred)))
print(sprintf("Specificity of DT for k = 25: %.3f", Specificity(y_true=tree_pred, y_pred=actual)))

#SVM.
set.seed(123)
svm_fit <- svm(status ~ ., data = train)
summary(svm_fit)

svm_pred <- predict(svm_fit, test) 

print(sprintf("Accuracy of SVM for k = 25:    %.3f", Accuracy(y_true=actual, y_pred=svm_pred)))
print(sprintf("Precision of SVM for k = 25:   %.3f", Precision(y_true=actual, y_pred=svm_pred)))
print(sprintf("F1 Score of SVM for k = 25:    %.3f", F1_Score(svm_pred, actual)))
print(sprintf("Sensitivity of SVM for k = 25: %.3f", Sensitivity(y_true=actual, y_pred=svm_pred)))
print(sprintf("Specificity of SVM for k = 25: %.3f", Specificity(y_true=svm_pred, y_pred=actual)))


#kknn...
set.seed(123)

kknn_fit = train.kknn(status ~ ., train, kmax=7)
summary(kknn_fit)

kknn_pred = predict(kknn_fit, test)

print(sprintf("Accuracy of KNN for k = 25:    %.3f", Accuracy(y_true=actual, y_pred=kknn_pred)))
print(sprintf("Precision of KNN for k = 25:   %.3f", Precision(y_true=actual, y_pred=kknn_pred)))
print(sprintf("F1 Score of KNN for k = 25:    %.3f", F1_Score(kknn_pred, actual)))
print(sprintf("Sensitivity of KNN for k = 25: %.3f", Sensitivity(y_true=actual, y_pred=kknn_pred)))
print(sprintf("Specificity of KNN for k = 25: %.3f", Specificity(y_true=kknn_pred, y_pred=actual)))

# #NN...91%
# install.packages("nnet")
# library(nnet)
# set.seed(123)
# 
# train_nn <- (train)
# test_nn <- test
# str(train_nn)
# 
# train_nn$status <- as.numeric(train_nn$status)
# train_nn$sector <- as.numeric(train_nn$sector)
# train_nn$country <- as.numeric(train_nn$country)
# train_nn$gender <- as.numeric(train_nn$gender)
# train_nn$countrygender <- as.numeric(as.factor(as.character(train_nn$countrygender)))
# train_nn$CountrySector <- as.numeric(as.factor(as.character(train_nn$CountrySector)))
# train_nn$nonpayment <- as.numeric (train_nn$nonpayment)
# 
# test_nn$status <- as.numeric(test_nn$status)
# test_nn$sector <- as.numeric(test_nn$sector)
# test_nn$country <- as.numeric(test_nn$country)
# test_nn$gender <- as.numeric(test_nn$gender)
# test_nn$countrygender <- as.numeric(as.factor(as.character(test_nn$countrygender)))
# test_nn$CountrySector <- as.numeric(as.factor(as.character(test_nn$CountrySector)))
# test_nn$nonpayment <- as.numeric (test_nn$nonpayment)
# str(train)
# ##start Neural Network analysis
# 
# my.grid <- expand.grid( .size = c(1,2,4),.decay = c(0.25,1,2)) # Tuning grid for Neural Net
# 
# model_NN <- train(status~ ., data = train_nn, method = "nnet", tuneGrid = my.grid, trace = TRUE, na.action =na.omit)
# 
# plot(model_NN) #Visualize the relationship between the number of layers, decay and accuracy
# NN_prediction<-predict(model_NN, test_nn) #Predict classification 
# print(sprintf("Accuracy of NN for k = 25:    %.3f", Accuracy(y_true=test_nn$status, y_pred=NN_prediction)))
# print(sprintf("Precision of NN for k = 25:   %.3f", Precision(y_true=test_nn$status, y_pred=NN_prediction)))
# print(sprintf("F1 Score of NN for k = 25:    %.3f", F1_Score(NN_prediction, test_nn$status)))
# print(sprintf("Sensitivity of NN for k = 25: %.3f", Sensitivity(y_true=test_nn$status, y_pred=NN_prediction)))
# print(sprintf("Specificity of NN for k = 25: %.3f", Specificity(y_true=NN_prediction, y_pred=test_nn$status)))

### removing topics from this part
topics <- cbind(topics, df[,14:38])
### removing topics from this part
df[,14:38] <- NULL

##### Topic Modelling with Sensitive Study ###############

#### With k = 30 ##########

##### fitting an LDA Model #####

thirty_model <- FitLdaModel(dtm = dtm_1_3, 
                            k = 30, 
                            iterations = 500, # recommend a larger value, 500 or more
                            burnin = 180,
                            optimize_alpha = TRUE,
                            calc_likelihood = TRUE,
                            calc_coherence = TRUE,
                            calc_r2 = TRUE,
                            alpha = 0.1, # this is the default value
                            beta = 0.05, # this is the default value
                            cpus = 21) # Note, this is for a big machine  



# #NN...91%
# install.packages("nnet")
# library(nnet)
# set.seed(123)
# 
# train_nn <- (train)
# test_nn <- test
# str(train_nn)
# 
# train_nn$status <- as.numeric(train_nn$status)
# train_nn$sector <- as.numeric(train_nn$sector)
# train_nn$country <- as.numeric(train_nn$country)
# train_nn$gender <- as.numeric(train_nn$gender)
# train_nn$countrygender <- as.numeric(as.factor(as.character(train_nn$countrygender)))
# train_nn$CountrySector <- as.numeric(as.factor(as.character(train_nn$CountrySector)))
# train_nn$nonpayment <- as.numeric (train_nn$nonpayment)
# 
# test_nn$status <- as.numeric(test_nn$status)
# test_nn$sector <- as.numeric(test_nn$sector)
# test_nn$country <- as.numeric(test_nn$country)
# test_nn$gender <- as.numeric(test_nn$gender)
# test_nn$countrygender <- as.numeric(as.factor(as.character(test_nn$countrygender)))
# test_nn$CountrySector <- as.numeric(as.factor(as.character(test_nn$CountrySector)))
# test_nn$nonpayment <- as.numeric (test_nn$nonpayment)
# str(train)
# ##start Neural Network analysis
# 
# my.grid <- expand.grid( .size = c(1,2,4),.decay = c(0.25,1,2)) # Tuning grid for Neural Net
# 
# model_NN <- train(status~ ., data = train_nn, method = "nnet", tuneGrid = my.grid, trace = TRUE, na.action =na.omit)
# 
# plot(model_NN) #Visualize the relationship between the number of layers, decay and accuracy
# NN_prediction<-predict(model_NN, test_nn) #Predict classification 
# print(sprintf("Accuracy of NN for k = 30:    %.3f", Accuracy(y_true=test_nn$status, y_pred=NN_prediction)))
# print(sprintf("Precision of NN for k = 30:   %.3f", Precision(y_true=test_nn$status, y_pred=NN_prediction)))
# print(sprintf("F1 Score of NN for k = 30:    %.3f", F1_Score(NN_prediction, test_nn$status)))
# print(sprintf("Sensitivity of NN for k = 30: %.3f", Sensitivity(y_true=test_nn$status, y_pred=NN_prediction)))
# print(sprintf("Specificity of NN for k = 30: %.3f", Specificity(y_true=NN_prediction, y_pred=test_nn$status)))

### removing topics from this part
topics <- cbind(topics,df[,14:43])
### removing topics from this part
df[,14:43] <- NULL



##########################################################################
######### Topic Modelling ################################################
################### LDA #######################

df <- read.csv("D:\\kiva_data.csv", stringsAsFactors=FALSE)

########## Renaming and type setting of attributes within the dataset ###########

df = df %>%
  rename(story = en)
str(df)
df$id = 1:nrow(df)
df$status = as.factor(df$status)
df$sector = as.factor(df$sector)
df$country = as.factor(df$country)
df$gender = as.factor(df$gender)
df$nonpayment = as.factor(df$nonpayment)

########## Remove HTML Tags ###################
df = df %>% 
  mutate(story = gsub("<.*?>", "", story))

########## Retain distinct entries ################

dim(df)
df = distinct(df, story, .keep_all = TRUE)
dim(df)

############## Clustering the applicants based on all attributes but the story #######

cl <- df
cl$X <- NULL
cl$loan_nonpayment <- NULL
cl$id <- NULL
str(cl)
cl$status <- as.numeric(cl$status)
cl$sector <- as.numeric(cl$sector)
cl$story <- NULL
cl$country <- as.numeric(cl$country)
cl$gender <- as.numeric(cl$gender)
cl$countrygender <- as.numeric(as.factor(as.character(cl$countrygender)))
cl$CountrySector <- as.numeric(as.factor(as.character(cl$CountrySector)))
cl$loan_amount <- as.numeric (cl$loan_amount)
cl$nonpayment <- as.numeric (cl$nonpayment)

## Elbow plot
wss <- (nrow(cl)-1)*sum(apply(cl,2,var))

for (i in 1:30) wss[i] <- sum(kmeans(cl, 
                                     centers=i)$withinss)
plot(1:30, wss, type="b", xlab="Number of Clusters",
     ylab="Within groups sum of squares")

## Choose 4 clusters 

clust <- kmeans(cl, centers = 4, n = 4)

head(clust$cluster)
df <- cbind(df, cluster_name = clust$cluster)

#################### Sentiment Detection ######################

Sentiment <- sentiment_by(df$story)
summary(Sentiment$ave_sentiment)

qplot(Sentiment$ave_sentiment, geom = "histogram", binwidth = 0.1, main = "Review Sentiment Histogram")
head(sentiment)
df$ave_sentiment <- Sentiment$ave_sentiment
df$sd_sentiment <- Sentiment$sd
df$word_count <- Sentiment$word_count

rm(Sentiment, cl, clust)
########## Removing text that is not neccessary for topic identification or further analysis #############
dim(df)
df = distinct(df, story, .keep_all = TRUE)
dim(df)

idx = grep("Translated from Spanish", df$story)
df$story[idx[1]]
df$story = gsub("Translated from Spanish.*Kiva volunteer", " ", df$story, ignore.case = TRUE)
grep("Translated from Spanish", df$story)

grep("Mifex offers", df$story)
df$story = gsub("Mifex offers.*www.mifex.org", " ", df$story, ignore.case = TRUE)
grep("Mifex offers", df$story)

df$story = gsub("About KADET:.*within those communities", " ", df$story, ignore.case = TRUE)
df$story = gsub("Disclaimer: Due to recent events in.*making their loan.", " ", df$story, ignore.case = TRUE)



########## Topic Modelling ######################################
################### LDA #######################

####### Create Document Term Matrix ###########
# setting n-gram window to be exactly 1 -- unigrams are necessary for our analysis ########## Eliminating stopwords from our text based on english, french, spanish and SMART distionaries ##### converting all uppercase letters to lowercase ##### removing both puctuation and numbers from our text #### using Textstem library, we are lemmatizing words to their root forms instead of stemming #########

dtm_3_3 <- CreateDtm(doc_vec = df$story, # character vector of documents
                     doc_names = df$id, # document names
                     ngram_window = c(3,3), # minimum and maximum n-gram length
                     stopword_vec = c(tm::stopwords("english"), # stopwords from tm
                                      tm::stopwords("french"), # stopwords from tm
                                      tm::stopwords("spanish"), # stopwords from tm
                                      tm::stopwords("SMART")), # this is the default value
                     lower = TRUE, # lowercase - this is the default value
                     remove_punctuation = TRUE, # punctuation - this is the default
                     remove_numbers = TRUE, # numbers - this is the default
                     verbose = FALSE, # Turn off status bar for this demo,
                     stem_lemma_function = function(x) textstem::lemmatize_words(x),
                     cpus = 20) # default is all available cpus on the system


# Filter rare words
dim(dtm_3_3)
dtm_3_3 <- dtm_3_3[ , colSums(dtm_3_3 > 0) > 50 ]
dim(dtm_3_3)

max_num = nrow(dtm_3_3) * 0.3
max_num
dtm_3_3 <- dtm_3_3[ , colSums(dtm_3_3 > 0) <= max_num ]
dim(dtm_3_3)

# Capture how long each document is
df$doc_lengths = rowSums(dtm_3_3)

### identify frequent terms and highest 50 IDFs ###
tf_mat <- TermDocFreq(dtm = dtm_3_3)
head(tf_mat[ order(tf_mat$term_freq, decreasing = TRUE) , ], 50)
# write.csv(tf_mat, "term_frequency.csv")
head(tf_mat[ order(tf_mat$idf, decreasing = TRUE) , ], 50)

# look at the most frequent bigrams
tf_bigrams <- tf_mat[ stringr::str_detect(tf_mat$term, "_") , ]
head(tf_bigrams[ order(tf_bigrams$term_freq, decreasing = TRUE) , ], 50)
write.csv(head(tf_bigrams[ order(tf_bigrams$term_freq, decreasing = TRUE) , ], 50), "3.3_kiva_bi_grams.csv")

# look at the most frequent trigrams
tf_trigrams <- tf_mat[ stringr::str_detect(tf_mat$term, "_") , ]
head(tf_trigrams[ order(tf_trigrams$term_freq, decreasing = TRUE) , ], 50)
write.csv(head(tf_trigrams[ order(tf_trigrams$term_freq, decreasing = TRUE) , ], 50), "3.3_kiva_tri_grams.csv")


##### fitting an LDA Model #####

twenty_model <- FitLdaModel(dtm = dtm_3_3, 
                            k = 20, 
                            iterations = 500, # recommend a larger value, 500 or more
                            burnin = 180,
                            optimize_alpha = TRUE,
                            calc_likelihood = TRUE,
                            calc_coherence = TRUE,
                            calc_r2 = TRUE,
                            alpha = 0.1, # this is the default value
                            beta = 0.05, # this is the default value
                            cpus = 21) # Note, this is for a big machine  

plot(twenty_model$log_likelihood, type = "l")


twenty_model$top_terms <- GetTopTerms(phi = twenty_model$phi, M = 8)
head(twenty_model$top_terms) 

# probabilistic coherence, a measure of topic quality
# this measure can be used with any topic model, not just probabilistic ones
twenty_model$coherence <- CalcProbCoherence(phi = twenty_model$phi, dtm = dtm_3_3, M = 5)

# Get the prevalence of each topic
# You can make this discrete by applying a threshold, say 0.05, for
# topics in/out of docuemnts. 
twenty_model$prevalence <- colSums(twenty_model$theta) / sum(twenty_model$theta) * 100


# textmineR has a naive topic labeling tool based on probable bigrams
twenty_model$labels <- LabelTopics(assignments = twenty_model$theta > 0.20, 
                                   dtm = dtm_3_3,
                                   M = 2)

head(twenty_model$labels)


# put them together, with coherence into a summary table
twenty_model$summary <- data.frame(topic = rownames(twenty_model$phi),
                                   label = twenty_model$labels,
                                   coherence = round(twenty_model$coherence, 2),
                                   prevalence = round(twenty_model$prevalence,1),
                                   top_terms = apply(twenty_model$top_terms, 2, function(x){
                                     paste(x, collapse = ", ")
                                   }),
                                   stringsAsFactors = FALSE)

twenty_model$summary %>%
  arrange(-prevalence)

write.csv(twenty_model$top_terms, file = "3.3_20_Topics_kiva.csv")

#### Adding topics to data frame #####

df = cbind(df, round(twenty_model$theta, digits=2))
head(df, n=10)

##### Model Building for Classification ######
# Helper function to print the confusion matrix and other performance metrics of the models.
printPerformance = function(pred, actual, positive="Yes") {
  print(caret::confusionMatrix(data=pred, reference=actual, positive=positive, dnn=c("Predicted", "Actual")))
}

#### Data Type casting ####

str(df$status)
str(df)
df$id <- NULL
df <- as.data.frame(unclass(df))#converting all characters variables to factors
str(df)
df$story <- NULL

# Splitting the Data into train and test

set.seed(123) # Set the seed to make it reproducible

train.index <- createDataPartition(df$status, p = .8, list = FALSE)
train <- df[ train.index,]
test  <- df[-train.index,]

actual <- test$status
formula <- status ~ .
positive <- "yes"

# Random Forest with great results
set.seed(123)
model_forest <- randomForest(status~ ., data=train, 
                             importance=TRUE,proximity=TRUE,
                             cutoff = c(0.5, 0.5),type="classification", na.action=na.exclude)
print(model_forest)   
plot(model_forest)
importance(model_forest)

varImpPlot(model_forest)#Shows the decrease in accuracy with the removal of a variables so non Annual removal the accuray can go down by 30% for example

predicted <- predict(model_forest, test, type="class") 

print(sprintf("Accuracy of Random Forest for k = 20:    %.3f", Accuracy(y_true=actual, y_pred=predicted)))
print(sprintf("AUC of Random Forest for k = 20:     %.3f", AUC(y_pred=predicted, y_true=actual)))
print(sprintf("Precision of Random Forest for k = 20:   %.3f", Precision(y_true=actual, y_pred=predicted)))
print(sprintf("Recall of Random Forest for k = 20:      %.3f", Recall(y_true=actual, y_pred=predicted)))
print(sprintf("F1 Score of Random Forest for k = 20:    %.3f", F1_Score(predicted, actual)))
print(sprintf("Sensitivity of Random Forest for k = 20: %.3f", Sensitivity(y_true=actual, y_pred=predicted)))
print(sprintf("Specificity of Random Forest for k = 20: %.3f", Specificity(y_true=predicted, y_pred=actual)))

# Naive Bayes.
set.seed(123)

nb_fit = naiveBayes(status ~ ., data=train)
nb_fit

NB_pred = predict(nb_fit, test, type="class") 
print(sprintf("Accuracy of NB for k = 20:    %.3f", Accuracy(y_true=actual, y_pred=NB_pred)))
print(sprintf("Precision of NB for k = 20:   %.3f", Precision(y_true=actual, y_pred=NB_pred)))
print(sprintf("F1 Score of NB for k = 20:    %.3f", F1_Score(NB_pred, actual)))
print(sprintf("Sensitivityof NB for k = 20: %.3f", Sensitivity(y_true=actual, y_pred=NB_pred)))
print(sprintf("Specificity of NB for k = 20: %.3f", Specificity(y_true=NB_pred, y_pred=actual)))

# Decision Tree
set.seed(123)

tree <- rpart(status ~ ., method="class", data=train)
tree_pred = predict(tree, test, type="class") 
print(sprintf("Accuracy of DT for k = 20:    %.3f", Accuracy(y_true=actual, y_pred=tree_pred)))
print(sprintf("Precision of DT for k = 20:   %.3f", Precision(y_true=actual, y_pred=tree_pred)))
print(sprintf("F1 Score of DT for k = 20:    %.3f", F1_Score(tree_pred, actual)))
print(sprintf("Sensitivity of DT for k = 20: %.3f", Sensitivity(y_true=actual, y_pred=tree_pred)))
print(sprintf("Specificity of DT for k = 20: %.3f", Specificity(y_true=tree_pred, y_pred=actual)))

#SVM.
set.seed(123)
svm_fit <- svm(status ~ ., data = train)
summary(svm_fit)

svm_pred <- predict(svm_fit, test) 

print(sprintf("Accuracy of SVM for k = 20:    %.3f", Accuracy(y_true=actual, y_pred=svm_pred)))
print(sprintf("Precision of SVM for k = 20:   %.3f", Precision(y_true=actual, y_pred=svm_pred)))
print(sprintf("F1 Score of SVM for k = 20:    %.3f", F1_Score(svm_pred, actual)))
print(sprintf("Sensitivity of SVM for k = 20: %.3f", Sensitivity(y_true=actual, y_pred=svm_pred)))
print(sprintf("Specificity of SVM for k = 20: %.3f", Specificity(y_true=svm_pred, y_pred=actual)))


#kknn...
set.seed(123)
kknn_fit = train.kknn(status ~ ., train, kmax=7)
summary(kknn_fit)

kknn_pred = predict(kknn_fit, test)

print(sprintf("Accuracy of KNN for k = 20:    %.3f", Accuracy(y_true=actual, y_pred=kknn_pred)))
print(sprintf("Precision of KNN for k = 20:   %.3f", Precision(y_true=actual, y_pred=kknn_pred)))
print(sprintf("F1 Score of KNN for k = 20:    %.3f", F1_Score(kknn_pred, actual)))
print(sprintf("Sensitivity of KNN for k = 20: %.3f", Sensitivity(y_true=actual, y_pred=kknn_pred)))
print(sprintf("Specificity of KNN for k = 20: %.3f", Specificity(y_true=kknn_pred, y_pred=actual)))

# #NN...91%
# install.packages("nnet")
# library(nnet)
# set.seed(123)
# 
# train_nn <- (train)
# test_nn <- test
# str(train_nn)
# 
# train_nn$status <- as.numeric(train_nn$status)
# train_nn$sector <- as.numeric(train_nn$sector)
# train_nn$country <- as.numeric(train_nn$country)
# train_nn$gender <- as.numeric(train_nn$gender)
# train_nn$countrygender <- as.numeric(as.factor(as.character(train_nn$countrygender)))
# train_nn$CountrySector <- as.numeric(as.factor(as.character(train_nn$CountrySector)))
# train_nn$nonpayment <- as.numeric (train_nn$nonpayment)
# 
# test_nn$status <- as.numeric(test_nn$status)
# test_nn$sector <- as.numeric(test_nn$sector)
# test_nn$country <- as.numeric(test_nn$country)
# test_nn$gender <- as.numeric(test_nn$gender)
# test_nn$countrygender <- as.numeric(as.factor(as.character(test_nn$countrygender)))
# test_nn$CountrySector <- as.numeric(as.factor(as.character(test_nn$CountrySector)))
# test_nn$nonpayment <- as.numeric (test_nn$nonpayment)
# str(train)
# ##start Neural Network analysis
# 
# my.grid <- expand.grid( .size = c(1,2,4),.decay = c(0.25,1,2)) # Tuning grid for Neural Net
# 
# model_NN <- train(status~ ., data = train_nn, method = "nnet", tuneGrid = my.grid, trace = TRUE, na.action =na.omit)
# 
# plot(model_NN) #Visualize the relationship between the number of layers, decay and accuracy
# NN_prediction<-predict(model_NN, test_nn) #Predict classification 
# print(sprintf("Accuracy of NN for k = 20:    %.3f", Accuracy(y_true=test_nn$status, y_pred=NN_prediction)))
# print(sprintf("Precision of NN for k = 20:   %.3f", Precision(y_true=test_nn$status, y_pred=NN_prediction)))
# print(sprintf("F1 Score of NN for k = 20:    %.3f", F1_Score(NN_prediction, test_nn$status)))
# print(sprintf("Sensitivity of NN for k = 20: %.3f", Sensitivity(y_true=test_nn$status, y_pred=NN_prediction)))
# print(sprintf("Specificity of NN for k = 20: %.3f", Specificity(y_true=NN_prediction, y_pred=test_nn$status)))

### removing topics from this part
topics <- cbind(topics,df[,14:33])
df[,14:33] <- NULL


##### Topic Modelling with Sensitive Study ###############

#### With k = 10 ##########
##### fitting an LDA Model #####

ten_model <- FitLdaModel(dtm = dtm_3_3, 
                         k = 10, 
                         iterations = 500, # recommend a larger value, 500 or more
                         burnin = 180,
                         optimize_alpha = TRUE,
                         calc_likelihood = TRUE,
                         calc_coherence = TRUE,
                         calc_r2 = TRUE,
                         alpha = 0.1, # this is the default value
                         beta = 0.05, # this is the default value
                         cpus = 21) # Note, this is for a big machine  

plot(ten_model$log_likelihood, type = "l")


ten_model$top_terms <- GetTopTerms(phi = ten_model$phi, M = 8)
head(ten_model$top_terms) 

# probabilistic coherence, a measure of topic quality
# this measure can be used with any topic model, not just probabilistic ones
ten_model$coherence <- CalcProbCoherence(phi = ten_model$phi, dtm = dtm_3_3, M = 5)

# Get the prevalence of each topic
# You can make this discrete by applying a threshold, say 0.05, for
# topics in/out of docuemnts. 
ten_model$prevalence <- colSums(ten_model$theta) / sum(ten_model$theta) * 100


# textmineR has a naive topic labeling tool based on probable bigrams
ten_model$labels <- LabelTopics(assignments = ten_model$theta > 0.20, 
                                dtm = dtm_3_3,
                                M = 2)

head(ten_model$labels)


# put them together, with coherence into a summary table
ten_model$summary <- data.frame(topic = rownames(ten_model$phi),
                                label = ten_model$labels,
                                coherence = round(ten_model$coherence, 2),
                                prevalence = round(ten_model$prevalence,1),
                                top_terms = apply(ten_model$top_terms, 2, function(x){
                                  paste(x, collapse = ", ")
                                }),
                                stringsAsFactors = FALSE)

ten_model$summary %>%
  arrange(-prevalence)


write.csv(ten_model$top_terms, file = "3.3_10_Topics_kiva.csv")

#### Adding topics to data frame #####

df = cbind(df, round(ten_model$theta, digits=2))
head(df, n=10)

####################################################################
########## Model Building for k = 10 ###################

# Splitting the Data into train and test

set.seed(123) # Set the seed to make it reproducible

train.index <- createDataPartition(df$status, p = .8, list = FALSE)
train <- df[ train.index,]
test  <- df[-train.index,]

actual <- test$status
formula <- status ~ .
positive <- "yes"

# Random Forest with great results
set.seed(123)
model_forest <- randomForest(status~ ., data=train, 
                             importance=TRUE,proximity=TRUE,
                             cutoff = c(0.5, 0.5),type="classification", na.action=na.exclude)
print(model_forest)   
plot(model_forest)
importance(model_forest)

varImpPlot(model_forest)#Shows the decrease in accuracy with the removal of a variables so non Annual removal the accuray can go down by 30% for example

predicted <- predict(model_forest, test, type="class") 

print(sprintf("Accuracy of Random Forest for k = 10:    %.3f", Accuracy(y_true=actual, y_pred=predicted)))
print(sprintf("AUC of Random Forest for k = 10:     %.3f", AUC(y_pred=predicted, y_true=actual)))
print(sprintf("Precision of Random Forest for k = 10:   %.3f", Precision(y_true=actual, y_pred=predicted)))
print(sprintf("Recall of Random Forest for k = 10:      %.3f", Recall(y_true=actual, y_pred=predicted)))
print(sprintf("F1 Score of Random Forest for k = 10:    %.3f", F1_Score(predicted, actual)))
print(sprintf("Sensitivity of Random Forest for k = 10: %.3f", Sensitivity(y_true=actual, y_pred=predicted)))
print(sprintf("Specificity of Random Forest for k = 10: %.3f", Specificity(y_true=predicted, y_pred=actual)))

# Naive Bayes...Accuracy of 0.73 not very good
set.seed(123)

nb_fit = naiveBayes(status ~ ., data=train)
nb_fit

NB_pred = predict(nb_fit, test, type="class") 
print(sprintf("Accuracy of NB for k = 10:    %.3f", Accuracy(y_true=actual, y_pred=NB_pred)))
print(sprintf("Precision of NB for k = 10:   %.3f", Precision(y_true=actual, y_pred=NB_pred)))
print(sprintf("F1 Score of NB for k = 10:    %.3f", F1_Score(NB_pred, actual)))
print(sprintf("Sensitivityof NB for k = 10: %.3f", Sensitivity(y_true=actual, y_pred=NB_pred)))
print(sprintf("Specificity of NB for k = 10: %.3f", Specificity(y_true=NB_pred, y_pred=actual)))

# Decision Tree
set.seed(123)

tree <- rpart(status ~ ., method="class", data=train)
tree_pred = predict(tree, test, type="class") 
print(sprintf("Accuracy of DT for k = 10:    %.3f", Accuracy(y_true=actual, y_pred=tree_pred)))
print(sprintf("Precision of DT for k = 10:   %.3f", Precision(y_true=actual, y_pred=tree_pred)))
print(sprintf("F1 Score of DT for k = 10:    %.3f", F1_Score(tree_pred, actual)))
print(sprintf("Sensitivity of DT for k = 10: %.3f", Sensitivity(y_true=actual, y_pred=tree_pred)))
print(sprintf("Specificity of DT for k = 10: %.3f", Specificity(y_true=tree_pred, y_pred=actual)))

#SVM...this is the second best
set.seed(123)
svm_fit <- svm(status ~ ., data = train)
summary(svm_fit)

svm_pred <- predict(svm_fit, test) 

print(sprintf("Accuracy of SVM for k = 10:    %.3f", Accuracy(y_true=actual, y_pred=svm_pred)))
print(sprintf("Precision of SVM for k = 10:   %.3f", Precision(y_true=actual, y_pred=svm_pred)))
print(sprintf("F1 Score of SVM for k = 10:    %.3f", F1_Score(svm_pred, actual)))
print(sprintf("Sensitivity of SVM for k = 10: %.3f", Sensitivity(y_true=actual, y_pred=svm_pred)))
print(sprintf("Specificity of SVM for k = 10: %.3f", Specificity(y_true=svm_pred, y_pred=actual)))


#kknn....89%
set.seed(123)

kknn_fit = train.kknn(status ~ ., train, kmax=7)
summary(kknn_fit)

kknn_pred = predict(kknn_fit, test)

print(sprintf("Accuracy of KNN for k = 10:    %.3f", Accuracy(y_true=actual, y_pred=kknn_pred)))
print(sprintf("Precision of KNN for k = 10:   %.3f", Precision(y_true=actual, y_pred=kknn_pred)))
print(sprintf("F1 Score of KNN for k = 10:    %.3f", F1_Score(kknn_pred, actual)))
print(sprintf("Sensitivity of KNN for k = 10: %.3f", Sensitivity(y_true=actual, y_pred=kknn_pred)))
print(sprintf("Specificity of KNN for k = 10: %.3f", Specificity(y_true=kknn_pred, y_pred=actual)))

# #NN...91%
# install.packages("nnet")
# library(nnet)
# set.seed(123)
# 
# train_nn <- (train)
# test_nn <- test
# str(train_nn)
# 
# train_nn$status <- as.numeric(train_nn$status)
# train_nn$sector <- as.numeric(train_nn$sector)
# train_nn$country <- as.numeric(train_nn$country)
# train_nn$gender <- as.numeric(train_nn$gender)
# train_nn$countrygender <- as.numeric(as.factor(as.character(train_nn$countrygender)))
# train_nn$CountrySector <- as.numeric(as.factor(as.character(train_nn$CountrySector)))
# train_nn$nonpayment <- as.numeric (train_nn$nonpayment)
# 
# test_nn$status <- as.numeric(test_nn$status)
# test_nn$sector <- as.numeric(test_nn$sector)
# test_nn$country <- as.numeric(test_nn$country)
# test_nn$gender <- as.numeric(test_nn$gender)
# test_nn$countrygender <- as.numeric(as.factor(as.character(test_nn$countrygender)))
# test_nn$CountrySector <- as.numeric(as.factor(as.character(test_nn$CountrySector)))
# test_nn$nonpayment <- as.numeric (test_nn$nonpayment)
# str(train)
# ##start Neural Network analysis
# 
# my.grid <- expand.grid( .size = c(1,2,4),.decay = c(0.25,1,2)) # Tuning grid for Neural Net
# 
# model_NN <- train(status~ ., data = train_nn, method = "nnet", tuneGrid = my.grid, trace = TRUE, na.action =na.omit)
# 
# plot(model_NN) #Visualize the relationship between the number of layers, decay and accuracy
# NN_prediction<-predict(model_NN, test_nn) #Predict classification 
# print(sprintf("Accuracy of NN for k = 10:    %.3f", Accuracy(y_true=test_nn$status, y_pred=NN_prediction)))
# print(sprintf("Precision of NN for k = 10:   %.3f", Precision(y_true=test_nn$status, y_pred=NN_prediction)))
# print(sprintf("F1 Score of NN for k = 10:    %.3f", F1_Score(NN_prediction, test_nn$status)))
# print(sprintf("Sensitivity of NN for k = 10: %.3f", Sensitivity(y_true=test_nn$status, y_pred=NN_prediction)))
# print(sprintf("Specificity of NN for k = 10: %.3f", Specificity(y_true=NN_prediction, y_pred=test_nn$status)))

### removing topics from this part
topics <- cbind(topics, df[,14:23])
### removing topics from this part
df[,14:23] <- NULL

##### Topic Modelling with Sensitive Study ###############

#### With k = 15 ##########

##### fitting an LDA Model #####

fifteen_model <- FitLdaModel(dtm = dtm_3_3, 
                             k = 15, 
                             iterations = 500, # recommend a larger value, 500 or more
                             burnin = 180,
                             optimize_alpha = TRUE,
                             calc_likelihood = TRUE,
                             calc_coherence = TRUE,
                             calc_r2 = TRUE,
                             alpha = 0.1, # this is the default value
                             beta = 0.05, # this is the default value
                             cpus = 21) # Note, this is for a big machine  

plot(fifteen_model$log_likelihood, type = "l")


fifteen_model$top_terms <- GetTopTerms(phi = fifteen_model$phi, M = 8)
head(fifteen_model$top_terms) 

# probabilistic coherence, a measure of topic quality
# this measure can be used with any topic model, not just probabilistic ones
fifteen_model$coherence <- CalcProbCoherence(phi = fifteen_model$phi, dtm = dtm_3_3, M = 5)

# Get the prevalence of each topic
# You can make this discrete by applying a threshold, say 0.05, for
# topics in/out of docuemnts. 
fifteen_model$prevalence <- colSums(fifteen_model$theta) / sum(fifteen_model$theta) * 100


# textmineR has a naive topic labeling tool based on probable bigrams
fifteen_model$labels <- LabelTopics(assignments = fifteen_model$theta > 0.20, 
                                    dtm = dtm_3_3,
                                    M = 2)

head(fifteen_model$labels)


# put them together, with coherence into a summary table
fifteen_model$summary <- data.frame(topic = rownames(fifteen_model$phi),
                                    label = fifteen_model$labels,
                                    coherence = round(fifteen_model$coherence, 2),
                                    prevalence = round(fifteen_model$prevalence,1),
                                    top_terms = apply(fifteen_model$top_terms, 2, function(x){
                                      paste(x, collapse = ", ")
                                    }),
                                    stringsAsFactors = FALSE)

fifteen_model$summary %>%
  arrange(-prevalence)


write.csv(fifteen_model$top_terms, file = "3.3_15_Topics_kiva.csv")

#### Adding topics to data frame #####

df = cbind(df, round(fifteen_model$theta, digits=2))
head(df, n=10)

####################################################################
########## Model Building for k = 15 ###################

# Splitting the Data into train and test

set.seed(123) # Set the seed to make it reproducible

train.index <- createDataPartition(df$status, p = .8, list = FALSE)
train <- df[ train.index,]
test  <- df[-train.index,]

actual <- test$status
formula <- status ~ .
positive <- "yes"

# Random Forest with great results
set.seed(123)
model_forest <- randomForest(status~ ., data=train, 
                             importance=TRUE,proximity=TRUE,
                             cutoff = c(0.5, 0.5),type="classification", na.action=na.exclude)
print(model_forest)   
plot(model_forest)
importance(model_forest)

varImpPlot(model_forest)#Shows the decrease in accuracy with the removal of a variables so non Annual removal the accuray can go down by 30% for example

predicted <- predict(model_forest, test, type="class") 

print(sprintf("Accuracy of Random Forest for k = 15:    %.3f", Accuracy(y_true=actual, y_pred=predicted)))
print(sprintf("AUC of Random Forest for k = 15:     %.3f", AUC(y_pred=predicted, y_true=actual)))
print(sprintf("Precision of Random Forest for k = 15:   %.3f", Precision(y_true=actual, y_pred=predicted)))
print(sprintf("Recall of Random Forest for k = 15:      %.3f", Recall(y_true=actual, y_pred=predicted)))
print(sprintf("F1 Score of Random Forest for k = 15:    %.3f", F1_Score(predicted, actual)))
print(sprintf("Sensitivity of Random Forest for k = 15: %.3f", Sensitivity(y_true=actual, y_pred=predicted)))
print(sprintf("Specificity of Random Forest for k = 15: %.3f", Specificity(y_true=predicted, y_pred=actual)))

# Naive Bayes...Accuracy of 0.73 not very good
set.seed(123)

nb_fit = naiveBayes(status ~ ., data=train)
nb_fit

NB_pred = predict(nb_fit, test, type="class") 
print(sprintf("Accuracy of NB for k = 15:    %.3f", Accuracy(y_true=actual, y_pred=NB_pred)))
print(sprintf("Precision of NB for k = 15:   %.3f", Precision(y_true=actual, y_pred=NB_pred)))
print(sprintf("F1 Score of NB for k = 15:    %.3f", F1_Score(NB_pred, actual)))
print(sprintf("Sensitivityof NB for k = 15: %.3f", Sensitivity(y_true=actual, y_pred=NB_pred)))
print(sprintf("Specificity of NB for k = 15: %.3f", Specificity(y_true=NB_pred, y_pred=actual)))
# Decision Tree

set.seed(123)

tree <- rpart(status ~ ., method="class", data=train)
tree_pred = predict(tree, test, type="class") 
print(sprintf("Accuracy of DT for k = 15:    %.3f", Accuracy(y_true=actual, y_pred=tree_pred)))
print(sprintf("Precision of DT for k = 15:   %.3f", Precision(y_true=actual, y_pred=tree_pred)))
print(sprintf("F1 Score of DT for k = 15:    %.3f", F1_Score(tree_pred, actual)))
print(sprintf("Sensitivity of DT for k = 15: %.3f", Sensitivity(y_true=actual, y_pred=tree_pred)))
print(sprintf("Specificity of DT for k = 15: %.3f", Specificity(y_true=tree_pred, y_pred=actual)))

#SVM...this is the second best

svm_fit <- svm(status ~ ., data = train)
summary(svm_fit)

svm_pred <- predict(svm_fit, test) 

print(sprintf("Accuracy of SVM for k = 15:    %.3f", Accuracy(y_true=actual, y_pred=svm_pred)))
print(sprintf("Precision of SVM for k = 15:   %.3f", Precision(y_true=actual, y_pred=svm_pred)))
print(sprintf("F1 Score of SVM for k = 15:    %.3f", F1_Score(svm_pred, actual)))
print(sprintf("Sensitivity of SVM for k = 15: %.3f", Sensitivity(y_true=actual, y_pred=svm_pred)))
print(sprintf("Specificity of SVM for k = 15: %.3f", Specificity(y_true=svm_pred, y_pred=actual)))


#kknn....89%
set.seed(123)
kknn_fit = train.kknn(status ~ ., train, kmax=7)
summary(kknn_fit)

kknn_pred = predict(kknn_fit, test)

print(sprintf("Accuracy of KNN for k = 15:    %.3f", Accuracy(y_true=actual, y_pred=kknn_pred)))
print(sprintf("Precision of KNN for k = 15:   %.3f", Precision(y_true=actual, y_pred=kknn_pred)))
print(sprintf("F1 Score of KNN for k = 15:    %.3f", F1_Score(kknn_pred, actual)))
print(sprintf("Sensitivity of KNN for k = 15: %.3f", Sensitivity(y_true=actual, y_pred=kknn_pred)))
print(sprintf("Specificity of KNN for k = 15: %.3f", Specificity(y_true=kknn_pred, y_pred=actual)))

# #NN...91%
# install.packages("nnet")
# library(nnet)
# set.seed(123)
# 
# train_nn <- (train)
# test_nn <- test
# str(train_nn)
# 
# train_nn$status <- as.numeric(train_nn$status)
# train_nn$sector <- as.numeric(train_nn$sector)
# train_nn$country <- as.numeric(train_nn$country)
# train_nn$gender <- as.numeric(train_nn$gender)
# train_nn$countrygender <- as.numeric(as.factor(as.character(train_nn$countrygender)))
# train_nn$CountrySector <- as.numeric(as.factor(as.character(train_nn$CountrySector)))
# train_nn$nonpayment <- as.numeric (train_nn$nonpayment)
# 
# test_nn$status <- as.numeric(test_nn$status)
# test_nn$sector <- as.numeric(test_nn$sector)
# test_nn$country <- as.numeric(test_nn$country)
# test_nn$gender <- as.numeric(test_nn$gender)
# test_nn$countrygender <- as.numeric(as.factor(as.character(test_nn$countrygender)))
# test_nn$CountrySector <- as.numeric(as.factor(as.character(test_nn$CountrySector)))
# test_nn$nonpayment <- as.numeric (test_nn$nonpayment)
# str(train)
# ##start Neural Network analysis
# 
# my.grid <- expand.grid( .size = c(1,2,4),.decay = c(0.25,1,2)) # Tuning grid for Neural Net
# 
# model_NN <- train(status~ ., data = train_nn, method = "nnet", tuneGrid = my.grid, trace = TRUE, na.action =na.omit)
# 
# plot(model_NN) #Visualize the relationship between the number of layers, decay and accuracy
# NN_prediction<-predict(model_NN, test_nn) #Predict classification 
# print(sprintf("Accuracy of NN for k = 15:    %.3f", Accuracy(y_true=test_nn$status, y_pred=NN_prediction)))
# print(sprintf("Precision of NN for k = 15:   %.3f", Precision(y_true=test_nn$status, y_pred=NN_prediction)))
# print(sprintf("F1 Score of NN for k = 15:    %.3f", F1_Score(NN_prediction, test_nn$status)))
# print(sprintf("Sensitivity of NN for k = 15: %.3f", Sensitivity(y_true=test_nn$status, y_pred=NN_prediction)))
# print(sprintf("Specificity of NN for k = 15: %.3f", Specificity(y_true=NN_prediction, y_pred=test_nn$status)))

### removing topics from this part
topics <- cbind(topics, df[,14:28])
### removing topics from this part
df[,14:28] <- NULL


##### Topic Modelling with Sensitive Study ###############

#### With k = 25 ##########

##### fitting an LDA Model #####

twentyfive_model <- FitLdaModel(dtm = dtm_3_3, 
                                k = 25, 
                                iterations = 500, # recommend a larger value, 500 or more
                                burnin = 180,
                                optimize_alpha = TRUE,
                                calc_likelihood = TRUE,
                                calc_coherence = TRUE,
                                calc_r2 = TRUE,
                                alpha = 0.1, # this is the default value
                                beta = 0.05, # this is the default value
                                cpus = 21) # Note, this is for a big machine  

plot(twentyfive_model$log_likelihood, type = "l")


twentyfive_model$top_terms <- GetTopTerms(phi = twentyfive_model$phi, M = 8)
head(twentyfive_model$top_terms) 

# probabilistic coherence, a measure of topic quality
# this measure can be used with any topic model, not just probabilistic ones
twentyfive_model$coherence <- CalcProbCoherence(phi = twentyfive_model$phi, dtm = dtm_3_3, M = 5)

# Get the prevalence of each topic
# You can make this discrete by applying a threshold, say 0.05, for
# topics in/out of docuemnts. 
twentyfive_model$prevalence <- colSums(twentyfive_model$theta) / sum(twentyfive_model$theta) * 100


# textmineR has a naive topic labeling tool based on probable bigrams
twentyfive_model$labels <- LabelTopics(assignments = twentyfive_model$theta > 0.20, 
                                       dtm = dtm_3_3,
                                       M = 2)

head(twentyfive_model$labels)


# put them together, with coherence into a summary table
twentyfive_model$summary <- data.frame(topic = rownames(twentyfive_model$phi),
                                       label = twentyfive_model$labels,
                                       coherence = round(twentyfive_model$coherence, 2),
                                       prevalence = round(twentyfive_model$prevalence,1),
                                       top_terms = apply(twentyfive_model$top_terms, 2, function(x){
                                         paste(x, collapse = ", ")
                                       }),
                                       stringsAsFactors = FALSE)

twentyfive_model$summary %>%
  arrange(-prevalence)


write.csv(twentyfive_model$top_terms, file = "3.3_25_Topics_kiva.csv")

#### Adding topics to data frame #####

df = cbind(df, round(twentyfive_model$theta, digits=2))
head(df, n=10)

####################################################################
########## Model Building for k = 25 ###################

# Splitting the Data into train and test

set.seed(123) # Set the seed to make it reproducible

train.index <- createDataPartition(df$status, p = .8, list = FALSE)
train <- df[ train.index,]
test  <- df[-train.index,]

actual <- test$status
formula <- status ~ .
positive <- "yes"

# Random Forest with great results
set.seed(123)
model_forest <- randomForest(status~ ., data=train, 
                             importance=TRUE,proximity=TRUE,
                             cutoff = c(0.5, 0.5),type="classification", na.action=na.exclude)
print(model_forest)   
plot(model_forest)
importance(model_forest)

varImpPlot(model_forest)#Shows the decrease in accuracy with the removal of a variables so non Annual removal the accuray can go down by 30% for example

predicted <- predict(model_forest, test, type="class") 

print(sprintf("Accuracy of Random Forest for k = 25:    %.3f", Accuracy(y_true=actual, y_pred=predicted)))
print(sprintf("AUC of Random Forest for k = 25:     %.3f", AUC(y_pred=predicted, y_true=actual)))
print(sprintf("Precision of Random Forest for k = 25:   %.3f", Precision(y_true=actual, y_pred=predicted)))
print(sprintf("Recall of Random Forest for k = 25:      %.3f", Recall(y_true=actual, y_pred=predicted)))
print(sprintf("F1 Score of Random Forest for k = 25:    %.3f", F1_Score(predicted, actual)))
print(sprintf("Sensitivity of Random Forest for k = 25: %.3f", Sensitivity(y_true=actual, y_pred=predicted)))
print(sprintf("Specificity of Random Forest for k = 25: %.3f", Specificity(y_true=predicted, y_pred=actual)))

# Naive Bayes...Accuracy of 0.73 not very good
set.seed(123)

nb_fit = naiveBayes(status ~ ., data=train)
nb_fit

NB_pred = predict(nb_fit, test, type="class") 
print(sprintf("Accuracy of NB for k = 25:    %.3f", Accuracy(y_true=actual, y_pred=NB_pred)))
print(sprintf("Precision of NB for k = 25:   %.3f", Precision(y_true=actual, y_pred=NB_pred)))
print(sprintf("F1 Score of NB for k = 25:    %.3f", F1_Score(NB_pred, actual)))
print(sprintf("Sensitivityof NB for k = 25: %.3f", Sensitivity(y_true=actual, y_pred=NB_pred)))
print(sprintf("Specificity of NB for k = 25: %.3f", Specificity(y_true=NB_pred, y_pred=actual)))
# Decision Tree
set.seed(123)

tree <- rpart(status ~ ., method="class", data=train)
tree_pred = predict(tree, test, type="class") 
print(sprintf("Accuracy of DT for k = 25:    %.3f", Accuracy(y_true=actual, y_pred=tree_pred)))
print(sprintf("Precision of DT for k = 25:   %.3f", Precision(y_true=actual, y_pred=tree_pred)))
print(sprintf("F1 Score of DT for k = 25:    %.3f", F1_Score(tree_pred, actual)))
print(sprintf("Sensitivity of DT for k = 25: %.3f", Sensitivity(y_true=actual, y_pred=tree_pred)))
print(sprintf("Specificity of DT for k = 25: %.3f", Specificity(y_true=tree_pred, y_pred=actual)))

#SVM...this is the second best
set.seed(123)
svm_fit <- svm(status ~ ., data = train)
summary(svm_fit)

svm_pred <- predict(svm_fit, test) 

print(sprintf("Accuracy of SVM for k = 25:    %.3f", Accuracy(y_true=actual, y_pred=svm_pred)))
print(sprintf("Precision of SVM for k = 25:   %.3f", Precision(y_true=actual, y_pred=svm_pred)))
print(sprintf("F1 Score of SVM for k = 25:    %.3f", F1_Score(svm_pred, actual)))
print(sprintf("Sensitivity of SVM for k = 25: %.3f", Sensitivity(y_true=actual, y_pred=svm_pred)))
print(sprintf("Specificity of SVM for k = 25: %.3f", Specificity(y_true=svm_pred, y_pred=actual)))


#kknn....89%
set.seed(123)
kknn_fit = train.kknn(status ~ ., train, kmax=7)
summary(kknn_fit)

kknn_pred = predict(kknn_fit, test)

print(sprintf("Accuracy of KNN for k = 25:    %.3f", Accuracy(y_true=actual, y_pred=kknn_pred)))
print(sprintf("Precision of KNN for k = 25:   %.3f", Precision(y_true=actual, y_pred=kknn_pred)))
print(sprintf("F1 Score of KNN for k = 25:    %.3f", F1_Score(kknn_pred, actual)))
print(sprintf("Sensitivity of KNN for k = 25: %.3f", Sensitivity(y_true=actual, y_pred=kknn_pred)))
print(sprintf("Specificity of KNN for k = 25: %.3f", Specificity(y_true=kknn_pred, y_pred=actual)))

# #NN...91%
# install.packages("nnet")
# library(nnet)
# set.seed(123)
# 
# train_nn <- (train)
# test_nn <- test
# str(train_nn)
# 
# train_nn$status <- as.numeric(train_nn$status)
# train_nn$sector <- as.numeric(train_nn$sector)
# train_nn$country <- as.numeric(train_nn$country)
# train_nn$gender <- as.numeric(train_nn$gender)
# train_nn$countrygender <- as.numeric(as.factor(as.character(train_nn$countrygender)))
# train_nn$CountrySector <- as.numeric(as.factor(as.character(train_nn$CountrySector)))
# train_nn$nonpayment <- as.numeric (train_nn$nonpayment)
# 
# test_nn$status <- as.numeric(test_nn$status)
# test_nn$sector <- as.numeric(test_nn$sector)
# test_nn$country <- as.numeric(test_nn$country)
# test_nn$gender <- as.numeric(test_nn$gender)
# test_nn$countrygender <- as.numeric(as.factor(as.character(test_nn$countrygender)))
# test_nn$CountrySector <- as.numeric(as.factor(as.character(test_nn$CountrySector)))
# test_nn$nonpayment <- as.numeric (test_nn$nonpayment)
# str(train)
# ##start Neural Network analysis
# 
# my.grid <- expand.grid( .size = c(1,2,4),.decay = c(0.25,1,2)) # Tuning grid for Neural Net
# 
# model_NN <- train(status~ ., data = train_nn, method = "nnet", tuneGrid = my.grid, trace = TRUE, na.action =na.omit)
# 
# plot(model_NN) #Visualize the relationship between the number of layers, decay and accuracy
# NN_prediction<-predict(model_NN, test_nn) #Predict classification 
# print(sprintf("Accuracy of NN for k = 25:    %.3f", Accuracy(y_true=test_nn$status, y_pred=NN_prediction)))
# print(sprintf("Precision of NN for k = 25:   %.3f", Precision(y_true=test_nn$status, y_pred=NN_prediction)))
# print(sprintf("F1 Score of NN for k = 25:    %.3f", F1_Score(NN_prediction, test_nn$status)))
# print(sprintf("Sensitivity of NN for k = 25: %.3f", Sensitivity(y_true=test_nn$status, y_pred=NN_prediction)))
# print(sprintf("Specificity of NN for k = 25: %.3f", Specificity(y_true=NN_prediction, y_pred=test_nn$status)))

### removing topics from this part
topics <- cbind(topics,df[,14:38])
### removing topics from this part
df[,14:38] <- NULL

##### Topic Modelling with Sensitive Study ###############

#### With k = 30 ##########

##### fitting an LDA Model #####

thirty_model <- FitLdaModel(dtm = dtm_3_3, 
                            k = 30, 
                            iterations = 500, # recommend a larger value, 500 or more
                            burnin = 180,
                            optimize_alpha = TRUE,
                            calc_likelihood = TRUE,
                            calc_coherence = TRUE,
                            calc_r2 = TRUE,
                            alpha = 0.1, # this is the default value
                            beta = 0.05, # this is the default value
                            cpus = 21) # Note, this is for a big machine  

plot(thirty_model$log_likelihood, type = "l")


thirty_model$top_terms <- GetTopTerms(phi = thirty_model$phi, M = 8)
head(thirty_model$top_terms) 

# probabilistic coherence, a measure of topic quality
# this measure can be used with any topic model, not just probabilistic ones
thirty_model$coherence <- CalcProbCoherence(phi = thirty_model$phi, dtm = dtm_3_3, M = 5)

# Get the prevalence of each topic
# You can make this discrete by applying a threshold, say 0.05, for
# topics in/out of docuemnts. 
thirty_model$prevalence <- colSums(thirty_model$theta) / sum(thirty_model$theta) * 100


# textmineR has a naive topic labeling tool based on probable bigrams
thirty_model$labels <- LabelTopics(assignments = thirty_model$theta > 0.20, 
                                   dtm = dtm_3_3,
                                   M = 2)

head(thirty_model$labels)


# put them together, with coherence into a summary table
thirty_model$summary <- data.frame(topic = rownames(thirty_model$phi),
                                   label = thirty_model$labels,
                                   coherence = round(thirty_model$coherence, 2),
                                   prevalence = round(thirty_model$prevalence,1),
                                   top_terms = apply(thirty_model$top_terms, 2, function(x){
                                     paste(x, collapse = ", ")
                                   }),
                                   stringsAsFactors = FALSE)

thirty_model$summary %>%
  arrange(-prevalence)


write.csv(thirty_model$top_terms, file = "3.3_30_Topics_kiva.csv")

#### Adding topics to data frame #####

df = cbind(df, round(thirty_model$theta, digits=2))
head(df, n=10)

####################################################################
########## Model Building for k = 30 ###################

# Splitting the Data into train and test

set.seed(123) # Set the seed to make it reproducible

train.index <- createDataPartition(df$status, p = .8, list = FALSE)
train <- df[ train.index,]
test  <- df[-train.index,]

actual <- test$status
formula <- status ~ .
positive <- "yes"

# Random Forest with great results
set.seed(123)
model_forest <- randomForest(status~ ., data=train, 
                             importance=TRUE,proximity=TRUE,
                             cutoff = c(0.5, 0.5),type="classification", na.action=na.exclude)
print(model_forest)   
plot(model_forest)
importance(model_forest)

varImpPlot(model_forest)#Shows the decrease in accuracy with the removal of a variables so non Annual removal the accuray can go down by 30% for example

predicted <- predict(model_forest, test, type="class") 

print(sprintf("Accuracy of Random Forest for k = 30:    %.3f", Accuracy(y_true=actual, y_pred=predicted)))
print(sprintf("AUC of Random Forest for k = 30:     %.3f", AUC(y_pred=predicted, y_true=actual)))
print(sprintf("Precision of Random Forest for k = 30:   %.3f", Precision(y_true=actual, y_pred=predicted)))
print(sprintf("Recall of Random Forest for k = 30:      %.3f", Recall(y_true=actual, y_pred=predicted)))
print(sprintf("F1 Score of Random Forest for k = 30:    %.3f", F1_Score(predicted, actual)))
print(sprintf("Sensitivity of Random Forest for k = 30: %.3f", Sensitivity(y_true=actual, y_pred=predicted)))
print(sprintf("Specificity of Random Forest for k = 30: %.3f", Specificity(y_true=predicted, y_pred=actual)))

# Naive Bayes...Accuracy of 0.73 not very good
set.seed(123)

nb_fit = naiveBayes(status ~ ., data=train)
nb_fit

NB_pred = predict(nb_fit, test, type="class") 
print(sprintf("Accuracy of NB for k = 30:    %.3f", Accuracy(y_true=actual, y_pred=NB_pred)))
print(sprintf("Precision of NB for k = 30:   %.3f", Precision(y_true=actual, y_pred=NB_pred)))
print(sprintf("F1 Score of NB for k = 30:    %.3f", F1_Score(NB_pred, actual)))
print(sprintf("Sensitivityof NB for k = 30: %.3f", Sensitivity(y_true=actual, y_pred=NB_pred)))
print(sprintf("Specificity of NB for k = 30: %.3f", Specificity(y_true=NB_pred, y_pred=actual)))

# Decision Tree
set.seed(123)

tree <- rpart(status ~ ., method="class", data=train)
tree_pred = predict(tree, test, type="class") 
print(sprintf("Accuracy of DT for k = 30:    %.3f", Accuracy(y_true=actual, y_pred=tree_pred)))
print(sprintf("Precision of DT for k = 30:   %.3f", Precision(y_true=actual, y_pred=tree_pred)))
print(sprintf("F1 Score of DT for k = 30:    %.3f", F1_Score(tree_pred, actual)))
print(sprintf("Sensitivity of DT for k = 30: %.3f", Sensitivity(y_true=actual, y_pred=tree_pred)))
print(sprintf("Specificity of DT for k = 30: %.3f", Specificity(y_true=tree_pred, y_pred=actual)))

#SVM...this is the second best
set.seed(123)
svm_fit <- svm(status ~ ., data = train)
summary(svm_fit)

svm_pred <- predict(svm_fit, test) 

print(sprintf("Accuracy of SVM for k = 30:    %.3f", Accuracy(y_true=actual, y_pred=svm_pred)))
print(sprintf("Precision of SVM for k = 30:   %.3f", Precision(y_true=actual, y_pred=svm_pred)))
print(sprintf("F1 Score of SVM for k = 30:    %.3f", F1_Score(svm_pred, actual)))
print(sprintf("Sensitivity of SVM for k = 30: %.3f", Sensitivity(y_true=actual, y_pred=svm_pred)))
print(sprintf("Specificity of SVM for k = 30: %.3f", Specificity(y_true=svm_pred, y_pred=actual)))


#kknn....89%
set.seed(123)

kknn_fit = train.kknn(status ~ ., train, kmax=7)
summary(kknn_fit)

kknn_pred = predict(kknn_fit, test)

print(sprintf("Accuracy of KNN for k = 30:    %.3f", Accuracy(y_true=actual, y_pred=kknn_pred)))
print(sprintf("Precision of KNN for k = 30:   %.3f", Precision(y_true=actual, y_pred=kknn_pred)))
print(sprintf("F1 Score of KNN for k = 30:    %.3f", F1_Score(kknn_pred, actual)))
print(sprintf("Sensitivity of KNN for k = 30: %.3f", Sensitivity(y_true=actual, y_pred=kknn_pred)))
print(sprintf("Specificity of KNN for k = 30: %.3f", Specificity(y_true=kknn_pred, y_pred=actual)))

# #NN...91%
# install.packages("nnet")
# library(nnet)
# set.seed(123)
# 
# train_nn <- (train)
# test_nn <- test
# str(train_nn)
# 
# train_nn$status <- as.numeric(train_nn$status)
# train_nn$sector <- as.numeric(train_nn$sector)
# train_nn$country <- as.numeric(train_nn$country)
# train_nn$gender <- as.numeric(train_nn$gender)
# train_nn$countrygender <- as.numeric(as.factor(as.character(train_nn$countrygender)))
# train_nn$CountrySector <- as.numeric(as.factor(as.character(train_nn$CountrySector)))
# train_nn$nonpayment <- as.numeric (train_nn$nonpayment)
# 
# test_nn$status <- as.numeric(test_nn$status)
# test_nn$sector <- as.numeric(test_nn$sector)
# test_nn$country <- as.numeric(test_nn$country)
# test_nn$gender <- as.numeric(test_nn$gender)
# test_nn$countrygender <- as.numeric(as.factor(as.character(test_nn$countrygender)))
# test_nn$CountrySector <- as.numeric(as.factor(as.character(test_nn$CountrySector)))
# test_nn$nonpayment <- as.numeric (test_nn$nonpayment)
# str(train)
# ##start Neural Network analysis
# 
# my.grid <- expand.grid( .size = c(1,2,4),.decay = c(0.25,1,2)) # Tuning grid for Neural Net
# 
# model_NN <- train(status~ ., data = train_nn, method = "nnet", tuneGrid = my.grid, trace = TRUE, na.action =na.omit)
# 
# plot(model_NN) #Visualize the relationship between the number of layers, decay and accuracy
# NN_prediction<-predict(model_NN, test_nn) #Predict classification 
# print(sprintf("Accuracy of NN for k = 30:    %.3f", Accuracy(y_true=test_nn$status, y_pred=NN_prediction)))
# print(sprintf("Precision of NN for k = 30:   %.3f", Precision(y_true=test_nn$status, y_pred=NN_prediction)))
# print(sprintf("F1 Score of NN for k = 30:    %.3f", F1_Score(NN_prediction, test_nn$status)))
# print(sprintf("Sensitivity of NN for k = 30: %.3f", Sensitivity(y_true=test_nn$status, y_pred=NN_prediction)))
# print(sprintf("Specificity of NN for k = 30: %.3f", Specificity(y_true=NN_prediction, y_pred=test_nn$status)))

### removing topics from this part
topics <- cbind(topics, df[,14:43])
### removing topics from this part
df[,14:43] <- NULL

####################################

