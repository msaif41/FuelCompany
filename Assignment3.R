#install.packages("quanteda")
library('quanteda')

textSM <- read.csv('Downloads/gastext.csv',stringsAsFactors = F)
str(textSM)

textSM$Cust_ID<- factor(textSM$Cust_ID)
textSM[,3:15]<- lapply(textSM[,3:15],factor)
str(textSM)

# Establish the corpus and initial DFM matrix
myCorpusSM <- corpus(textSM$Comment)
summary(myCorpusSM) # unique types of tokens given by the Type counts

myDfmSM <- dfm(myCorpusSM)
dim(myDfmSM) # 287 documents and 793 terms (columns/dimensions)

# Simple frequency analyses
tstat_freq <- textstat_frequency(myDfmSM)
head(tstat_freq, 100)

# Visualize the most frequent terms
library(ggplot2)
myDfmSM %>% 
  textstat_frequency(n = 20) %>% 
  ggplot(aes(x = reorder(feature, frequency), y = frequency)) +
  geom_point() +
  labs(x = NULL, y = "Frequency") +
  theme_minimal()

# Wordcloud
textplot_wordcloud(myDfmSM,max_words=200)

# Remove stop words and perform stemming
library(stopwords)
myDfmSM <- dfm(myCorpusSM, 
             remove_punc = T,
             remove = c(stopwords("english")),
             stem = T)
dim(myDfmSM) # reduced to 582 terms

# Simple frequency analyses
tstat_freq <- textstat_frequency(myDfmSM)
head(tstat_freq, 100)

topfeatures(myDfmSM,100)

# Add more user-defined stop words
# It is a bit subjective, and exercise your judgment with caution
stopwords1 <-c('get','gets','getting','can','don','t','dont','take','just','cant','alway','always','everything','anything','better','�') 

myDfmSM <- dfm(myCorpusSM,
             remove_punc = T,
             remove=c(stopwords('english'),stopwords1),
             stem = T) 
dim(myDfmSM) # further reduced to 571 terms

# Simple frequency analyses
tstat_freq <- textstat_frequency(myDfmSM)
head(tstat_freq, 100)

topfeatures(myDfmSM,100)

# Wordcloud
textplot_wordcloud(myDfmSM,max_words=200)

# Control sparse terms: to further remove some very infrequent words
#myDfmSM<- dfm_trim(myDfmSM,min_termfreq=4, min_docfreq=2)
#dim(myDfmSM)

# Explore terms most similar to "price"
term_sim <- textstat_simil(myDfmSM,
                           selection="price",
                           margin="feature",
                           method="correlation")
as.list(term_sim,n=5)

# Explore terms most similar to "service"
term_sim2 <- textstat_simil(myDfmSM,
                           selection="servic",
                           margin="feature",
                           method="correlation")
as.list(term_sim2,n=5)

# Further remove some very frequent words

myDfmSM %>% 
  textstat_frequency(n = 20) %>% 
  ggplot(aes(x = reorder(feature, frequency), y = frequency)) +
  geom_point() +
  labs(x = NULL, y = "Frequency") +
  theme_minimal()

stopwords2 <- c('shower','point') #'productx','servic','use','price','park','card','clean','food','drink','free','easi','good'
myDfmSM <- dfm_remove(myDfmSM,stopwords2)
topfeatures(myDfmSM,100)
dim(myDfmSM) # further reduced to 569 terms

# Topic Modeling
library(topicmodels)
library(tidytext)

# Explore the option with 4 topics
myDfmSM <- as.matrix(myDfmSM)
myDfmSM <-myDfmSM[which(rowSums(myDfmSM)>0),]
myDfmSM <- as.dfm(myDfmSM)

myLdaSM <- LDA(myDfmSM,k=4,control=list(seed=101))
myLdaSM

# Term-topic probabilities
myLdaSM_td <- tidy(myLdaSM)
myLdaSM_td

# Visualize most common terms in each topic
library(ggplot2)
library(dplyr)

top_terms <- myLdaSM_td %>%
  group_by(topic) %>%
  top_n(15, beta) %>%
  ungroup() %>%
  arrange(topic, -beta)

top_terms %>%
  mutate(term = reorder(term, beta)) %>%
  ggplot(aes(term, beta, fill = factor(topic))) +
  geom_bar(stat = "identity", show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free") +
  coord_flip()

# View topic 15 terms in each topic
Lda_term<-as.matrix(terms(myLdaSM,15))
View(Lda_term)

# Document-topic probabilities
doc_prob <- tidy(myLdaSM, matrix = "gamma")
doc_prob

# View document-topic probabilities in a table
Lda_document<-as.data.frame(myLdaSM@gamma)
View(Lda_document)

# Decision Tree (non-text)

library(caret)
library(car)
library(Hmisc)
library(ggplot2)
library(pROC)
library(rpart)
library(rpart.plot)

# Remove some independent variables
dfSM <- subset(textSM, select = -c(Cust_ID,Comment))
summary(dfSM)
str(dfSM)

# Data partition
trainIndex <- createDataPartition(dfSM$Target,
                                  p=0.7,
                                  list=FALSE,
                                  times=1)
# Create Training Data
dfSM.train <- dfSM[trainIndex,]

# Create Validation data
dfSM.valid <- dfSM[-trainIndex,]

# Build decision tree model
treeSM.model <- train(Target~.,
                      data=dfSM.train,
                      method="rpart",
                      na.action=na.pass)
treeSM.model

# Display decision tree plot
prp(treeSM.model$finalModel,type=2,extra=106)

# Evaluation model performance using validation dataset

# Confusion matrix
prediction <- predict(treeSM.model,newdata=dfSM.valid,na.action=na.pass)
confusionMatrix(prediction,dfSM.valid$Target)

# Decision Tree (with Text)

# Pre-process the training corpus
modelDfmSM <- dfm(myCorpusSM,
                remove_punc = T,
                remove=c(stopwords('english'),stopwords1),
                stem = T) 
# Further remove very infrequent words 
modelDfmSM <- dfm_trim(modelDfmSM,min_termfreq=4, min_docfreq = 2)
dim(modelDfmSM)

# Weight the predictiv DFM by tf-idf
modelDfm_tfidf <- dfm_tfidf(modelDfmSM)
dim(modelDfm_tfidf)

# Perform SVD for dimension reduction
# Choose the number of reduced dimensions as 8
library(quanteda.textmodels)
modelSvdSM <- textmodel_lsa(modelDfm_tfidf, nd=8)
head(modelSvdSM$docs)

# Add to dfSM
modelDataSM <- cbind(dfSM,as.data.frame(modelSvdSM$docs))
head(modelDataSM)

# Data partition
trainIndex <- createDataPartition(modelDataSM$Target,
                                  p=0.7,
                                  list=FALSE,
                                  times=1)
# Create Training Data
dfSM.train <- modelDataSM[trainIndex,]

# Create Validation data
dfSM.valid <- modelDataSM[-trainIndex,]

# Build decision tree model
combinedtreeSM.model <- train(Target~.,
                      data=dfSM.train,
                      method="rpart",
                      na.action=na.pass)
combinedtreeSM.model

# Display decision tree plot
prp(combinedtreeSM.model$finalModel,type=2,extra=106)

# Evaluation model performance using validation dataset

# Confusion matrix
prediction <- predict(combinedtreeSM.model,newdata=dfSM.valid,na.action=na.pass)
confusionMatrix(prediction,dfSM.valid$Target)