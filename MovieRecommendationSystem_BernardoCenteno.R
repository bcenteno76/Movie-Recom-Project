## Loading necessary libraries
library(tidyverse)
library(caret)
library(ggplot2)

## Loading the edx and final_holdout_test datasets
load("C:/Users/berna/Downloads/edx.RData") # dataset for training and testing the models
load('C:/Users/berna/Downloads/final_holdout_test.RData') # dataset for validating the final model

## Data exploration, cleaning and visualization

# Exploring the structure of the edx dataset
str(edx)

# Checking the presence of missing values (NAs)
any(is.na(edx))
    
# Plotting the Distribution of the number of ratings per movie

edx%>% group_by(movieId)%>% summarise(n = n())%>%
  mutate(movieId = factor(movieId, levels = unique(movieId)))%>%
  filter(n < 1000)%>%
  ggplot(aes(n))+ geom_histogram(binwidth = 1)+
  ggtitle('Movies')+ xlab('Number of Ratings')+
  ylab('Number of Movies')

# Plotting the Distribution of the number of ratings per user

edx%>% group_by(userId)%>% summarise(n = n())%>%
  mutate(userId = factor(userId, levels = unique(userId)))%>%
  filter(n < 500)%>%
  ggplot(aes(n))+ geom_histogram(binwidth = 1)+
  ggtitle('Users')+ xlab('Number of Ratings')+
  ylab('Number of Users')

# Plotting the Distribution of the average rating per movie

edx%>% group_by(movieId)%>% summarise(rating_avg = mean(rating))%>%
  ggplot(aes(rating_avg))+ geom_histogram(binwidth = 0.5)+ xlab('Average Rating')+
  ylab('Number of Movies')

# Plotting the Distribution of the average rating per user

edx%>% group_by(userId)%>% summarise(rating_avg = mean(rating))%>%
  ggplot(aes(rating_avg))+ geom_histogram(binwidth = 0.5)+ xlab('Average Rating')+
  ylab('Number of Users')

# Plotting the Distribution of the average rating per genre

edx%>% group_by(genres)%>% summarise(n = n(), genres_avg = mean(rating))%>%
  filter(n > 1000)%>%ggplot(aes(genres_avg))+ geom_histogram(binwidth = 0.5)+ xlab('Average Rating')+
  ylab('Number of Genres')

## Data splitting

set.seed(1) # seed set to 1, to have reproducible results
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.1, list = FALSE)
train_set <- edx[-test_index,] # dataset for training the models
test_set <- edx[test_index,] # dataset for testing the models

# To make sure that the test_set didn´t have a movie or a user not present in the train_set

test_set <- test_set %>%
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

## Model development and testing

# Model 1 (only the average)

mu <- mean(train_set$rating)

rmse1 <- RMSE(test_set$rating,mu)
rmse1 # result of model 1

# Model 2 (average + movie effect)

movie_avgs <- train_set %>% group_by(movieId)%>% summarise(bi = mean(rating - mu))
pred1 <- left_join(test_set,movie_avgs)%>% mutate(pred = mu + bi)%>% pull(pred)
rmse2 <- RMSE(pred1,test_set$rating)
rmse2 # result of model 2

# Model 3 (average + movie and user effects)

user_avgs <- train_set %>%
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(bu = mean(rating - mu - bi))

pred2 <- test_set %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + bi + bu) %>%
  pull(pred)

rmse3 <- RMSE(pred2, test_set$rating)
rmse3 # result of model 3

# Model 4 (average + movie and user effects + regularization)

# Choosing the lambda parameter

lambdas <- seq(0, 10, 0.25)
rmses <- sapply(lambdas, function(l){
  mu <- mean(train_set$rating)
  bi <- train_set %>%
    group_by(movieId) %>%
    summarize(bi = sum(rating - mu)/(n()+l))
  bu <- train_set %>%
    left_join(bi, by="movieId") %>%
    group_by(userId) %>%
    summarize(bu = sum(rating - bi - mu)/(n()+l))
  predicted_ratings <-
    test_set %>%
    left_join(bi, by = "movieId") %>%
    left_join(bu, by = "userId") %>%
    mutate(pred = mu + bi + bu) %>%
    pull(pred)
  return(RMSE(predicted_ratings, test_set$rating))
})
lambda <- lambdas[which.min(rmses)]
lambda # lambda chosen

# Computing the regularized movie and user averages

movie_reg_avgs <- train_set %>%
  group_by(movieId) %>%
  summarize(bi = sum(rating - mu)/(n()+lambda), n_i = n())

user_reg_avgs <- train_set %>%
  left_join(movie_reg_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(bu = sum(rating - mu - bi)/(n()+lambda), n_i = n())

# Testing model 4

pred3 <- test_set %>%
  left_join(movie_reg_avgs, by='movieId') %>%
  left_join(user_reg_avgs, by='userId') %>%
  mutate(pred = mu + bi + bu) %>%
  pull(pred)
rmse4 <- RMSE(pred3, test_set$rating)
rmse4 # result of model 4

# Model 5 (Mov. + User + Reg. + Genres)

genres_avg <- train_set %>%
  left_join(movie_reg_avgs, by = 'movieId')%>%
  left_join(user_reg_avgs, by= 'userId')%>%
  group_by(genres)%>%
  summarize(g = mean(rating- mu - bi - bu))

pred4 <- test_set %>%
  left_join(movie_reg_avgs, by='movieId') %>%
  left_join(user_reg_avgs, by='userId') %>%
  left_join(genres_avg)%>%
  mutate(pred = mu + bi + bu + g) %>%
  pull(pred)
rmse5 <- RMSE(pred4, test_set$rating)
rmse5 # result of model 5

## Model validation

# Testing model 5 on the final_holdout_test

validation_predictions <- final_holdout_test %>%
  left_join(movie_reg_avgs, by='movieId') %>%
  left_join(user_reg_avgs, by='userId') %>%
  left_join(genres_avg, by = 'genres')%>%
  mutate(pred = mu + bi + bu + g) %>%
  pull(pred)
rmse6 <- RMSE(validation_predictions,final_holdout_test$rating)
rmse6 # result of validation

## Results

rmse_results <- tibble(method = c("Just the average",'Movie Effect Model',
                                  'Movie + User effect Model','Mov. + User + Reg. Model',
                                  'Mov. + User + Reg. + Genres Model',
                                  'Final Model Validation'),
                       RMSE = c(rmse1, rmse2, rmse3,rmse4,rmse5,rmse6))
rmse_results # summary table showing the RMSE obtained for each model