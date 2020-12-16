#-----------------------------Loading Libraries and Data -----------------------------

library(readr) #For importing data
library(dplyr) #For data manipulation
library(ggplot2) #For plotting
library(tidyr) #tidying data
library(forcats) #working with factors
library(tidytext) #For text analysis techniques
library(textfeatures) #to get features
library(tidylo) #to get log odds
library(tidymodels) #For modelling
library(themis) #For dealing with class imbalance
library(textrecipes) #extracts out features produced in the textfeatures package
library(SnowballC) #performs word stemming
library(vip) #For getting feature importances
library(tm) #For topic modelling
library(topicmodels) #For topic modelling (LDA)

# Loading in data - data taken from https://www.kaggle.com/tboyle10/medicaltranscriptions
path = "mtsamples.csv"
transcripts <- read_csv(path)

# The X1 column is just the indices so we can drop it
transcripts <- transcripts %>% select(!X1)

#------------------------------ Some EDA -------------------------------------------

transcripts %>% View()

# Lots of repeats in medical_specialty - let's see how many transcripts are in each
counts <- transcripts %>%
  group_by(medical_specialty) %>%
  count(sort = TRUE)

# Since there are so many classes, let's just select the top 4.
# These will be used for classification of medical specialty from the transcription
count_vector <- as.vector(counts$medical_specialty)[1:4]
transcripts <- transcripts %>%
  filter(medical_specialty %in% count_vector)


# Since we are predicting specialty from transcript, let's convert medical_specialty 
# to a factor
transcripts <- transcripts %>%
  mutate(medical_specialty = factor(medical_specialty))

# Lots of overlap - let's visualize the top few
transcripts %>%
  select(medical_specialty) %>%
  group_by(medical_specialty) %>%
  count(sort = TRUE) %>%
  ungroup() %>%
  slice(1:10) %>%
  ggplot(aes(
    x = reorder(medical_specialty, -n),
    y = n)) + 
  geom_bar(stat = "identity", show.legend = FALSE) + 
  xlab("Medical Specialty")
# The disproportionate number of transcripts classified as 
# surgery could be an issue when performing classification modelling

# Let's see the most popular keywords, using the keywords column
transcripts %>%
  select("keywords") %>%
  unnest_tokens(word, keywords) %>%
  anti_join(stop_words) %>%
  group_by(word) %>%
  drop_na() %>%
  count(sort = TRUE)
# We can see that words like artery, pulmonary, and cervical are common

# Calculating log odds to see which words with highest log odds
transcripts_lo <- transcripts %>%
  unnest_tokens(word, transcription) %>%
  anti_join(stop_words) %>%
  count(medical_specialty, word) %>%
  bind_log_odds(medical_specialty, word, n) %>%
  arrange(-log_odds_weighted)

# Find the most common log odds in each specialty
transcripts_lo %>%
  group_by(medical_specialty) %>%
  slice_max(log_odds_weighted, n = 15) %>%
  ungroup() %>%
  mutate(word = reorder(word, log_odds_weighted)) %>% 
  ggplot(aes(log_odds_weighted, word, fill = medical_specialty)) +
  geom_col(alpha = 0.8, show.legend = FALSE) +
  facet_wrap(~medical_specialty, scales = "free") + 
  labs(y = NULL)

transcripts <- transcripts %>%
  rename(text = transcription)
tf <- textfeatures(transcripts, sentiment = FALSE, word_dims = 0,
                   normalize = FALSE)
tf %>%
  bind_cols(transcripts) %>%
  group_by(medical_specialty) %>%
  summarize(across(starts_with("n_"), mean)) %>%
  pivot_longer(starts_with("n_"), names_to = "text_feature") %>%
  filter(value > 0.01) %>%
  mutate(text_feature = fct_reorder(text_feature, -value)) %>%
  ggplot(aes(medical_specialty, value, fill = medical_specialty)) +
  geom_col(position = "dodge", alpha = 0.8, show.legend = FALSE) + 
  facet_wrap(~text_feature, scales = "free", ncol = 6) +
  labs(x = NULL, y = "Mean text features for each specialty")


#------------------- Initial Modelling ------------------------

# Keeping only columns of dataset we need for modelling
transcripts_ml <- transcripts %>%
  select(medical_specialty, text)

# dropping na value rows
transcripts_ml <- drop_na(transcripts_ml)

# Splitting into training and testing using rsample
set.seed(123)
ml_split <- initial_split(transcripts_ml, strata = medical_specialty)
ml_train <- training(ml_split)
ml_test <- testing(ml_split)

#Taking a quick glance at the training data

# Making a 10 fold cv of the training data
set.seed(234)
ml_folds <- vfold_cv(ml_train, strata = medical_specialty)

# doing preprocessing - 1. downsampling (class imbalance) 2.got text features,
# 3. remove variables with 0 variance 4. normalize predictors
ml_recipe <- recipe(medical_specialty ~ text, data = ml_train) %>%
  step_downsample(medical_specialty) %>%
  step_textfeature(text) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_predictors())

# taking a quick look at what data looks like after recipe is performed
ml_prep <- prep(ml_recipe)
juice(ml_prep)
skimr::skim(juice(ml_prep))

# setting up a logreg model with l2 regularization (ridge)
glm_spec <- multinom_reg(penalty = 1, mixture = 0) %>%
  set_engine("glmnet") %>%
  set_mode("classification")
glm_spec
# setting up a random forest model specification using ranger
rf_spec <- rand_forest(trees = 1000) %>%
  set_engine("ranger") %>%
  set_mode("classification")
rf_spec

# setting up a RBF SVM model specification using kernlab
svm_spec <- svm_rbf(cost = 0.5) %>%
  set_engine("kernlab") %>%
  set_mode("classification")
svm_spec

# setting up a XGBoost model specification using xgboost (not any tuning for now, though)
xgb_spec <- boost_tree(trees = 1000) %>%
  set_engine("xgboost") %>%
  set_mode("classification")
xgb_spec

# setting up a workflow to add models to
ml_wf <- workflow() %>%
  add_recipe(ml_recipe)

# running rf, svm models in parallel
doParallel::registerDoParallel()

# fitting logistic regression
set.seed(3456)
glm_rs <- ml_wf %>%
  add_model(glm_spec) %>%
  fit_resamples(
    resamples = ml_folds,
    metrics = metric_set(roc_auc, accuracy, sens, spec),
    control = control_resamples(save_pred = TRUE)
  )
# collecting metrics
collect_metrics(glm_rs) # accuracy: 0.472 mAUC: 0.68
conf_mat_resampled(glm_rs)

# ROC Curve for C / P Class
glm_rs %>%
  collect_predictions() %>%
  group_by(id) %>%
  mutate(medical_specialty = ifelse(
    medical_specialty == "Cardiovascular / Pulmonary",
    "Cardiovascular / Pulmonary", "Not C/P" )) %>%
  mutate(medical_specialty = factor(medical_specialty)) %>%
  roc_curve(medical_specialty, `.pred_Cardiovascular / Pulmonary`) %>%
  ggplot(aes(1 - specificity, sensitivity, color = id)) + 
  geom_abline(lty = 2, color = "grey70", size = 1.5) +
  geom_path(show.legend = FALSE, alpha = 0.6, size = 1.2) +
  coord_equal() # not the best

# fitting random forest
set.seed(1234)
rf_rs <- ml_wf %>%
  add_model(rf_spec) %>%
  fit_resamples(
    resamples = ml_folds,
    metrics = metric_set(roc_auc, accuracy, sens, spec),
    control = control_resamples(save_pred = TRUE)
  )
# collecting metrics
collect_metrics(rf_rs) # accuracy: 0.455 mAUC: 0.709
conf_mat_resampled(rf_rs)

# ROC Curve for C / P Class
rf_rs %>%
  collect_predictions() %>%
  group_by(id) %>%
  mutate(medical_specialty = ifelse(
    medical_specialty == "Cardiovascular / Pulmonary",
    "Cardiovascular / Pulmonary", "Not C/P" )) %>%
  mutate(medical_specialty = factor(medical_specialty)) %>%
  roc_curve(medical_specialty, `.pred_Cardiovascular / Pulmonary`) %>%
  ggplot(aes(1 - specificity, sensitivity, color = id)) + 
  geom_abline(lty = 2, color = "grey70", size = 1.5) +
  geom_path(show.legend = FALSE, alpha = 0.6, size = 1.2) +
  coord_equal() # sometimes dips below guessing

# fitting SVM
set.seed(2345)
svm_rs <- ml_wf %>%
  add_model(svm_spec) %>%
  fit_resamples(
    resamples = ml_folds,
    metrics = metric_set(roc_auc, accuracy, sens, spec),
    control = control_resamples(save_pred = TRUE)
  )
# collecting metrics
collect_metrics(svm_rs) # accuracy: 0.529 mAUC: 0.731
conf_mat_resampled(svm_rs)
# ROC Curve
svm_rs %>%
  collect_predictions() %>%
  group_by(id) %>%
  mutate(medical_specialty = ifelse(
    medical_specialty == "Cardiovascular / Pulmonary",
    "Cardiovascular / Pulmonary", "Not C/P" )) %>%
  mutate(medical_specialty = factor(medical_specialty)) %>%
  roc_curve(medical_specialty, `.pred_Cardiovascular / Pulmonary`) %>%
  ggplot(aes(1 - specificity, sensitivity, color = id)) + 
  geom_abline(lty = 2, color = "grey70", size = 1.5) +
  geom_path(show.legend = FALSE, alpha = 0.6, size = 1.2) +
  coord_equal() # much better than random forest, but also sometimes dips below guessing

set.seed(4567)
xgb_rs <- ml_wf %>%
  add_model(svm_spec) %>%
  fit_resamples(
    resamples = ml_folds,
    metrics = metric_set(roc_auc, accuracy, sens, spec),
    control = control_resamples(save_pred = TRUE)
  )
# collecting metrics
collect_metrics(xgb_rs) # accuracy: 0.531 mAUC: 0.733 (didn't tune it, so this performance makes sense)
conf_mat_resampled(svm_rs)
# ROC Curve
xgb_rs %>%
  collect_predictions() %>%
  group_by(id) %>%
  mutate(medical_specialty = ifelse(
    medical_specialty == "Cardiovascular / Pulmonary",
    "Cardiovascular / Pulmonary", "Not C/P" )) %>%
  mutate(medical_specialty = factor(medical_specialty)) %>%
  roc_curve(medical_specialty, `.pred_Cardiovascular / Pulmonary`) %>%
  ggplot(aes(1 - specificity, sensitivity, color = id)) + 
  geom_abline(lty = 2, color = "grey70", size = 1.5) +
  geom_path(show.legend = FALSE, alpha = 0.6, size = 1.2) +
  coord_equal() # a little better than random forest - dips below guessing a few times though


#-------------------- Hyperparameter tuning ------------------
#Setting up the different models:
# 1. logreg ; 2. rf ; 3. svm ; 4. xgb

logreg_tune <- multinom_reg(
    penalty = tune(),
    mixture = tune()
  ) %>%
  set_mode("classification") %>%
  set_engine("glmnet")

rf_tune <- rand_forest(
    mtry = tune(),
    trees = 1000,
    min_n = tune()
  ) %>%
  set_mode("classification") %>%
  set_engine("ranger")

svm_tune <- svm_rbf(
    cost = tune(),
    rbf_sigma = tune()
  ) %>%
  set_mode("classification") %>%
  set_engine("kernlab")

xgb_tune <- boost_tree(
  trees = 1000,
  tree_depth = tune(),
  min_n = tune(),
  loss_reduction = tune(),
  sample_size = tune(),
  mtry = tune(),
  learn_rate = tune(),
) %>%
  set_mode("classification") %>%
  set_engine("xgboost")

logreg_tune_wf <-
  ml_wf %>%
  add_model(logreg_tune)

rf_tune_wf <-
  ml_wf %>%
  add_model(rf_tune)

svm_tune_wf <-
  ml_wf %>%
  add_model(svm_tune)

# Tuning the models
doParallel::registerDoParallel()
set.seed(234)
logreg_grid <- tune_grid(
  logreg_tune_wf,
  resamples = ml_folds,
  grid = 20
)
rf_grid <- tune_grid(
  rf_tune_wf,
  resamples = ml_folds,
  grid = 20
)
svm_grid <- tune_grid(
  svm_tune_wf,
  resamples = ml_folds,
  grid = 20
)

xgb_grid <- grid_latin_hypercube(
  tree_depth(),
  min_n(),
  loss_reduction(),
  sample_size = sample_prop(),
  finalize(mtry(), ml_train),
  learn_rate(),
  size = 30
)
xgb_wf <- workflow() %>%
  add_recipe(ml_recipe) %>%
  add_model(xgb_tune)

set.seed(234)
xgb_res <- tune_grid(
  xgb_wf,
  resamples = ml_folds,
  grid = xgb_grid,
  control = control_grid(save_pred = TRUE)
)

# Getting the metrics for each model
collect_metrics(logreg_grid)
collect_metrics(rf_grid)
collect_metrics(svm_grid)
collect_metrics(xgb_res)
# Getting models with best accuracy
logreg_grid %>% select_best("accuracy")
rf_grid %>% select_best("accuracy")
svm_grid %>% select_best("accuracy")
xgb_res %>% select_best("accuracy")

# Getting models with best mAUC  
logreg_grid %>% select_best("roc_auc")
rf_grid %>% select_best("roc_auc")
svm_grid %>% select_best("roc_auc")
xgb_res %>% select_best("roc_auc")

# Visualizing results
logreg_grid %>%
  collect_metrics() %>% 
  filter(.metric == "roc_auc") %>% 
  select(mean, penalty, mixture) %>%
  pivot_longer(penalty:mixture, 
               values_to = "value",
               names_to = "parameter") %>%
  ggplot(aes(value, mean, color = parameter)) +
  geom_point(show.legend = FALSE) +
  facet_wrap(~ parameter, scales = "free_x")

rf_grid %>%
  collect_metrics() %>% 
  filter(.metric == "roc_auc") %>% 
  select(mean, mtry, min_n) %>%
  pivot_longer(mtry:min_n, 
               values_to = "value",
               names_to = "parameter") %>%
  ggplot(aes(value, mean, color = parameter)) +
  geom_point(show.legend = FALSE) +
  facet_wrap(~ parameter, scales = "free_x")

svm_grid %>%
  collect_metrics() %>% 
  filter(.metric == "roc_auc") %>% 
  select(mean, cost, rbf_sigma) %>%
  pivot_longer(cost:rbf_sigma, 
               values_to = "value",
               names_to = "parameter") %>%
  ggplot(aes(value, mean, color = parameter)) +
  geom_point(show.legend = FALSE) +
  facet_wrap(~ parameter, scales = "free_x")

xgb_res %>%
  collect_metrics() %>% 
  filter(.metric == "roc_auc") %>% 
  select(mean, mtry:sample_size) %>%
  pivot_longer(mtry:sample_size, 
               values_to = "value",
               names_to = "parameter") %>%
  ggplot(aes(value, mean, color = parameter)) +
  geom_point(show.legend = FALSE) +
  facet_wrap(~ parameter, scales = "free_x")

best_logreg <- select_best(logreg_grid, "roc_auc")
best_rf <- select_best(rf_grid, "roc_auc")
best_svm <- select_best(svm_grid, "roc_auc")
best_xgb <- select_best(xgb_res, "roc_auc")

final_logreg <- finalize_model(
  logreg_tune,
  best_logreg
)

final_rf <- finalize_model(
  rf_tune,
  best_rf
)

final_svm <- finalize_model(
  svm_tune,
  best_svm
)

final_svm <- finalize_model(
  svm_tune,
  best_svm
)

final_xgb <- finalize_model(
  xgb_tune,
  best_xgb
)

# Fitting best model on training data and testing on test set
log_final <- last_fit(final_logreg, ml_recipe, ml_split)
rf_final <- last_fit(final_rf, ml_recipe, ml_split)
svm_final <- last_fit(final_svm, ml_recipe, ml_split)
xgb_final <- last_fit(final_xgb, ml_recipe, ml_split)

log_final %>% collect_metrics() # Accuracy: 0.511 mAUC: 0.741
rf_final %>% collect_metrics()  # Accuracy: 0.506 mAUC: 0.747
svm_final %>% collect_metrics() # Accuracy: 0.561 mAUC: 0.728
xgb_final %>% collect_metrics() # Accuracy: 0.482 mAUC: 0.728

#
log_final %>%
  collect_predictions() %>%
  mutate(correct = case_when(medical_specialty = .pred_class ~ "Correct",
                             TRUE ~ "Incorrect")) %>%
  bind_cols(ml_test)
  

# Feature Importances
final_logreg %>%
  set_engine("glmnet", importance = "permutation") %>%
  fit(medical_specialty ~ ., 
      data = juice(ml_prep)) %>%
  vip(geom = "point")

final_rf %>%
  set_engine("ranger", importance = "permutation") %>%
  fit(medical_specialty ~ ., 
      data = juice(ml_prep)) %>%
  vip(geom = "point")

final_xgb %>%
  set_engine("xgboost", importance = "permutation") %>%
  fit(medical_specialty ~ ., 
      data = juice(ml_prep)) %>%
  vip(geom = "point")



#-------------------- Topic Modelling ----------------------
#getting original data
transcripts_original <- read_csv(path) %>% select(-X1)

# getting tokens from transcription, removing stopwords,
# and performing stemming
transcript_tokens <- transcripts_original %>%
  select(medical_specialty, transcription) %>% 
  unnest_tokens("word", transcription) %>%
  anti_join(stop_words) %>%
  mutate(word = wordStem(word))
transcript_tokens

# putting tokens in a document term matrix with each specialty being its own document
transcript_matrix <- transcript_tokens %>%
  count(medical_specialty, word) %>%
  cast_dtm(document = medical_specialty, term = word,
           value = n, weighting = tm::weightTf)

# Performing LDA
transcript_lda <- LDA(transcript_matrix, k = 4, method = "Gibbs",
                      control = list(seed = 1234))
transcript_lda

transcript_betas <- 
  tidy(transcript_lda, matrix = "beta")

# top words by topic
transcript_betas %>% 
  group_by(topic) %>% 
  top_n(10, beta) %>% 
  arrange(topic, -beta) %>% View()

transcript_gammas <- 
  tidy(transcript_lda, matrix = "gamma")

# how much a specialty fits into a topic
transcript_gammas %>% View()

#top specialty in topic 1
transcript_gammas %>% 
  filter(topic == 1) %>%
  arrange(-gamma)

# Setup train and test data
sample_size <- floor(0.90 * nrow(transcript_matrix))
set.seed(1111)
train_ind <- sample(nrow(transcript_matrix), size = sample_size)
train <- transcript_matrix[train_ind, ]
test <- transcript_matrix[-train_ind, ]

# Peform topic modeling 
lda_model <- LDA(train, k = 5, method = "Gibbs",
                 control = list(seed = 1111))
# see how well model fits training data
perplexity(lda_model, newdata = train) 
# how well model fits test data
perplexity(lda_model, newdata = test) 

# warning: this some time as well
values = c()
doParallel::registerDoParallel()
for (i in c(2:35)){
  lda_model <- LDA(train, k = i, method = "Gibbs",
                   control = list(iter = 25, seed = 1111))
  values <- c(values, perplexity(lda_model, newdata = test))
}

perplex_tibble <- tibble(topics = c(2:35), perplexity = values)

perplex_tibble %>%
  ggplot(aes(x = topics, y = perplexity)) + 
  geom_point(alpha = 0.7, size = 2) + 
  ggtitle("Perplexity for Topics") + 
  xlab("Number of Topics") + 
  ylab("Perplexity") 
#more topics reduce perplexity
#diminishing returns beyond about 15 topics, so 15 topics is ideal
#however, deciding what each topic means is a lot harder

