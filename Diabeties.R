# Load necessary libraries
install.packages(c("readr", "caret", "Matrix", "xgboost"))
library(readr)
library(caret)
library(Matrix)
library(xgboost)


# Load dataset
df <- read_csv("C:/Users/HP/OneDrive - axamansard.com/Desktop/MS.BA/Tools for business Analytics/New folder/archive (3)/diabetes_012_health_indicators_BRFSS2015.csv")

# Convert target variable to factor (for multiclass)
df$Diabetes_012 <- as.factor(df$Diabetes_012)

# Train-test split
set.seed(123)
trainIndex <- createDataPartition(df$Diabetes_012, p = 0.8, list = FALSE)
train_data <- df[trainIndex, ]
test_data <- df[-trainIndex, ]

# Convert labels to numeric for XGBoost (0, 1, 2)
train_label <- as.numeric(train_data$Diabetes_012) - 1
test_label <- as.numeric(test_data$Diabetes_012) - 1

# Remove target column from features
train_matrix <- sparse.model.matrix(Diabetes_012 ~ . - 1, data = train_data)
test_matrix <- sparse.model.matrix(Diabetes_012 ~ . - 1, data = test_data)

# Train XGBoost model
xgb_model <- xgboost(
  data = train_matrix,
  label = train_label,
  objective = "multi:softprob",
  num_class = 3,
  nrounds = 100,
  max_depth = 6,
  eta = 0.1,
  eval_metric = "mlogloss",
  verbose = 0
)


# Save the Model
saveRDS(xgb_model, "C:/diabetes_prediction/diabetes_model.rds")


# # Predict on test set
# pred_probs <- predict(xgb_model, test_matrix)
# pred_labels <- max.col(matrix(pred_probs, ncol = 3, byrow = TRUE)) - 1
# 
# # Evaluate with confusion matrix
# confusionMatrix(factor(pred_labels), factor(test_label))






# 
# # Get feature importance
# importance <- xgb.importance(feature_names = colnames(train_matrix), model = xgb_model)

# # Select top 7 features
# top7_features <- importance$Feature[1:7]
# print(top7_features)
# 
# 
# 
# 
# # Create new dataset with only top 8 features and target
# df_top7 <- df %>%
#   select(Diabetes_012, all_of(top7_features))
# 
# # Redo train-test split
# set.seed(123)
# trainIndex <- createDataPartition(df_top7$Diabetes_012, p = 0.8, list = FALSE)
# train_data <- df_top7[trainIndex, ]
# test_data <- df_top7[-trainIndex, ]
# 
# train_label <- as.numeric(train_data$Diabetes_012) - 1
# test_label <- as.numeric(test_data$Diabetes_012) - 1
# 
# train_matrix <- sparse.model.matrix(Diabetes_012 ~ . - 1, data = train_data)
# test_matrix <- sparse.model.matrix(Diabetes_012 ~ . - 1, data = test_data)
# 
# # Re-train model
# xgb_model_top7 <- xgboost(
#   data = train_matrix,
#   label = train_label,
#   objective = "multi:softprob",
#   num_class = 3,
#   nrounds = 100,
#   max_depth = 6,
#   eta = 0.1,
#   eval_metric = "mlogloss",
#   verbose = 0
# )
# 
# # Evaluate
# pred_probs <- predict(xgb_model_top7, test_matrix)
# pred_labels <- max.col(matrix(pred_probs, ncol = 3, byrow = TRUE)) - 1
# 
# confusionMatrix(factor(pred_labels), factor(test_label))



# # Catboost 
# install.packages("devtools")  # If not already installed
# devtools::install_url("https://github.com/catboost/catboost/releases/download/v1.2.2/catboost-R-Windows-1.2.2.tgz", repos = NULL, type = "source")
# library(catboost)
# 
# 
# # Drop target column from features
# train_pool <- catboost.load_pool(data = train_data[, -which(names(train_data) == "Diabetes_012")],
#                                  label = train_label)
# test_pool <- catboost.load_pool(data = test_data[, -which(names(test_data) == "Diabetes_012")],
#                                 label = test_label)
# 
# # Train CatBoost model
# cat_model <- catboost.train(
#   learn_pool = train_pool,
#   params = list(
#     loss_function = 'MultiClass',
#     iterations = 100,
#     depth = 6,
#     learning_rate = 0.1,
#     eval_metric = 'MultiClass',
#     random_seed = 123,
#     verbose = 0
#   )
# )
# 
# # Predict on test set
# pred_probs <- catboost.predict(cat_model, test_pool, prediction_type = "Probability")
# 
# # Get predicted class (0-based)
# pred_labels <- max.col(pred_probs) - 1
# 
# # Evaluate with confusion matrix
# confusionMatrix(factor(pred_labels), factor(test_label))
# 
# 

