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


train_matrix_full <- sparse.model.matrix(Diabetes_012 ~ . - 1, data = df)
train_label_full <- as.numeric(df$Diabetes_012) - 1

xgb_model_full <- xgboost(
  data        = train_matrix_full,
  label       = train_label_full,
  objective   = "multi:softprob",
  num_class   = 3,
  nrounds     = 100,
  max_depth   = 6,
  eta         = 0.1,
  eval_metric = "mlogloss",
  verbose     = 0
)


# Step 2: Get feature importance and select top 7
importance <- xgb.importance(feature_names = colnames(train_matrix_full), model = xgb_model_full)
top7_features <- head(importance$Feature, 7)
print(top7_features)

# Step 3: Create new dataset with only top 7 features + target
df_top7 <- df %>%
  select(Diabetes_012, all_of(top7_features))

# Step 4: Redo train-test split
set.seed(123)
train_index <- createDataPartition(df_top7$Diabetes_012, p = 0.8, list = FALSE)
train_data <- df_top7[train_index, ]
test_data  <- df_top7[-train_index, ]

train_label <- as.numeric(train_data$Diabetes_012) - 1
test_label  <- as.numeric(test_data$Diabetes_012) - 1

train_matrix <- sparse.model.matrix(Diabetes_012 ~ . - 1, data = train_data)
test_matrix  <- sparse.model.matrix(Diabetes_012 ~ . - 1, data = test_data)




# Step 5: Train new model on top 7 features
xgb_model_top7 <- xgboost(
  data        = train_matrix,
  label       = train_label,
  objective   = "multi:softprob",
  num_class   = 3,
  nrounds     = 100,
  max_depth   = 6,
  eta         = 0.1,
  eval_metric = "mlogloss",
  verbose     = 0
)

# Step 6: Evaluate model performance
pred_prob <- predict(xgb_model_top7, test_matrix)
pred_matrix <- matrix(pred_prob, ncol = 3, byrow = TRUE)
pred_class <- max.col(pred_matrix) - 1

length(pred_class)  # should match
length(test_label)


confusion <- table(Predicted = pred_class, Actual = test_label)
print(confusion)

accuracy <- mean(pred_class == test_label)
cat(sprintf("Accuracy: %.2f%%\n", 100 * accuracy))

# Step 7: Save model and feature names for Shiny app use
xgb_model_top7$feature_names <- top7_features
saveRDS(list(model = xgb_model_top7,
             features = top7_features),
        "C:/diabetes_prediction/diabetes_model.rds")
