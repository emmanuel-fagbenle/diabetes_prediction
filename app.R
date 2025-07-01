# app.R ────────────────────────────────────────────────────────────────
# Shiny front-end for the Top-7 XGBoost diabetes model
# — fixes “incorrect number of dimensions” by:
#   • keeping the design matrix strictly 2-D (drop = FALSE)
#   • reshaping XGBoost’s output vector into a 1 × 3 matrix

library(shiny)
library(xgboost)
library(Matrix)     # sparse matrices + sparse.model.matrix()
library(dplyr)      # for mutate()

# ── 1. Load the trained model & its feature list ----------------------
mod_obj        <- readRDS("diabetes_model.rds")      # adjust path if needed
xgb_model      <- mod_obj$model
feature_names  <- mod_obj$features                      # character vector len = 7
NUM_CLASS      <- 3

# ── 2. Helper: build a design matrix that ALWAYS matches training -----
make_design <- function(df_raw, feat) {
  df <- df_raw %>% mutate(
    Age     = factor(Age,     levels = 1:13),
    GenHlth = factor(GenHlth, levels = 1:5),
    Income  = factor(Income,  levels = 1:8)
  )
  
  mat <- sparse.model.matrix(Diabetes_012 ~ . - 1, data = df)
  
  # add training-time cols that might be missing (all zeros)
  miss <- setdiff(feat, colnames(mat))
  if (length(miss))
    mat <- cbind(
      mat,
      Matrix(0, nrow(mat), length(miss),
             dimnames = list(NULL, miss))
    )
  
  # drop extras and KEEP 2-D structure
  mat <- mat[, feat, drop = FALSE]        # <- critical (no vector collapse)
  mat
}

# ── 3. User interface --------------------------------------------------
ui <- fluidPage(
  titlePanel("Diabetes Risk Predictor (Top-7 XGBoost)"),
  sidebarLayout(
    sidebarPanel(
      numericInput("HighBP",               "HighBP (0/1)",               0,  0, 1, 1),
      numericInput("BMI",                  "BMI",                        25, 10, 60),
      numericInput("Age",                  "Age Category (1–13)",        3,  1, 13, 1),
      numericInput("HighChol",             "HighChol (0/1)",             0,  0, 1, 1),
      numericInput("GenHlth",              "General Health (1–5)",       3,  1, 5, 1),
      numericInput("Income",               "Income Category (1–8)",      5,  1, 8, 1),
      numericInput("HeartDiseaseorAttack", "Heart Disease/Attack (0/1)", 1,  0, 1, 1),
      actionButton("predict", "Predict")
    ),
    mainPanel(
      h3("Prediction Result"),
      verbatimTextOutput("result")
    )
  )
)

# ── 4. Server logic ----------------------------------------------------
server <- function(input, output, session) {
  
  observeEvent(input$predict, {
    
    # 4a. one-row data frame from user inputs
    raw <- data.frame(
      Diabetes_012         = 0,             # dummy y for model matrix
      HighBP               = input$HighBP,
      BMI                  = input$BMI,
      Age                  = input$Age,
      HighChol             = input$HighChol,
      GenHlth              = input$GenHlth,
      Income               = input$Income,
      HeartDiseaseorAttack = input$HeartDiseaseorAttack
    )
    
    # 4b. design matrix aligned to training columns
    dmat <- make_design(raw, feature_names)       # dgCMatrix, 1 × 7
    
    # 4c. predict → flat vector → reshape to 1 × 3 matrix
    p_vec  <- predict(xgb_model, dmat)            # length = 3
    probs  <- matrix(p_vec, ncol = NUM_CLASS, byrow = TRUE)   # 1 × 3
    pred   <- max.col(probs) - 1                  # 0,1,2
    
    # 4d. output (flatten single row safely)
    cls <- c("No-Diabetes", "Prediabetes", "Diabetes")
    output$result <- renderPrint({
      list(
        Predicted_Class = pred,
        Probabilities   = setNames(round(as.numeric(probs[1, ]), 3), cls)
      )
    })
  })
}

# ── 5. Run the app -----------------------------------------------------
shinyApp(ui, server)



