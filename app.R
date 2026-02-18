library(shiny)
library(xgboost)
library(Matrix)
library(dplyr)

# 1. Load model
mod_obj       <- readRDS("diabetes_model.rds")
xgb_model     <- mod_obj$model
feature_names <- mod_obj$features
NUM_CLASS     <- 3

# 2. Helper
make_design <- function(df_raw, feat) {
  df <- df_raw %>% mutate(
    Age     = factor(Age,     levels = 1:12),
    GenHlth = factor(GenHlth, levels = 1:4),
    Income  = factor(Income,  levels = 1:8)
  )
  mat <- sparse.model.matrix(Diabetes_012 ~ . - 1, data = df)
  miss <- setdiff(feat, colnames(mat))
  if (length(miss))
    mat <- cbind(mat,
                 Matrix(0, nrow(mat), length(miss),
                        dimnames = list(NULL, miss)))
  mat[, feat, drop = FALSE]
}

# 3. UI
ui <- fluidPage(
  titlePanel("Diabetes Risk Predictor"),
  sidebarLayout(
    sidebarPanel(
      numericInput("HighBP",  "HighBP (0/1)", 0, 0,1,1),
      numericInput("BMI",     "BMI",          25,10,60,1),
      numericInput("Age",     "Age (1–13)",   3, 1,13,1),
      numericInput("HighChol","HighChol (0/1)",0,0,1,1),
      numericInput("GenHlth","Gen Health (1–5)",3,1,5,1),
      numericInput("Income","Income (1–8)", 5,1,8,1),
      numericInput("HeartDiseaseorAttack","Heart Disease (0/1)",1,0,1,1),
      actionButton("predict","Predict")
    ),
    mainPanel(
      h3("Result"),
      verbatimTextOutput("result")
    )
  )
)

# 4. Server
server <- function(input, output, session) {
  observeEvent(input$predict, {
    raw <- data.frame(
      Diabetes_012 = 0,
      HighBP               = input$HighBP,
      BMI                  = input$BMI,
      Age                  = input$Age,
      HighChol             = input$HighChol,
      GenHlth              = input$GenHlth,
      Income               = input$Income,
      HeartDiseaseorAttack = input$HeartDiseaseorAttack
    )
    dmat  <- make_design(raw, feature_names)
    p_vec <- predict(xgb_model, dmat)
    probs <- matrix(p_vec, ncol = NUM_CLASS, byrow = TRUE)
    pred  <- max.col(probs) - 1
    cls   <- c("No-Diabetes","Prediabetes","Diabetes")

    output$result <- renderPrint({
      list(
        Predicted_Class = pred,
        Probabilities   = setNames(round(probs[1,],3), cls)
      )
    })
  })
}

# 5. Run
shinyApp(ui, server)
