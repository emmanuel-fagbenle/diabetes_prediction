FROM rocker/shiny:4.2.2

# Install needed R packages
RUN R -e "install.packages(c('shiny','xgboost','Matrix','dplyr'), repos='https://cloud.r-project.org/')"

# Copy your app & model
COPY app.R /app/app.R
COPY diabetes_model.rds /app/diabetes_model.rds

# Set working dir
WORKDIR /app

# Expose the port HF expects
EXPOSE 8080


# Run the Shiny app directly on 0.0.0.0:8080
CMD ["R", "-e", "shiny::runApp('.', host='0.0.0.0', port=7860)"]
