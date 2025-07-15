# 🩺 Diabetes Risk Prediction

**Multiclass Classification with XGBoost & CatBoost**  
Author: Emmanuel Fagbenle  
Dataset: CDC BRFSS 2015

---

## 📌 Overview

Diabetes remains one of the most pervasive chronic diseases in the United States, with ~34 million diagnosed cases and ~88 million pre-diabetics. Early detection improves outcomes and reduces the estimated $327 billion annual economic burden.

This project develops and benchmarks **two gradient-boosted tree models** that classify an individual’s diabetes status using self-reported health indicators:

| Class | Meaning                                |
|-------|----------------------------------------|
| **0** | No diabetes  |
| **1** | Prediabetes                            |
| **2** | Diabetes                               |

---

## 📦 Dataset

- **Source**: Behavioral Risk Factor Surveillance System (BRFSS 2015) by the CDC  
- **File used**: `diabetes_012_health_indicators_BRFSS2015.csv`  
- **Samples / Features**: 253,680 × 21  
- **Target**: `Diabetes_012`  
- **Note**: The classes are imbalanced (most respondents fall in class 0)

> Each row is a respondent with 21 health & lifestyle attributes (blood pressure, BMI, income, etc.)

---

## 🧠 Models & Methodology

### 1. XGBoost

### 2. CatBoost
Both models were trained on **80%** of the data and validated on the remaining **20%**.

---

## 📈 Model Performance

| Model    | Accuracy   |
|----------|------------|
| XGBoost  | **0.8491** |
| CatBoost | **0.8494** |

CatBoost delivered a marginal improvement (+0.03%) while retaining similar interpretability through built-in feature importance.

---

### 🔑 Top 8 Predictive Features (both models)

1. **HighBP** – High blood pressure  
2. **GenHlth** – Self-reported general health  
3. **BMI** – Body-Mass Index  
4. **Age** – Age category  
5. **HighChol** – High cholesterol  
6. **Income** – Income bracket  
7. **HeartDiseaseorAttack** – History of heart disease/attack  
8. **PhysHlth** – Days of poor physical health  

---

## 🚀 Use Cases

| Stakeholder                | Value Proposition                                    |
|---------------------------|------------------------------------------------------|
| 🏥 Clinicians              | Decision support during routine screenings          |
| 🏛 Public Health Agencies  | Early identification of high-risk communities       |
| 🧑‍🔬 Researchers          | Analysis of social & behavioral risk factors         |
| 💻 Data Scientists         | Benchmark for multiclass classification on health data |

---

## 🙌 Acknowledgments
1. CDC BRFSS team for data collection
2. Kaggle community for dataset cleaning

---

## 📬 Contact
Feel free to reach out or open an issue.
[LinkedIn](https://www.linkedin.com/in/fagbenle-emmanuel/)
