# 📊 Customer Churn Prediction

> Predicting telecom customer churn using Python & Machine Learning on **7,043 real customer records**.
> Built by **Vaishnavi Wagh** | Data Analyst Portfolio Project

---

## 🎯 Project Objective

Identify customers likely to churn so the business can take **proactive retention action** — reducing revenue loss before it happens.

---

## 📈 Results

| Model               | Accuracy | ROC-AUC |
|---------------------|----------|---------|
| Logistic Regression | 80%    | 79%   |
| **Random Forest**   | **82%**| **83%** |

> ✏️ Replace XX.X% with your actual numbers after running the code

---

## 🔍 Key Business Insights

- 📌 **Month-to-month customers** churn **3× more** than annual contract customers
- 📌 Customers in their **first 12 months** are at the highest churn risk
- 📌 **High monthly charges** (above $65) strongly correlate with churn
- 📌 Customers **without tech support** churn significantly more
- 📌 **Top churn drivers:** Tenure, Contract Type, Monthly Charges

---

## 📊 Visualizations

### Churn by Contract Type
![Churn by Contract](chart_1_contract_churn.png)

### Churn by Tenure
![Churn by Tenure](chart_2_tenure_churn.png)

### Monthly Charges vs Churn
![Monthly Charges](chart_3_charges_churn.png)

### Correlation Heatmap
![Heatmap](chart_4_heatmap.png)

### Confusion Matrix
![Confusion Matrix](chart_5_confusion_matrix.png)

### Top 10 Churn Drivers — Feature Importance
![Feature Importance](chart_6_feature_importance.png)

### Model Comparison
![Model Comparison](chart_7_model_comparison.png)

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| `Python` | Core programming language |
| `Pandas` | Data loading, cleaning, manipulation |
| `NumPy` | Numerical operations |
| `Scikit-learn` | ML models (Logistic Regression, Random Forest) |
| `Matplotlib` | Chart plotting |
| `Seaborn` | Statistical visualizations |
| `Power BI` | Dashboard using churn_predictions.csv |

---

## 📁 Project Files

| File | Description |
|------|-------------|
| `churn_project.py` | Full pipeline: data loading, EDA, ML models, evaluation |
| `churn_predictions.csv` | Test set with predicted churn + probability + risk level |
| `chart_1_contract_churn.png` | Churn rate by contract type |
| `chart_2_tenure_churn.png` | Churn distribution by tenure |
| `chart_3_charges_churn.png` | Monthly charges vs churn |
| `chart_4_heatmap.png` | Correlation heatmap |
| `chart_5_confusion_matrix.png` | Confusion matrices — both models |
| `chart_6_feature_importance.png` | Top 10 features driving churn |
| `chart_7_model_comparison.png` | Accuracy vs ROC-AUC comparison |

---

## 🚀 How to Run

**1. Install dependencies**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

**2. Download the dataset**
[Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
Save as `WA_Fn-UseC_-Telco-Customer-Churn.csv` in the same folder.

**3. Run**
```bash
python churn_project.py
```
All 7 charts + churn_predictions.csv will be generated automatically.

---

## 👩‍💻 About Me

**Vaishnavi Wagh** — Entry-level Data Analyst | Pune, Maharashtra
🔗 [LinkedIn](https://www.linkedin.com/in/vaishnavi-wagh-data-analyst) | [GitHub](https://github.com/vaishnaviwagh2108)

*Actively looking for Data Analyst / BI Analyst roles in Pune,Mumbai,Banglore.*

---

## 📄 License
MIT License# customer-churn-prediction
Predicting telecom customer churn using Python &amp; Scikit-learn 
