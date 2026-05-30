# ============================================================
#  Customer Churn Prediction
#  Author  : Vaishnavi Wagh
#  Tools   : Python · Pandas · Scikit-learn · Matplotlib · Seaborn
#  Dataset : Telco Customer Churn (Kaggle) — 7,043 records
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score)
import warnings
warnings.filterwarnings('ignore')

print("=" * 55)
print("   CUSTOMER CHURN PREDICTION — Vaishnavi Wagh")
print("=" * 55)

# ── STEP 1: LOAD DATA ────────────────────────────────────────
print("\n[1/5] Loading dataset...")
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
print(f"  Shape : {df.shape[0]} rows x {df.shape[1]} columns")

# ── STEP 2: CLEAN DATA ───────────────────────────────────────
print("\n[2/5] Cleaning data...")
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)
df.drop('customerID', axis=1, inplace=True)
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
print(f"  Cleaned shape : {df.shape[0]} rows")
print(f"  Churn rate    : {round(df['Churn'].mean() * 100, 1)}%")

# ── STEP 3: EDA CHARTS ───────────────────────────────────────
print("\n[3/5] Saving EDA charts...")
sns.set_style("whitegrid")

# Chart 1 — Churn by Contract Type
plt.figure(figsize=(8, 4))
sns.countplot(data=df, x='Contract', hue='Churn',
              palette={0: '#9FE1CB', 1: '#F0997B'})
plt.title('Churn Rate by Contract Type', fontsize=14, fontweight='bold')
plt.legend(labels=['Stayed', 'Churned'])
plt.tight_layout()
plt.savefig('chart_1_contract_churn.png', dpi=150)
plt.close()

# Chart 2 — Churn by Tenure
plt.figure(figsize=(8, 4))
sns.histplot(data=df, x='tenure', hue='Churn', bins=30,
             palette={0: '#9FE1CB', 1: '#F0997B'})
plt.title('Churn Distribution by Tenure', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('chart_2_tenure_churn.png', dpi=150)
plt.close()

# Chart 3 — Monthly Charges vs Churn
# NOTE: Churn column is int (0/1) — palette keys must match
df['Churn'] = df['Churn'].astype(int)
plt.figure(figsize=(6, 4))
sns.boxplot(data=df, x='Churn', y='MonthlyCharges',
            palette={0: '#9FE1CB', 1: '#F0997B'})
plt.title('Monthly Charges: Churned vs Stayed', fontsize=14, fontweight='bold')
plt.xticks([0, 1], ['Stayed', 'Churned'])
plt.tight_layout()
plt.savefig('chart_3_charges_churn.png', dpi=150)
plt.close()

# Chart 4 — Correlation Heatmap
plt.figure(figsize=(8, 5))
numeric_cols = df.select_dtypes(include=np.number)
sns.heatmap(numeric_cols.corr(), annot=True, fmt='.2f',
            cmap='Blues', linewidths=0.5)
plt.title('Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('chart_4_heatmap.png', dpi=150)
plt.close()
print("  EDA charts saved!")

# ── STEP 4: PREPROCESSING ────────────────────────────────────
print("\n[4/5] Preprocessing...")
le = LabelEncoder()
cat_cols = df.select_dtypes(include='object').columns
for col in cat_cols:
    if col != 'Churn':   # Churn already encoded — skip it
        df[col] = le.fit_transform(df[col])

X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)
print(f"  Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# ── STEP 5: TRAIN MODELS ─────────────────────────────────────
print("\n[5/5] Training models...")

# Logistic Regression
lr = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
lr.fit(X_train_sc, y_train)
lr_pred = lr.predict(X_test_sc)
lr_acc  = accuracy_score(y_test, lr_pred)
lr_auc  = roc_auc_score(y_test, lr_pred)

# Random Forest
rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_acc  = accuracy_score(y_test, rf_pred)
rf_auc  = roc_auc_score(y_test, rf_pred)

# ── EVALUATION CHARTS ────────────────────────────────────────

# Chart 5 — Confusion Matrix
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, pred, title in zip(axes,
        [lr_pred, rf_pred],
        ['Logistic Regression', 'Random Forest']):
    cm = confusion_matrix(y_test, pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Stayed', 'Churned'],
                yticklabels=['Stayed', 'Churned'])
    ax.set_title(f'Confusion Matrix — {title}', fontweight='bold')
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
plt.tight_layout()
plt.savefig('chart_5_confusion_matrix.png', dpi=150)
plt.close()

# Chart 6 — Feature Importance
feat_imp = pd.DataFrame({
    'Feature'   : X.columns,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False).head(10)

plt.figure(figsize=(8, 5))
sns.barplot(data=feat_imp, x='Importance', y='Feature', palette='Blues_r')
plt.title('Top 10 Features Driving Customer Churn', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('chart_6_feature_importance.png', dpi=150)
plt.close()

# Chart 7 — Model Comparison
models     = ['Logistic Regression', 'Random Forest']
accuracies = [round(lr_acc*100,1), round(rf_acc*100,1)]
aucs       = [round(lr_auc*100,1), round(rf_auc*100,1)]
x = np.arange(len(models))
fig, ax = plt.subplots(figsize=(8, 4))
b1 = ax.bar(x-0.2, accuracies, 0.35, label='Accuracy', color='#9FE1CB')
b2 = ax.bar(x+0.2, aucs,       0.35, label='ROC-AUC',  color='#6BC5F8')
ax.set_ylabel('Score (%)')
ax.set_title('Model Comparison: Accuracy vs ROC-AUC', fontsize=14, fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(models); ax.set_ylim([60,100]); ax.legend()
for b in list(b1)+list(b2):
    ax.annotate(f'{b.get_height()}%',
                xy=(b.get_x()+b.get_width()/2, b.get_height()),
                xytext=(0,3), textcoords="offset points", ha='center', fontsize=10)
plt.tight_layout()
plt.savefig('chart_7_model_comparison.png', dpi=150)
plt.close()

# ── EXPORT PREDICTIONS CSV ───────────────────────────────────
out = X_test.copy()
out['Actual_Churn']      = y_test.values
out['Predicted_Churn']   = rf_pred
out['Churn_Probability'] = rf.predict_proba(X_test)[:,1].round(2)
out['Risk_Level']        = pd.cut(out['Churn_Probability'],
                                  bins=[0,0.3,0.6,1.0],
                                  labels=['Low','Medium','High'])
out.to_csv('churn_predictions.csv', index=False)

# ── FINAL RESULTS ────────────────────────────────────────────
print("\n" + "="*55)
print("   RESULTS SUMMARY")
print("="*55)
print(f"\n  {'Model':<25} {'Accuracy':>10} {'ROC-AUC':>10}")
print(f"  {'-'*45}")
print(f"  {'Logistic Regression':<25} {lr_acc*100:>9.1f}% {lr_auc*100:>9.1f}%")
print(f"  {'Random Forest':<25} {rf_acc*100:>9.1f}% {rf_auc*100:>9.1f}%")
print(f"\n  Top 3 churn drivers:")
for _, row in feat_imp.head(3).iterrows():
    print(f"    -> {row['Feature']} ({row['Importance']:.3f})")
print("\n  All charts saved! Import churn_predictions.csv into Power BI.")
print("="*55)
