import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load and Clean Dataset
df = pd.read_csv("/Users/thuannguyen/Library/Mobile Documents/com~apple~CloudDocs/USYD/Project Student/Data + Code - Stage 2/student_mental_health (dataset 1).csv")

# Clean column names
df.columns = df.columns.str.strip().str.lower()

# Rename for consistency
df = df.rename(columns={
    'choose your gender': 'gender',
    'age': 'age',
    'what is your course?': 'course',
    'your current year of study': 'year',
    'what is your cgpa?': 'cgpa',
    'marital status': 'marital',
    'do you have depression?': 'depression',
    'do you have anxiety?': 'anxiety',
    'do you have panic attack?': 'panic',
    'did you seek any specialist for a treatment?': 'treatment'
})

# Clean gender column
df['gender'] = df['gender'].astype(str).str.lower().str.strip()
df['gender'] = df['gender'].replace({
    'male ': 'male', 'female ': 'female',
    'm': 'male', 'f': 'female'
})
#Drop Timestamp
drop_cols = []
if 'timestamp' in df.columns:
    drop_cols.append('timestamp')

df = df.drop(columns=drop_cols, errors='ignore')
    
# Convert age to numeric, fill missing with median
df['age'] = pd.to_numeric(df['age'], errors='coerce')
median_age = df['age'].median()
df['age'] = df['age'].fillna(median_age)

# Clean categorical fields
df['cgpa'] = df['cgpa'].astype(str).str.strip()
df['year'] = df['year'].astype(str).str.title().str.strip()
df['marital'] = df['marital'].astype(str).str.title().str.strip()

# Convert Yes/No to 1/0 for binary fields
for col in ['depression', 'anxiety', 'panic', 'treatment']:
    df[col] = df[col].astype(str).str.lower().str.strip().map({'yes': 1, 'no': 0})

# Save cleaned dataset
df.to_csv("student_mental_health_clean.csv", index=False)
print("Dataset cleaned and saved as 'student_mental_health_clean.csv'")
print("Columns:", list(df.columns))
print("Shape:", df.shape)

# Encode important features
cat_cols = ['gender', 'year', 'cgpa', 'marital', 'course']
le = LabelEncoder()
for col in cat_cols:
    if col in df.columns:
        df[col] = le.fit_transform(df[col].astype(str))

X = df.drop(columns=['depression'])
y = df['depression']

# Split Train / Validation / Test 
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.25, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.4, random_state=42)
print(f"\nData split → Train: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")

# Scale numeric features 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Models training
# Logistic Regression
lr = LogisticRegression(max_iter=200, random_state=42)
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)

# Decision Tree
dt = DecisionTreeClassifier(max_depth=4, random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

# K-Nearest Neighbours
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
y_pred_knn = knn.predict(X_test_scaled)

# Evaluating models
def evaluate_model(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(f"\n===== {name} =====")
    print(f"Accuracy : {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall   : {rec:.3f}")
    print(f"F1-score : {f1:.3f}")
    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))
    return [name, acc, prec, rec, f1]

results = []
results.append(evaluate_model("Logistic Regression", y_test, y_pred_lr))
results.append(evaluate_model("Decision Tree", y_test, y_pred_dt))
results.append(evaluate_model("KNN", y_test, y_pred_knn))

# Comparing models
results_df = pd.DataFrame(results, columns=["Model","Accuracy","Precision","Recall","F1"])
print("\n=== Model Comparison ===")
print(results_df.sort_values(by="F1", ascending=False).reset_index(drop=True))

# Model Accuracy
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(6,4))
sns.barplot(data=results_df, x="Model", y="Accuracy", palette=["skyblue","lightgreen","salmon"])
plt.title("Model Comparison (Accuracy)")
plt.ylabel("Accuracy")
plt.ylim(0,1)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()


#Confusion Matrix

from sklearn.metrics import ConfusionMatrixDisplay

plt.figure(figsize=(4,4))
ConfusionMatrixDisplay.from_estimator(lr, X_test_scaled, y_test, cmap="Blues")
plt.title("Confusion Matrix – Logistic Regression (Best Model)")
plt.show()


# Predicted Probability Distribution
# 1. Get the linear scores (z) and predicted probabilities (p)
# The decision_function gives raw model outputs before applying the sigmoid function.
z = lr.decision_function(X_test_scaled)
p = 1 / (1 + np.exp(-z))  # Apply sigmoid transformation to obtain probabilities

# 2. Sort values for a smooth curve
order = np.argsort(z)
z_sorted = z[order]
p_sorted = p[order]
y_sorted = y_test.values[order]

# 3. Plot the sigmoid curve and decision boundary
plt.figure(figsize=(6,5))
plt.plot(z_sorted, p_sorted, color='black', linewidth=2, label="Sigmoid curve")
plt.axhline(0.5, color='royalblue', linestyle='--', linewidth=2, label='Decision boundary (0.5)')

# 4. Scatter the actual data points by true class
# Grey = non-depressed (0), Red = depressed (1)
plt.scatter(z_sorted[y_sorted==0], p_sorted[y_sorted==0], color='gray', s=40, label='Actual: No')
plt.scatter(z_sorted[y_sorted==1], p_sorted[y_sorted==1], color='red', s=40, label='Actual: Yes')

# 5. Customize the figure style and labels
plt.xlabel("Linear score  (z = w·x + b)")
plt.ylabel("Predicted Probability of Depression (σ(z))")
plt.title("Logistic Regression – Sigmoid and Decision Boundary")
plt.ylim(-0.05, 1.05)
plt.grid(alpha=0.4, linestyle='--')
plt.legend(loc='lower right')
plt.show()

# Feature influence from Logistic Regression
coef_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": lr.coef_[0]
}).sort_values(by="Coefficient", ascending=False)

plt.figure(figsize=(7,4))
plt.barh(coef_df["Feature"], coef_df["Coefficient"], color="cornflowerblue")
plt.title("Feature Influence on Depression – Logistic Regression")
plt.xlabel("Coefficient Value")
plt.axvline(0, color='gray', linestyle='--', linewidth=1)
plt.gca().invert_yaxis()
plt.grid(axis="x", linestyle="--", alpha=0.6)
plt.show()

# Feature importance from Decision Tree
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": dt.feature_importances_
}).sort_values(by="Importance", ascending=True)

plt.figure(figsize=(7,4))
plt.barh(importance["Feature"], importance["Importance"], color="lightgreen")
plt.title("Feature Importance – Decision Tree")
plt.xlabel("Importance Score")
plt.grid(axis="x", linestyle="--", alpha=0.6)
plt.show()