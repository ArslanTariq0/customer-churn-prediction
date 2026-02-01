import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 1. Load dataset
df = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

print("Shape:", df.shape)
print("Churn distribution:\n", df['Churn'].value_counts(normalize=True))

# 2. Clean data
# Fix TotalCharges (has spaces → convert to number)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna()  # Drop few rows with missing TotalCharges

# Drop useless column
df = df.drop(['customerID'], axis=1)

# Convert Yes/No to 1/0
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# 3. One-hot encode all categorical columns
categorical = [
    'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
    'PaperlessBilling', 'PaymentMethod'
]
df = pd.get_dummies(df, columns=categorical, drop_first=True)

# 4. Features & target
X = df.drop('Churn', axis=1)
y = df['Churn']

feature_names = X.columns.tolist()

# 5. Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 6. Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 7. Train model (Random Forest – good & simple)
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight='balanced'
)
model.fit(X_train, y_train)

# 8. Check performance
preds = model.predict(X_test)
print("Accuracy:", round(accuracy_score(y_test, preds), 3))
print("\nClassification Report:\n", classification_report(y_test, preds))

# 9. Save files
joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(feature_names, 'feature_names.pkl')

print("Done! Model saved.")