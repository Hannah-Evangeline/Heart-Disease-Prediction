import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import pickle


# 1. Load Dataset

df = pd.read_csv("heart.csv")
print("✅ Dataset loaded successfully!\n")
print(df.head())



# 2. EDA Plots (Optional, can skip later)

sns.countplot(x='HeartDisease', data=df)
plt.title("Heart Disease Distribution")
plt.show()

sns.histplot(df['Age'], kde=True, bins=20)
plt.title("Age Distribution")
plt.show()



# 3. Heatmap (Fix: only numeric data)

numeric_df = df.select_dtypes(include='number')
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap (Numeric Only)")
plt.show()



# 4. Split Features & Target

X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]



# 5. Identify Column Types

numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X.select_dtypes(include=['object']).columns



# 6. Create Preprocessor (encode + scale)

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])



# 7. Build Pipeline (Preprocess + Model)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])



# 8. Train/Test Split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# 9. Train the Model

pipeline.fit(X_train, y_train)



# 10. Predict & Evaluate

y_pred = pipeline.predict(X_test)
print("\n✅ Classification Report:\n", classification_report(y_test, y_pred))
print("✅ Accuracy Score:", accuracy_score(y_test, y_pred))



# 11. Confusion Matrix

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()



# 12. Cross Validation

cv_score = cross_val_score(pipeline, X, y, cv=5)
print("\n✅ Cross Validation Score:", round(cv_score.mean(), 4))



# 13. Predict for a Single Patient

# Format: [Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope]
single_patient = pd.DataFrame([{
    'Age': 58,
    'Sex': 'M',
    'ChestPainType': 'ASY',
    'RestingBP': 130,
    'Cholesterol': 250,
    'FastingBS': 0,
    'RestingECG': 'Normal',
    'MaxHR': 140,
    'ExerciseAngina': 'Y',
    'Oldpeak': 1.5,
    'ST_Slope': 'Flat'
}])

single_prediction = pipeline.predict(single_patient)[0]
print("\n🧍 Single Patient Prediction:", "Heart Disease" if single_prediction == 1 else "No Heart Disease")



# 14. Save model

with open("heart_model.csv", "wb") as f:
    pickle.dump(pipeline, f)

print("\n✅ Model saved as 'heart_model.csv'")
