import pickle
import pandas as pd

# 1. Load the trained model from the .pkl file
with open("heart_model.pkl", "rb") as file:
    model = pickle.load(file)

# 2. Prepare input (same structure as training data)
new_patient = pd.DataFrame([{
    'Age': 65,
    'Sex': 'M',
    'ChestPainType': 'ASY',
    'RestingBP': 180,
    'Cholesterol': 320,
    'FastingBS': 1,
    'RestingECG': 'ST',
    'MaxHR': 95,
    'ExerciseAngina': 'Y',
    'Oldpeak': 4.2,
    'ST_Slope': 'Flat'
}])

new_patient2 = pd.DataFrame([{
    'Age': 47,
    'Sex': 'F',
    'ChestPainType': 'NAP',
    'RestingBP': 130,
    'Cholesterol': 250,
    'FastingBS': 0,
    'RestingECG': 'Normal',
    'MaxHR': 150,
    'ExerciseAngina': 'N',
    'Oldpeak': 1.0,
    'ST_Slope': 'Flat'
}])

# 3. Predict using the loaded model
result = model.predict(new_patient)[0]

result2 = model.predict(new_patient2)[0]


# 4. Print prediction result
print("💓 Prediction Result:", "Heart Disease" if result == 1 else "No Heart Disease")

print("💓 Prediction Result:", "Heart Disease" if result2 == 1 else "No Heart Disease")
