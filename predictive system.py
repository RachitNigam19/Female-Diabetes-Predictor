import numpy as np
import pickle 

# Loading the Saving Model 
loaded_model = pickle.load(open('C:/Users/Rachit Nigam/OneDrive/Desktop/Deployment Diabetes ML Model/trained_model.sav', 'rb'))

input1 = float(input("Enter the no. of Pregnancies: "))
input2 = float(input("Glucose Level (e.g., 80): "))
input3 = float(input("Blood Pressure (e.g., 70): "))  # Added Blood Pressure input
input4 = float(input("Skin Thickness (e.g., 29): "))
input5 = float(input("Insulin Level: "))
input6 = float(input("BMI (e.g., 33.6): "))
input7 = float(input("Diabetes Pedigree Function (e.g., 0.134): "))
input8 = float(input("Age of the Patient: "))

# Creating the input array with 8 features
user_input = np.array([[input1, input2, input3, input4, input5, input6, input7, input8]])

# Reshaping and standardizing input data
input_data_reshaped = user_input.reshape(1, -1)

# Making a prediction
prediction = loaded_model.predict(input_data_reshaped)

if prediction[0] == 1:
    print("The person is likely Diabetic.")
else:
    print("The person is likely Non-Diabetic.")