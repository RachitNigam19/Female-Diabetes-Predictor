import numpy as np
import pickle
import streamlit as st

# loading the saved model
loaded_model = pickle.load(open('C:\Users\Rachit Nigam\OneDrive\Desktop\Deployment Diabetes ML Model\ trained_model.sav', 'rb'))
model_accuracy = 0.89  # Replace with your model's accuracy

# Creating a function for prediction
def diabetes_prediction(user_input):
    prediction = loaded_model.predict(user_input)
    if prediction[0] == 1:
        return '🛑 The person is Diabetic'
    else:
        return '✅ The person is Non-Diabetic'

def main():
    # Streamlit configuration for dark mode
    st.set_page_config(page_title="Diabetes Prediction", layout="centered", initial_sidebar_state="expanded")
    st.markdown(
        """
        <style>
        body {
            color: #ffffff;
            background-color: #121212;
        }
        .stButton>button {
            background-color: #6c63ff;
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Title and description
    st.title('PMI Female Diabetes Web App 🩺')
    st.write("This app uses machine learning to predict if a female has diabetes based on health parameters.")

    # Sidebar
    with st.sidebar:
        st.header("Model Information")
        st.write(f"**Model Accuracy:** {model_accuracy * 100:.2f}%")
        st.write("The prediction is based on a machine learning model trained with health data.")

    # Input fields with tooltips
    Pregnancies = st.text_input("Number of Pregnancies", help="Enter the number of pregnancies.")
    Glucose = st.text_input("Glucose Level (mg/dL)", help="Enter the glucose level in mg/dL.")
    BloodPressure = st.text_input("Blood Pressure (mm Hg)", help="Enter the blood pressure in mm Hg.")
    SkinThickness = st.text_input("Skin Thickness (mm)", help="Enter the skin thickness in mm.")
    Insulin = st.text_input("Insulin Level (µU/mL)", help="Enter the insulin level in µU/mL.")
    BodyMassIndex = st.text_input("Body Mass Index (BMI)", help="Enter the BMI value (e.g., 24.5).")
    DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function", help="Enter the pedigree function value (e.g., 0.134).")
    Age = st.text_input("Age (years)", help="Enter the patient's age.")

    # Add prediction button and logic
    diagnosis = ""
    if st.button('Diabetes Test Result'):
        try:
            user_input = np.array([[int(Pregnancies), int(Glucose), int(BloodPressure), int(SkinThickness), 
                                     int(Insulin), float(BodyMassIndex), float(DiabetesPedigreeFunction), int(Age)]])
            diagnosis = diabetes_prediction(user_input)
        except ValueError:
            diagnosis = "⚠️ Please enter valid numeric values for all fields."

    # Display the prediction
    st.success(diagnosis)

if __name__ == '__main__':
    main()
