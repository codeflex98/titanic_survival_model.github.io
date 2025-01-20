import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import base64

# Load the trained model
model_path = os.path.join(os.path.dirname(__file__), 'titanic suvival model.pkl')
model = joblib.load(model_path)

# App title and description
st.title("Titanic Survivor Prediction")
st.write("This app predicts whether a passenger survived the Titanic disaster based on input features.")

# Input form for user features
st.sidebar.header("Passanger Details")

# User input form
def user_input_features():
    pclass = st.sidebar.selectbox("Passenger Class (Pclass)", [1, 2, 3], index=2)
    sex = st.sidebar.selectbox("Sex", ["male", "female"])
    age = st.sidebar.slider("Age", 0, 80, 30)
    sibsp = st.sidebar.slider("Number of Siblings/Spouses Aboard (SibSp)", 0, 8, 0)
    parch = st.sidebar.slider("Number of Parents/Children Aboard (Parch)", 0, 6, 0)
    fare = st.sidebar.slider("Fare", 0.0, 512.0, 32.2)
    embarked = st.sidebar.selectbox("Port of Embarkation", ["C", "Q", "S"], index=2)

    # Encode gender as numerical
    gender_encoded = 1 if sex == "Male" else 0
    embarked_encoded = 2 if embarked == "C" else (1 if embarked == "Q" else 0)
    
    # Create a dataframe for the input
    data = {
        "GENDER": gender_encoded,
        "AGE": age,
        "SIBLINGS": sibsp,
        "NUMBER OF PARENTS": parch,
        "TICKET PRICE": fare,
        "PORT OF EMBARKATION": embarked_encoded,
    }
    
    return pd.DataFrame(data, index=[0])
        
    
input_data = user_input_features()
print(f"Input data shape: {input_data.shape}")
    
# Button to trigger prediction
if st.sidebar.button("Predict"):
    prediction = model.predict(input_data)[0]  # Assuming binary classification (0 or 1)
    prediction_proba = model.predict_proba(input_data)[0]

       # Display results
    if prediction == 0:
        st.success(f"The transaction is  Non-Fraudulent")
        gif_url_1 = "https://media1.tenor.com/m/n8DB4bmpduIAAAAd/yeah-bwoi-grin.gif"
        st.markdown(f'<img src="{gif_url_1}" width="600" height="400">', unsafe_allow_html=True)
    else:
        st.error(f"The transaction is  Fraudulent")
        gif_url_1 = "https://media.tenor.com/9kVFrGqvcwsAAAAM/fraud-troll.gif"
        st.markdown(f'<img src="{gif_url_1}" width="600" height="400">', unsafe_allow_html=True)