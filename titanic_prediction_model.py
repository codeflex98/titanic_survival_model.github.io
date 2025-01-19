import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer

# Title and description
st.title("Titanic Survival Prediction")
st.markdown("This app predicts whether a passenger survived the Titanic disaster based on input features.")

# Sidebar for user input
st.sidebar.header("Passenger Input Features")

def user_input_features():
    pclass = st.sidebar.selectbox("Passenger Class (Pclass)", [1, 2, 3], index=2)
    sex = st.sidebar.selectbox("Sex", ["male", "female"], index=0)
    age = st.sidebar.slider("Age", 0, 80, 30)
    sibsp = st.sidebar.slider("Number of Siblings/Spouses Aboard (SibSp)", 0, 8, 0)
    parch = st.sidebar.slider("Number of Parents/Children Aboard (Parch)", 0, 6, 0)
    fare = st.sidebar.slider("Fare", 0.0, 512.0, 32.2)
    embarked = st.sidebar.selectbox("Port of Embarkation", ["C", "Q", "S"], index=2)

    data = {
        'Pclass': pclass,
        'Sex_male': 1 if sex == 'male' else 0,
        'Age': age,
        'SibSp': sibsp,
        'Parch': parch,
        'Fare': fare,
        'Embarked_Q': 1 if embarked == 'Q' else 0,
        'Embarked_S': 1 if embarked == 'S' else 0
    }
    return pd.DataFrame(data, index=[0])

input_data_prep = user_input_features()

# Load training data and train the Decision Tree model
@st.cache_data
def load_and_train_model():
    train_data = pd.read_csv('D:/Hochschule Fresenius notes (sem3)/Technical Applications and Data Management/Final_project/train.csv')

    # Preprocess training data
    def preprocess_data(data_prep):
        age_imputer = SimpleImputer(strategy='median')
        data_prep['Age'] = age_imputer.fit_transform(data_prep[['Age']])
        data_prep['Fare'] = data_prep['Fare'].fillna(data_prep['Fare'].median())
        embarked_imputer = SimpleImputer(strategy='most_frequent')
        data_prep['Embarked'] = embarked_imputer.fit_transform(data_prep[['Embarked']])

        data_prep = pd.get_dummies(data_prep, columns=['Sex', 'Embarked'], drop_first=True)
        data_prep.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)
        return data_prep

    train_cleaned = preprocess_data(train_data)
    X = train_cleaned.drop(columns=['Survived'])
    y = train_cleaned['Survived']

    model = DecisionTreeClassifier(random_state=42)
    model.fit(X, y)

    return model

model = load_and_train_model()

# Predict and display the result
if st.button("Predict Survival"):
    prediction = model.predict(input_data_prep)[0]
    prediction_text = "Survived" if prediction == 1 else "Did Not Survive"
    st.subheader(f"Prediction: {prediction_text}")

# Show input features
st.subheader("Passenger Features")
st.write(input_data_prep)
