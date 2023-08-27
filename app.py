import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import streamlit as st
import pickle

# Load the data
df = pd.read_csv('merged_data11.csv')
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Initialize LabelEncoder for 'Location'
label_encoder = LabelEncoder()
label_encoder.fit(df['Location'])

# Initialize StandardScaler
scaler = StandardScaler()
scaler.fit(df[['Year', 'rural pop', 'urban pop', 'electric_rural']])

# function to make predictions
def make_prediction(location, year, rural_population, urban_population, electric_rural):
    # Create DataFrame
    input_data = pd.DataFrame({
        'Location': [location],
        'Year': [year],
        'rural pop': [rural_population],
        'urban pop': [urban_population],
        'electric_rural': [electric_rural]
    })
    
    # Encode 'Location' using the label encoder
    input_data['Location'] = label_encoder.transform(input_data['Location'])
    # Scale the user input data using the scaler
    user_input_scale = pd.DataFrame(scaler.fit_transform(input_data), columns=input_data.columns)
  
    prediction_scores = model.predict_proba(user_input_scale)

    class_labels = df['Value_class'].unique()
    predict_class_label = class_labels[prediction_scores.argmax()]
    return predict_class_label

# Streamlit app
st.title("Electricity Class Prediction")

# Input fields for user
location = st.selectbox("Select Location", df['Location'].unique())
year = st.number_input("Select Year", value=2023, step=1)
rural_population = st.number_input("Select Rural Population (%)", min_value=0, max_value=100, value=50)
urban_population = 100 - rural_population
st.info(f"Urban Population (%): {urban_population}")
electric_rural = st.number_input("Select electric rural", min_value=0, value=0)

# Predict button
if st.button("Predict"):
    predicted_class_label = make_prediction(location, year, rural_population, urban_population, electric_rural)
    
    # Map the class label to a more descriptive message
    if predicted_class_label == "Low":
        predicted_message = "Low household electricity value"
    elif predicted_class_label == "Medium":
        predicted_message = "Medium household electricity value"
    elif predicted_class_label == "High":
        predicted_message = "High household electricity value"
    else:
        predicted_message = "Unknown"
    
    st.write(f"Predicted Value Class: {predicted_message}")