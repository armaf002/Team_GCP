import pickle
import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the dataset
df = pd.read_csv('merged_data11.csv')

# Initialize StandardScaler
scaler = StandardScaler()
# Scale the relevant columns
scaled_features = scaler.fit_transform(df[['Year', 'rural pop', 'urban pop', 'electric_rural', 'Location']])

# Load the trained model
with open('C:/Users/RIDWAN/Downloads/Team__GCP/model.pkl', 'rb') as file:
    model = pickle.load(file)

# Initialize LabelEncoder for 'Location'
label_encoder = LabelEncoder()
# Fit the encoder to the location data
label_encoder.fit(df['Location'])

def main():
    st.title("Global Household Electricity Value Prediction")

    # Create input form for user input
    location = st.selectbox("Select Location", label_encoder.classes_)
    year = st.number_input("Select Year", value=2023, step=1)
    
    # Number input for rural population
    rural_population = st.number_input("Select Rural Population (%)", min_value=0, max_value=100, value=50)
    urban_population = 100 - rural_population
    st.info(f"Urban Population (%): {urban_population}")
    
    # Number input for electric rural
    electric_rural = st.number_input("Select electric rural", min_value=0, value=0)

    if st.button('Predict'):
        # Encode the location using label_encoder
        encoded_location = label_encoder.transform([location])
        
        # Prepare the input data for scaling and prediction
        input_data = [[year, rural_population, urban_population, electric_rural, encoded_location]]
        scaled_input = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(scaled_input)

        # Display prediction
        st.success(f"The predicted electricity value for {location} in {year} is {prediction[0]}.")

if __name__ == "__main__":
    main()
