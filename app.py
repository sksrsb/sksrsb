import numpy as np
import pandas as pd
import streamlit as st 
from sklearn import preprocessing
import pickle

model = pickle.load(open('model.pkl', 'rb'))
encoder_dict = pickle.load(open('encoder.pkl', 'rb')) 
cols=['age','relationship','race','gender','state','health factors']    

def main(): 
    st.title("Gun Violence Predictor")
    html_temp = """
    <div style="background:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;">Gun Violence Predictor</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    
    age = st.number_input("Age", 0) 
    relationship = st.selectbox("Relationship", ["Husband", "Not-in-family", "Other-relative", "Own-child", "Unmarried", "Wife", "Friend", "Neighbour", "Family", "Aquaintance"]) 
    race = st.selectbox("Race", ["Amer Indian Eskimo", "Asian Pac Islander", "Black", "Other", "White"]) 
    gender = st.selectbox("Gender", ["Female", "Male"]) 
    state = st.selectbox("State", ["new jersey", "new york", "Connecticut", "indiana", "illinos", "chicago", "alabama", "alaska", "arizona", "california", "ohio", "new hampshire", "new mexico", "south dakota", "kentucky", "oregon", "tennessee", "washington", "maryland", "Iran", "florida", "georgia", "hawaii", "idaho", "missouri", "nevada", "kansas", "maryland", "north carolina", "south carolina", "texas", "utah", "virginia"]) 
    health_factor = st.selectbox("Health Factor", [
        "Primary Care Physician Rate",
        "Dentist Rate",
        "% With Access to Exercise Opportunities",
        "Preventable Hospital Stays Rate",
        "Teen Birth Rate",
        "Child Mortality Rate",
        "Age-Adjusted Mortality",
        "Motor Vehicle Mortality Rate",
        "Chlamydia Rate",
        "% Food Insecure",
        "HIV Prevalence Rate",
        "% Low Birth Weight",
        "Food Environment Index",
        "Other Primary Care Providers Rate",
        "Infant Mortality Rate"
    ])
    
    if st.button("Predict"): 
        data = {'age': age, 'relationship': relationship, 'race': race, 'gender': gender, 'state': state, 'health_factor': health_factor}
        df = pd.DataFrame([data])
        
        for col in encoder_dict:
            le = preprocessing.LabelEncoder()
            le.classes_ = encoder_dict[col]
            df[col] = le.transform(df[col].astype(str))
        
        prediction = model.predict(df.values)
        output = int(prediction[0])
        if output == 1:
            text = "High"
        else:
            text = "Low"
        st.success('Gun violence risk is {}'.format(text))
      
if __name__=='__main__': 
    main()
