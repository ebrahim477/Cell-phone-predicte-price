import streamlit as st
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

def load_resources():
 
    model = joblib.load(r'.....\cell phone price\workspace\hf.pkl')
    return model

def main():
    st.title('Cell Phones Price Prediction')
    model = load_resources()


    input_data = {
        'ram': st.number_input('ram', min_value=1, value=4),
        'cpu_core': st.number_input('cpu_core', min_value=1, value=1),
        'internal mem': st.number_input('internal mem', min_value=1, value=1),
        'battery': st.number_input('battery', min_value=1, value=1000),
        'Front_Cam': st.number_input('Front Camera', min_value=1, value=8),
        'RearCam': st.number_input('RearCam', min_value=1, value=16),
        'cpu_freq': st.number_input('cpu_freq', min_value=1, value=1)
    }

    input_df = pd.DataFrame([input_data])

  
    input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)

    if st.button('Predict Phone Price'):
        prediction = model.predict(input_df)
        st.success(f'Predicted Price is: ${prediction[0]:.2f}')

if __name__ == '__main__':
    main()