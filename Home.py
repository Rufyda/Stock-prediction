import streamlit as st
import requests
from streamlit_lottie import st_lottie

st.set_page_config(
    page_title="Home",
    page_icon="🚀",
)



st.title("Final graduate project at Epsilon AI")


def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    
    return r.json()

lottie_coding = load_lottieurl("https://lottie.host/499df7a8-c2a6-4670-90e3-117195004bb3/Lbq24MPBD3.json")

with st.container():  
    st.write("---")
    left_column, right_column = st.columns(2)
    with left_column:
        st.header("AI-Driven Stock Prediction Using Yahoo Finance Data")
        st.write("""
                    The project focuses on stock analysis and prediction, utilizing data scraping, analysis, and machine learning techniques.      
                    The project begins with data collection, where i scraped historical stock data from Yahoo Finance. 
                    This data serves as the foundation for my analysis and model-building processes.
                 """) 
    with right_column:
        st_lottie(lottie_coding, height=350, key="coding")

