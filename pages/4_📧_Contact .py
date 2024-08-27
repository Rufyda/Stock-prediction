import warnings
import requests
import streamlit as st
from streamlit_lottie import st_lottie
from PIL import Image

warnings.filterwarnings('ignore', category=DeprecationWarning)

st.set_page_config(page_title="Contct Me", page_icon=":📩:", layout="wide")


def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


# Use local CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


local_css("pages/style/style.css")

# ---- LOAD ASSETS ----
lottie_coding = load_lottieurl("https://lottie.host/148cda24-617a-4cbe-8e02-288eb7710759/SXI23XsDrZ.json")
img_satisfaction= Image.open("pages/images/satisfaction.png")
img_coffee = Image.open("pages/images/coffee.png")

# ---- HEADER SECTION ----
with st.container():
    st.subheader("Hi, I am Rufyda")
    st.title("A Data Scintest From Sudan")
    st.write(
        """
        I am a Data Scientist. I have capability, along with skills, to analyse and interpret the data in the way that can support the provider for making evidence based decison .
        """
    )
    st.write("[My Lincked in Profile >](https://www.linkedin.com/in/rufyda-rahma-96b656179?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=ios_app)")

# ---- WHAT I DO ----
with st.container():
    st.write("---")
    left_column, right_column = st.columns(2)
    with left_column:
        st.header("What I do")
        st.write(
            """My day-to-day work involves collecting, cleaning, and analyzing data to uncover trends and patterns. 

I use tools like Python, Mysql, and Power Bi to manipulate data and extract meaningful insights. 
                  
I launched an initiative to create a group where we apply projects on data by collecting, analyzing, and designing reports. 
We also hold weekly sessions to discuss the results we have reached.
            If you are interesting , consider subscribing and turning on the notifications.
            """
        )
        st.write("[YouTube Channel >](https://www.youtube.com/@user-data_projects)")
    with right_column:
        st_lottie(lottie_coding, height=400, key="coding")

# ---- PROJECTS ----
with st.container():
    st.write("---")
    st.header("My Projects")
    st.write("##")
    image_column, text_column = st.columns((1, 2))
    with image_column:
        st.image(img_coffee)
    with text_column:
        st.subheader("The Coffee Project")
        st.write(
            """
            The Coffee Project is designed to provide insights into the world of coffee through data analysis and visualization. The project involves collecting data related to coffee consumption, production, and market trends, and then applying various data science techniques to uncover meaningful patterns and insights.
            """
        )
        st.markdown("[Github...](https://github.com/Rufyda/Coffee-Project)")
with st.container():
    image_column, text_column = st.columns((1, 2))
    with image_column:
        st.image(img_satisfaction)
    with text_column:
        st.subheader("Customer Satisfaction Data Analysis")
        st.write(
            """
            The goal of this project is to explore and understand customer satisfaction metrics through data analysis. By examining customer feedback and satisfaction scores, the project aims to identify key factors influencing customer experiences and provide actionable recommendations for enhancing service quality.
            """
        )
        st.markdown("[Github...](https://github.com/Rufyda/Customer-satisfaction-data-analysis)")

# ---- CONTACT ----
with st.container():
    st.write("---")
    st.header("Get In Touch With Me!")
    st.write("##")

    # Documention: https://formsubmit.co/ !!! CHANGE EMAIL ADDRESS !!!
    contact_form = """
    <form action="https://formsubmit.co/harofida2020@gmail.com" method="POST">
        <input type="hidden" name="_captcha" value="false">
        <input type="text" name="name" placeholder="Your name" required>
        <input type="email" name="email" placeholder="Your email" required>
        <textarea name="message" placeholder="Your message here" required></textarea>
        <button type="submit">Send</button>
    </form>
    """
    left_column, right_column = st.columns(2)
    with left_column:
        st.markdown(contact_form, unsafe_allow_html=True)
    with right_column:
        st.empty()