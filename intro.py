import streamlit as st

# Set page config
st.set_page_config(page_title="Muhamad Danish's Portfolio", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
        .main-title {
            text-align: center;
            font-size: 36px;
            font-weight: bold;
        }
        .subtitle {
            text-align: center;
            font-size: 20px;
        }
        .project-box {
            border-radius: 10px;
            padding: 15px;
            background-color: #f4f4f4;
            margin: 10px 0px;
        }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 class='main-title'>👋 Welcome to My Portfolio</h1>", unsafe_allow_html=True)
st.markdown("<h3 class='subtitle'>Hi, I'm Muhamad Danish! Explore my projects below.</h3>", unsafe_allow_html=True)

# Introduction
st.write("""
I am passionate about software development, AI, and data science. Below, you'll find some of the projects I've developed using Streamlit.
""")

# Footer
st.write("---")
st.write("Feel free to connect with me on LinkedIn or GitHub!")
