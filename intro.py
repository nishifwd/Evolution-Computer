# Set page config
st.set_page_config(page_title="Nur Fatihah Amani's Portfolio", page_icon="🎨", layout="wide")

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
st.markdown("<h3 class='subtitle'>Hi, I'm Nur Fatihah Amani! Explore my projects below.</h3>", unsafe_allow_html=True)

# Introduction
st.write("""
I am passionate about software development, AI, and data science. Below, you'll find some of the projects I've developed using Streamlit.
""")

# List of projects
projects = [
    {"title": "Mental Ease - Mental Health Tracking App", "description": "A mental health app for students to track their mood and well-being."},
    {"title": "EcoWatch - Water Quality Monitoring", "description": "A machine learning-based system for water pollution detection."},
    {"title": "Flight Route Optimization", "description": "An AI-driven project optimizing flight routes using evolutionary strategies."},
    {"title": "AI Chatbot", "description": "An LLM-powered chatbot for answering user queries."}
]

# Display projects
for project in projects:
    st.markdown(f"""
        <div class='project-box'>
            <h4>{project['title']}</h4>
            <p>{project['description']}</p>
        </div>
    """, unsafe_allow_html=True)

# Footer
st.write("---")
st.write("Feel free to connect with me on LinkedIn or GitHub!")
