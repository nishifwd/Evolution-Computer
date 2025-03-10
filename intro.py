st.set_page_config(page_title='Welcome to EcoWatch', page_icon='🌊', layout='centered')

# Main title
st.title('🌍 Welcome to EcoWatch')

# Introduction text
st.markdown(
    """
    **EcoWatch** is a smart water monitoring system that detects and classifies water pollution 
    using machine learning. Our goal is to provide real-time insights to ensure cleaner and safer water sources.
    
    🌱 **Key Features:**
    - 📊 Real-time water quality monitoring
    - 🤖 Machine learning-based pollution classification
    - 📍 Interactive data visualization
    - 🔔 Smart alerts for potential contamination
    
    """
)

# Sidebar navigation
st.sidebar.title('Navigation')
st.sidebar.markdown("Select a page to explore different features.")

# Footer
st.markdown("---")
st.markdown("💙 Developed by Nur Fatihah Amani & Smartlink Innovation")
