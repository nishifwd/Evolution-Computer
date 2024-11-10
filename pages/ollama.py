import streamlit as st
import ollama
import requests

# Set up the page layout
st.set_page_config(page_title="Ollama Chatbot", page_icon=":robot:", layout="wide")

# Title of the app
st.title("Ollama Chatbot")

# Define a function to send a message to Ollama and get the response
def get_ollama_response(message):
    try:
        # Attempt to get a response from the Ollama API
        response = ollama.chat(model="benevolentjoker/havenmini", messages=[{"role": "user", "content": message}])
        return response['text']
    except requests.exceptions.RequestException as e:
        # Handle connection errors or HTTP issues
        st.error(f"An error occurred while contacting Ollama: {e}")
        return None
    except KeyError:
        # Handle missing key in the response
        st.error("Invalid response structure from Ollama API.")
        return None

# Create a chat interface
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

# Display previous messages
for msg in st.session_state['messages']:
    if msg['role'] == 'user':
        st.markdown(f"**You:** {msg['content']}")
    else:
        st.markdown(f"**Ollama:** {msg['content']}")

# User input box
user_input = st.text_input("Your message:")

# On submit, get the response from Ollama
if user_input:
    # Add user message to session state
    st.session_state['messages'].append({"role": "user", "content": user_input})
    
    # Get the response from Ollama
    bot_response = get_ollama_response(user_input)
    
    if bot_response:
        # Add bot response to session state
        st.session_state['messages'].append({"role": "assistant", "content": bot_response})

    # Refresh the page to display new messages
    st.experimental_rerun()
