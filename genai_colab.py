import streamlit as st
from transformers import pipeline
from pyngrok import ngrok
import os

# Title of the app
st.title("🧠 Lightweight GenAI Chatbot")
st.caption("🚀 Runs on bare minimum CPU using `flan-t5-small`!")

# Start ngrok tunnel (for public access)
ngrok.set_auth_token("YOUR_NGROK_AUTH_TOKEN")  # Replace with your ngrok token
public_url = ngrok.connect(port="8501")
st.write(f"🌍 Public URL: {public_url}")

# Load model (optimized for CPU)
@st.cache_resource
def load_model():
    return pipeline("text2text-generation", model="google/flan-t5-small")

chatbot = load_model()

# Chat Input
user_input = st.text_input("Type your question:")

# Process response
if user_input:
    with st.spinner("🤖 Thinking..."):
        response = chatbot(user_input, max_length=100, do_sample=True)
        st.write("**AI Response:**", response[0]['generated_text'])
