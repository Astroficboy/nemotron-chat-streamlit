import streamlit as st
from openai import OpenAI
import time
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("API_KEY")

# if not api_key:
#     st.error("API key not found. Please set the API key in the environment variables.")
#     st.stop()

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=api_key
)

# Function to generate a response from the model
def get_response(prompt):
    try:
        completion = client.chat.completions.create(
            model="nvidia/nemotron-4-340b-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            top_p=0.7,
            max_tokens=1024,
            stream=True
        )
        
        response = ""
        
        for chunk in completion:
            if chunk.choices[0].delta.content is not None:
                response += chunk.choices[0].delta.content + " "  # Append content with a space
                time.sleep(0.05)  # Add a slight delay for better streaming experience
        
        st.write(response)  # Write the entire response at once
        return response
    
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return ""


# Initialize session state for chat history
if "history" not in st.session_state:
    st.session_state.history = []

# Streamlit app layout
st.title("AI Chatbot")
st.write("Chat with the AI and explore the wonders of GPU computing!")

# User input
user_input = st.text_input("You: ", "")

if st.button("Send"):
    if user_input.strip() != "":
        # Add user message to history
        st.session_state.history.append({"role": "user", "content": user_input})
        
        # Get AI response
        with st.spinner("AI is thinking..."):
            ai_response = get_response(user_input)
        
        # Add AI response to history
        st.session_state.history.append({"role": "assistant", "content": ai_response})

# Display chat history
if st.session_state.history:
    for chat in st.session_state.history:
        if chat["role"] == "user":
            st.write(f"**You**: {chat['content']}")
        else:
            st.write(f"**AI**: {chat['content']}")

# Option to clear chat history
if st.button("Clear Chat"):
    st.session_state.history = []
    st.experimental_rerun()
