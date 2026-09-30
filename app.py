"""
Streamlit chat interface for the parking assistant.
Run with: streamlit run app.py
"""

import streamlit as st
import asyncio
from parking_assistant.service import process_chat_message


# Page configuration
st.set_page_config(
    page_title="Parking Assistant",
    page_icon="🚗",
    layout="centered"
)

st.title("🚗 Parking Lot Assistant")
st.caption("Ask me anything about our parking facility!")

# Initialize chat history and session ID in session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    import uuid
    st.session_state.session_id = str(uuid.uuid4())

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("How can I help you today?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Run async function with session ID for state tracking
            response = asyncio.run(
                process_chat_message(
                    user_message=prompt,
                    session_id=st.session_state.session_id,
                    chat_history=st.session_state.messages[:-1]
                )
            )
            st.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
