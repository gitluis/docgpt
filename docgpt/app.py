#!/usr/bin/env python3

# modules
import os
import time
import openai

import streamlit as st

from run import main


# functions
def set_default_model():
    """ Set a default model. """
    
    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-4-32k"
    
    return


def init_chat_history():
    """ Initiatilize chat history. """
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    return


def display_chat_history():
    """ Display chat history on app re-run. """
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    return


def app(qa):
    """ Run Sreamlit UI. """

    st.title("FCC Copilot")
    st.write("Interact with document(s) using a GPT AI model.")

    # init
    set_default_model()
    init_chat_history()
    display_chat_history()

    # take user input
    query = st.chat_input("Enter your question(s) here...")
    
    if query:
        # display user message in chat message container
        with st.chat_message("user"):
            st.markdown(query)
        
        # add user message to chat history
        st.session_state.messages.append({"role": "user", "content": query})
        
        # send query to assistant to obtain a response
        output = qa(query)
        response = output.get("result")

        # display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)

        # add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        
    return


if __name__ == "__main__":
    qa = main()
    app(qa)