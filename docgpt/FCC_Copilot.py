#!/usr/bin/env python3

# modules
import os
import time
import openai

import streamlit as st

from messages import questions


# functions
def app():
    """ Run Sreamlit UI. """

    st.title("📚🤖 BPPR FCC Co-Pilot")
    st.write("Interact with document(s) using a generative AI model.")
    
    # init chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # display chat history on app re-run
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # take user input
    query = st.chat_input("Enter your question(s) here...")
    
    if query:
        # display user message in chat message container
        with st.chat_message("user"):
            st.markdown(query)
        
        # add user message to chat history
        st.session_state.messages.append({"role": "user", "content": query})
        
        # send query to assistant to obtain a response
        response = questions[query]

        # display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            # simulate stream of response with milliseconds delay
            for sentence in response.split("\n"):
                for word in sentence.split():
                    full_response += word + " "
                    time.sleep(0.03)
            
                    # add a blinking cursor to simulate typing
                    message_placeholder.markdown(full_response + "▌")

                # add new line for the next content
                full_response += "\n"

            # display entire response
            message_placeholder.markdown(full_response)
        	
        # add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        
    return


if __name__ == "__main__":
    app()