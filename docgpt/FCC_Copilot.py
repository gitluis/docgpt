#!/usr/bin/env python3
"""
Scripted Streamlit demo that replays pre-recorded answers.

Answers come from the `questions` dict in `messages.py`; no model is called
and no `.env` is needed. Only the questions stored there have answers; for any
other input the demo lists the questions it can answer.

    streamlit run docgpt/FCC_Copilot.py
"""

# modules
import os
import time
import openai

import streamlit as st

from messages import questions


# functions
def app():
    """
    Render the demo chat UI and replay the stored answer to the latest question.

    The answer is revealed word by word to simulate streaming. A question that
    is not a key of `messages.questions` gets a reply listing the questions
    that have stored answers.
    """

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
        response = questions.get(query.strip())

        # reply with the available questions when there is no stored answer
        if response is None:
            response = "This demo only has answers for these questions:\n\n"
            response += "\n".join(f"- {question}" for question in questions)

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