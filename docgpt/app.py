#!/usr/bin/env python3
"""
Streamlit chat app for DocGPT.

Run it from the repository root so the folders in `.env` resolve:

    streamlit run docgpt/app.py

Streamlit reruns this script after every interaction. Each rerun calls
`run.main()`, which rebuilds the vector store from `DOCUMENTS_DIR` before the
question is answered.
"""

# modules
import os
import time
import openai

import streamlit as st

from run import main


# functions
def set_default_model():
    """
    Store a default model name in the session state if none is set.

    Notes
    -----
    The stored value is not used to answer questions. The chat model comes
    from `OPENAI_API_DEPLOYMENT_NAME` and `OPENAI_API_MODEL_NAME` (see `run.py`).
    """
    
    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-4-32k"
    
    return


def init_chat_history():
    """ Initialize an empty chat history in the session state on first run. """
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    return


def display_chat_history():
    """ Redraw the messages stored in the session state after each rerun. """
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    return


def app(qa):
    """
    Render the Streamlit chat UI and answer the latest question.

    Parameters
    ----------
    qa : langchain.chains.RetrievalQA
        Retrieval QA chain returned by `run.main()`. It is called with the
        user's question and its "result" key is shown as the answer. Earlier
        messages are displayed but not sent to the chain.
    """

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