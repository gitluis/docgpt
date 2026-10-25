#!/usr/bin/env python3
"""
Streamlit chat app for DocGPT.

Run it from the repository root so the folders in `.env` resolve:

    streamlit run docgpt/app.py

The vector store is built from `DOCUMENTS_DIR` once, when the first page
loads, and the resulting chain is shared by every session until the server
stops. Use "Rebuild index" in the sidebar after adding, changing or removing
documents.
"""

# modules
import os
import time
import openai

import streamlit as st

from run import main


# functions
@st.cache_resource(show_spinner="Indexing documents...")
def load_qa():
    """
    Build the retrieval QA chain once and reuse it across reruns and sessions.

    Returns
    -------
    qa : langchain.chains.RetrievalQA
        Retrieval QA chain returned by `run.main()`.
    """

    return main()


def display_header():
    """ Display the app's title and description. """

    st.title("FCC Copilot")
    st.write("Interact with document(s) using a GPT AI model.")

    return


def display_sidebar():
    """ Display the sidebar with a button that rebuilds the index. """

    with st.sidebar:
        st.write("Added, changed or removed documents? Rebuild the index to use them.")

        # clear the cached chain so the next load_qa() call rebuilds it
        if st.button("Rebuild index"):
            load_qa.clear()

    return


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
    display_header()
    display_sidebar()

    # build the chain on first run; later reruns reuse the cached chain
    try:
        qa = load_qa()
    except ValueError as e:
        # e.g. no supported documents found or an invalid setting
        st.error(str(e))
        st.stop()

    app(qa)