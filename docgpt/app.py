#!/usr/bin/env python3
"""
Streamlit chat app for DocGPT.

Run it from the repository root so the folders in `.env` resolve:

    streamlit run docgpt/app.py

The vector store is built from `DOCUMENTS_DIR` once, when the first page
loads, and the resulting chains are shared by every session until the server
stops. Use "Rebuild index" in the sidebar after adding, changing or removing
documents.

Each question goes to one of two chains (see `routing.py`): the top
`TARGET_SOURCES` chunks, or, for summary requests that fit in the model's
context window, the whole document. A note under each answer says which was
used, with a button to answer again the other way. "Chunks sent to the model"
in the sidebar overrides the automatic choice.
"""

# modules
import os
import time
import openai

import streamlit as st

import routing

from retrieval import format_sources
from run import main


# constants
OVERRIDE_LABELS = {
    routing.AUTO: "Auto (whole document for summaries)",
    routing.TOP_K: "Always top chunks",
    routing.WHOLE: "Always whole document",
}


# functions
@st.cache_resource(show_spinner="Indexing documents...")
def load_qa():
    """
    Build the retrieval QA chains once and reuse them across reruns and sessions.

    Returns
    -------
    chains : run.DocChains
        The chains returned by `run.main()`.
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

        # clear the cached chains so the next load_qa() call rebuilds them
        if st.button("Rebuild index"):
            load_qa.clear()

    return


def display_mode_setting(chains):
    """
    Show the "Chunks sent to the model" override in the sidebar.

    Parameters
    ----------
    chains : run.DocChains

    Returns
    -------
    override : str
        `routing.AUTO`, `routing.TOP_K` or `routing.WHOLE`.
    """

    with st.sidebar:
        if chains.covers_all:
            st.caption(f"Every question gets all {chains.n_splits} chunks (TARGET_SOURCES).")
            return routing.AUTO

        override = st.radio(
            "Chunks sent to the model",
            routing.OVERRIDES,
            format_func=lambda value: OVERRIDE_LABELS[value],
            key="mode_override",
        )

        size = f"~{chains.whole_prompt_tokens():,} of {chains.context_window:,} tokens"

        if chains.whole_fits():
            st.caption(f"Top chunks: {chains.k} of {chains.n_splits}. Whole document: {size}.")
        else:
            st.caption(f"Top chunks: {chains.k} of {chains.n_splits}. "
                       f"The whole document doesn't fit in the model's context window ({size}), "
                       "so top chunks are always used.")

    return override


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


def display_sources(sources):
    """
    Show the files an answer was based on, under the answer.

    Parameters
    ----------
    sources : list of str
        Entries from `retrieval.format_sources`, e.g. "bitcoin.pdf (pages 3, 4)".
    """

    if sources:
        st.caption("Sources: " + " · ".join(sources))

    return


def describe_mode(chains, mode, reason):
    """
    Say which chunks an answer was based on, e.g. "Whole document (all 58
    chunks, ~14,210 tokens) · summary request detected".

    Parameters
    ----------
    chains : run.DocChains
    mode : str
        `routing.TOP_K` or `routing.WHOLE`.
    reason : str or None
        From `run.DocChains.choose_mode`.

    Returns
    -------
    label : str
    """

    if mode == routing.WHOLE:
        label = f"Whole document (all {chains.n_splits} chunks, ~{chains.whole_prompt_tokens():,} tokens)"
    else:
        label = f"Top {chains.k} of {chains.n_splits} chunks"

    return f"{label} · {reason}" if reason else label


def request_redo(question, mode, reason):
    """ Button callback: answer `question` again in `mode` on the next rerun. """

    st.session_state.redo = {"question": question, "mode": mode, "reason": reason}

    return


def display_redo_button(chains, message, index):
    """
    Offer to answer a message's question again with the other mode.

    Parameters
    ----------
    chains : run.DocChains
    message : dict
        An assistant message from the chat history.
    index : int
        Its position in the history, for a unique button key.
    """

    if chains.covers_all or not message.get("question"):
        return

    if message.get("mode") == routing.WHOLE:
        st.button(
            f"Answer again with the top {chains.k} chunks",
            key=f"redo_{index}",
            on_click=request_redo,
            args=(message["question"], routing.TOP_K, "answered again on request"),
        )
    elif chains.whole_fits(message["question"]):
        st.button(
            "Answer again with the whole document",
            key=f"redo_{index}",
            on_click=request_redo,
            args=(message["question"], routing.WHOLE, "answered again on request"),
        )

    return


def display_chat_history(chains, show_redo=True):
    """
    Redraw the messages stored in the session state after each rerun.

    Parameters
    ----------
    chains : run.DocChains
    show_redo : bool
        Show the "Answer again" button under the latest answer. Turned off
        while a new answer is being produced.
    """

    messages = st.session_state.messages

    for index, message in enumerate(messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            display_sources(message.get("sources"))

            if message.get("mode_label"):
                st.caption(message["mode_label"])

            if show_redo and message["role"] == "assistant" and index == len(messages) - 1:
                display_redo_button(chains, message, index)

    return


def answer(chains, question, mode, reason):
    """
    Answer a question, show the answer and add it to the chat history.

    Parameters
    ----------
    chains : run.DocChains
    question : str
    mode : str
        `routing.TOP_K` or `routing.WHOLE`.
    reason : str or None
        Why `mode` was chosen, shown under the answer.
    """

    try:
        with st.spinner("Answering..."):
            output = chains.ask(question, mode)
    except openai.error.OpenAIError as e:
        # e.g. the prompt exceeded the model's context window
        st.error(f"The model returned an error: {e}")
        return

    response = output.get("result")
    sources = format_sources(output.get("source_documents"))
    mode_label = describe_mode(chains, mode, reason)
    message = {
        "role": "assistant",
        "content": response,
        "sources": sources,
        "mode_label": mode_label,
        "question": question,
        "mode": mode,
    }

    # add assistant response to chat history
    st.session_state.messages.append(message)

    # display assistant response, its sources, the mode used and the
    # "Answer again" button in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
        display_sources(sources)
        st.caption(mode_label)
        display_redo_button(chains, message, len(st.session_state.messages) - 1)

    return


def app(chains, override=routing.AUTO):
    """
    Render the Streamlit chat UI and answer the latest question.

    Parameters
    ----------
    chains : run.DocChains
        The chains returned by `run.main()`. Each question goes to the chain
        `chains.choose_mode` picks; its "result" key is shown as the answer,
        followed by the files of the chunks in "source_documents" and which
        mode was used. Earlier messages are displayed but not sent to the
        chain.
    override : str
        Sidebar setting: `routing.AUTO`, `routing.TOP_K` or `routing.WHOLE`.
    """

    # init
    set_default_model()
    init_chat_history()

    # take user input, or a request to answer the last question again
    query = st.chat_input("Enter your question(s) here...")
    redo = st.session_state.pop("redo", None)

    display_chat_history(chains, show_redo=not (query or redo))

    if query:
        # display user message in chat message container
        with st.chat_message("user"):
            st.markdown(query)
        
        # add user message to chat history
        st.session_state.messages.append({"role": "user", "content": query})

        # pick the chunks to send, then send query to assistant to obtain a response
        mode, reason = chains.choose_mode(query, override)
        answer(chains, query, mode, reason)

    elif redo:
        answer(chains, redo["question"], redo["mode"], redo["reason"])

    return


if __name__ == "__main__":
    display_header()
    display_sidebar()

    # build the chains on first run; later reruns reuse the cached chains
    try:
        chains = load_qa()
    except ValueError as e:
        # e.g. no supported documents found or an invalid setting
        st.error(str(e))
        st.stop()

    override = display_mode_setting(chains)

    app(chains, override)