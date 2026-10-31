#!/usr/bin/env python3
"""
Build DocGPT's retrieval QA chain from the documents folder.

Settings are read from `.env` when this module is imported (see
`env-example.txt`): `DOCUMENTS_DIR`, `VECTORSTORE_DB_DIR`, `TARGET_SOURCES`,
`RETRIEVER_SEARCH_TYPE` and the Azure OpenAI `OPENAI_*` variables.

Running this file directly builds the vector store and exits, which is a
quick way to check that documents load and embeddings work:

    python docgpt/run.py
"""

# modules
import os
import openai

import loaders
import retrieval
import splitters
import storage

from dotenv import load_dotenv

from langchain.chat_models import ChatOpenAI

from langchain.docstore.document import Document
from langchain.embeddings import OpenAIEmbeddings

from langchain.chains import RetrievalQA


# load environment
load_dotenv()

documents_dir = os.environ.get("DOCUMENTS_DIR")
vectorstore_db_dir = os.environ.get("VECTORSTORE_DB_DIR")

# number of chunks retrieved to answer each question ("all" sends every chunk)
target_sources = os.environ.get("TARGET_SOURCES")

# how those chunks are picked: "mmr" (default) or "similarity"
retriever_search_type = os.environ.get("RETRIEVER_SEARCH_TYPE")

openai.api_type = os.environ.get("OPENAI_API_TYPE")
openai.api_base = os.environ.get("OPENAI_API_BASE")
openai.api_version = os.environ.get("OPENAI_API_VERSION")
openai.api_key = os.environ.get("OPENAI_API_KEY")

llm_model_name = os.environ.get("OPENAI_API_MODEL_NAME")
llm_deployment_name = os.environ.get("OPENAI_API_DEPLOYMENT_NAME")

embeddings_model_name = os.environ.get("OPENAI_EMBEDDINGS_MODEL_NAME")
# defaults to the model name when not set
embeddings_deployment_name = os.environ.get("OPENAI_EMBEDDINGS_DEPLOYMENT_NAME") or embeddings_model_name


# functions
def main():
    """
    Index the documents and build a retrieval QA chain over them.

    Loads every file in `DOCUMENTS_DIR`, splits it into 500-character chunks
    with a 50-character overlap, replaces the Chroma store in
    `VECTORSTORE_DB_DIR` with a new one, and connects it to an Azure OpenAI
    chat model (temperature 0.7).

    Returns
    -------
    qa : langchain.chains.RetrievalQA
        A "stuff" chain that sends `TARGET_SOURCES` chunks (5 by default) to
        the model, picked by `RETRIEVER_SEARCH_TYPE` (see `retrieval.py`).
        Call it with a question string; the answer is in the "result" key of
        the returned dict.

    Raises
    ------
    ValueError
        If `DOCUMENTS_DIR` has no supported documents with text, or if
        `TARGET_SOURCES` or `RETRIEVER_SEARCH_TYPE` is invalid.

    Notes
    -----
    Each call deletes and re-creates the vector store and re-embeds every
    chunk, which costs one embeddings request per chunk. The Streamlit app
    caches the result so this happens once, not on every message.
    """
    
    # -------------- #
    # Load documents #
    # -------------- #
    
    # load documents inside the documents directory
    docs = loaders.load_documents(documents_dir, verbose=True)

    # --------------- #
    # Split documents #
    # --------------- #

    # set max. number of characters per split
    chunk_size = 500
    
    # set number of characters that may overlap per split
    chunk_overlap = chunk_size * 0.10

    splits = splitters.split_documents(docs, chunk_size, chunk_overlap, verbose=True)

    if not splits:
        raise ValueError(f"No supported documents with text found in directory '{documents_dir}'!")

    # validate retrieval settings before spending time and requests on embeddings
    retrieval.get_search_settings(target_sources, retriever_search_type, len(splits))

    # --------------------- #
    # Create local database #
    # --------------------- #
    
    # remove existing vectorstore
    if storage.vectorstore_exists(vectorstore_db_dir):
        storage.remove_vectorstore(vectorstore_db_dir, verbose=True)
        
    # create new vectorstore
    db = storage.create_vectorstore(
        splits,
        vectorstore_db_dir,
        embeddings_model_name,
        openai.api_key,
        verbose=True,
        embeddings_deployment_name=embeddings_deployment_name,
    )
    db.persist()

    # -------------------------- #
    # Configure LLM using OpenAI #
    # -------------------------- #
    
    chat = ChatOpenAI(temperature=0.70, engine=llm_deployment_name, model_name=llm_model_name)

    # ---------------------- #
    # Configure QA Retrieval #
    # ---------------------- #

    # set up a vectorstore database retriever that picks the chunks sent with each question
    retriever = retrieval.build_retriever(db, target_sources, retriever_search_type, verbose=True)
    
    # create a retrieval QA chain with a retriever to look for answers
    qa = RetrievalQA.from_chain_type(llm=chat, chain_type="stuff", retriever=retriever, return_source_documents=False)
    
    return qa


if __name__ == "__main__":
    main()