#!/usr/bin/env python3

# modules
import os
import openai

import loaders
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

openai.api_type = os.environ.get("OPENAI_API_TYPE")
openai.api_base = os.environ.get("OPENAI_API_BASE")
openai.api_version = os.environ.get("OPENAI_API_VERSION")
openai.api_key = os.environ.get("OPENAI_API_KEY")

llm_model_name = os.environ.get("OPENAI_API_MODEL_NAME")
llm_deployment_name = os.environ.get("OPENAI_API_DEPLOYMENT_NAME")

embeddings_model_name = os.environ.get("OPENAI_EMBEDDINGS_MODEL_NAME")
embeddings_deployment_name = os.environ.get("OPENAI_EMBEDDINGS_MODEL_NAME")


# functions
def main():
    """ Run DocGPT. """
    
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

    # --------------------- #
    # Create local database #
    # --------------------- #
    
    # remove existing vectorstore
    if storage.vectorstore_exists(vectorstore_db_dir):
        storage.remove_vectorstore(vectorstore_db_dir, verbose=True)
        
    # create new vectorstore
    db = storage.create_vectorstore(splits, vectorstore_db_dir, embeddings_model_name, openai.api_key, verbose=True)
    db.persist()

    # -------------------------- #
    # Configure LLM using OpenAI #
    # -------------------------- #
    
    chat = ChatOpenAI(temperature=0.70, engine=llm_deployment_name, model_name=llm_model_name)

    # ---------------------- #
    # Configure QA Retrieval #
    # ---------------------- #

    # set up a vectorstore database retriever to look for answers in all splits
    retriever = db.as_retriever(search_kwargs={"k": len(splits)})
    
    # create a retrieval QA chain with a retriever to look for answers
    qa = RetrievalQA.from_chain_type(llm=chat, chain_type="stuff", retriever=retriever, return_source_documents=False)
    
    return qa


if __name__ == "__main__":
    main()