#!/usr/bin/env python3
"""
Create, check, update and remove DocGPT's local Chroma vector store.

The store uses Chroma's legacy "duckdb+parquet" persistence, which requires
chromadb 0.3.x (0.4 and later removed it).
"""

# modules
import os

from shutil import rmtree

from chromadb.config import Settings

from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings


# functions
def get_embedding_function(embeddings_model_name, openai_key="", embeddings_deployment_name=None):
    """
    Create the embedding function used to embed splits and queries.

    Parameters
    ----------
    embeddings_model_name : str
        Embeddings model name. With `openai_key`, an OpenAI embeddings model
        (e.g. "text-embedding-ada-002"); without it, a Hugging Face
        sentence-transformers model (e.g. "all-MiniLM-L6-v2").
    openai_key : str
        OpenAI key for embedding models. If empty, embeddings are computed
        locally with Hugging Face instead.
    embeddings_deployment_name : str, optional
        Azure OpenAI deployment that serves the embeddings model. Defaults to
        `embeddings_model_name`. Ignored without `openai_key`.

    Returns
    -------
    langchain.embeddings.base.Embeddings
        An OpenAIEmbeddings or HuggingFaceEmbeddings object.
    """

    if openai_key:
        # chunk_size=1 sends one text per embeddings request
        return OpenAIEmbeddings(
            model=embeddings_model_name,
            deployment=embeddings_deployment_name or embeddings_model_name,
            chunk_size=1,
        )

    return HuggingFaceEmbeddings(model_name=embeddings_model_name)


def vectorstore_exists(dir):
    """
    Checks whether a vectorstore storage exists in the directory specified or not.

    A directory counts as a vectorstore when it contains the
    "chroma-embeddings.parquet" and "chroma-collections.parquet" files and an
    "index" folder with more than 3 files.

    Parameters
    ----------
    dir : str
        File path to directory.

    Returns
    -------
    exists : bool
        Returns True if vectorstore exists, False otherwise.
    """

    exists = False

    index = os.path.join(dir, "index")
    embeddings = os.path.join(dir, "chroma-embeddings.parquet")
    collections = os.path.join(dir, "chroma-collections.parquet")

    # checks whether path to vectorstore's index exists or not
    if os.path.exists(index):
        # checks whether parquet files exist or not
        if os.path.exists(embeddings) and os.path.exists(collections):
            contents = os.listdir(index)
            
            # vectorstore is valid and exists if index has more than 3 files
            if len(contents) > 3:
                exists = True
    
    return exists


def remove_vectorstore(dir, verbose=False):
    """
    Remove directory and all of its contents.

    Everything in `dir` is deleted, not only the vectorstore files. Does
    nothing if `dir` does not exist.

    Parameters
    ----------
    dir : str
        File path to directory.
    verbose : bool
        Used for verbose mode. True is ON, False is OFF.
    """

    if verbose:
        print(f"Removing vectorstore at dir '{dir}'")
    
    # remove existing dir
    if os.path.exists(dir):
        rmtree(dir)
    
    return


def create_vectorstore(splits, dir, embeddings_model_name, openai_key="", verbose=False, embeddings_deployment_name=None):
    """
    Create a local vectorstore database to store splits.

    The directory is created if needed. Call `db.persist()` on the result to
    write the store to disk.

    Parameters
    ----------
    splits : list of Documents
        A list of content splits in Document objects.
    dir : str
        File path to directory where database will be stored.
    embeddings_model_name : str
        Embeddings model name. With `openai_key`, an OpenAI embeddings model
        (e.g. "text-embedding-ada-002"); without it, a Hugging Face
        sentence-transformers model (e.g. "all-MiniLM-L6-v2").
    openai_key : str
        OpenAI key for embedding models. If empty, embeddings are computed
        locally with Hugging Face instead. With a key, one embeddings request
        is sent per split.
    verbose : bool
        Used for verbose mode. True is ON, False is OFF.
    embeddings_deployment_name : str, optional
        Azure OpenAI deployment that serves the embeddings model. Defaults to
        `embeddings_model_name`. Ignored without `openai_key`.

    Returns
    -------
    db : Chroma
        A chroma vectorstore database.
    """

    if not os.path.exists(dir):
        # create directory
        os.makedirs(dir)

    # create word embedding function
    embedding = get_embedding_function(embeddings_model_name, openai_key, embeddings_deployment_name)
    
    # define client settings
    settings = Settings(chroma_db_impl='duckdb+parquet', persist_directory=dir, anonymized_telemetry=False)

    if verbose:
        print(f"Creating vectorstore with chromadb at dir '{dir}'")
        print(f"Creating embeddings using '{embeddings_model_name}' ...")

    # create vectorstore db and embed splits
    db = Chroma.from_documents(splits, embedding=embedding, persist_directory=dir, client_settings=settings)

    if verbose:
        print(f"Done!")

    return db


def update_vectorstore(splits, dir, embeddings_model_name="all-MiniLM-L6-v2", verbose=False, openai_key="", embeddings_deployment_name=None):
    """
    Update an existing local vectorstore database.

    Adds only the splits whose "source" file is not already in the store.
    Use the same embeddings settings the store was created with, so that new
    and existing vectors are comparable. Call `db.persist()` on the result to
    write the changes to disk. Not used by the app.

    Parameters
    ----------
    splits : list of Documents
        A list of content splits in Document objects.
    dir : str
        File path to directory where database will be stored.
    embeddings_model_name : str
        Embeddings model name. With `openai_key`, an OpenAI embeddings model;
        without it, a Hugging Face sentence-transformers model.
    verbose : bool
        Used for verbose mode. True is ON, False is OFF.
    openai_key : str
        OpenAI key for embedding models. If empty, embeddings are computed
        locally with Hugging Face instead.
    embeddings_deployment_name : str, optional
        Azure OpenAI deployment that serves the embeddings model. Defaults to
        `embeddings_model_name`. Ignored without `openai_key`.

    Returns
    -------
    db : Chroma
        A chroma vectorstore database.
    """

    new_splits = list()

    # create word embedding function
    embeddings = get_embedding_function(embeddings_model_name, openai_key, embeddings_deployment_name)
    
    # define client settings
    settings = Settings(chroma_db_impl='duckdb+parquet', persist_directory=dir, anonymized_telemetry=False)

    if verbose:
        print(f"Updating vectorstore at dir '{dir}'")

    # retrieve vectorstore database
    db = Chroma(persist_directory=dir, embedding_function=embeddings, client_settings=settings)

    # retrieve existing documents in vs to avoid duplicates (needs to be reviewed)
    collection = db.get()
    vs_documents = {metadata.get("source") for metadata in collection["metadatas"] if metadata}
    new_document_sources = [s.metadata["source"] for s in splits]
    new_splits = [split for doc, split in zip(new_document_sources, splits) if doc not in vs_documents]

    if new_splits:
        if verbose:
            print(f"Adding {len(new_splits)} new splits...")
        
        # add new splits
        db.add_documents(new_splits)

    elif verbose:
        print("No new splits to add")

    if verbose:
        print(f"Done!")
    
    return db