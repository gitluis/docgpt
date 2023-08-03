#!/usr/bin/env python3

# modules
import os

from shutil import rmtree

from chromadb.config import Settings

from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings


# functions
def vectorstore_exists(dir):
    """
    Checks whether a vectorstore storage exists in the directory specified or not.

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
        # checks wether parquet files exists or not
        if os.path.exists(embeddings) and os.path.exists(collections):
            contents = os.listdir(index)
            
            # vectorstore is valid and exists if index has more than 3 files
            if len(contents) > 3:
                exists = True
    
    return exists


def remove_vectorstore(dir, verbose=False):
    """ Remove directory and all of its contents. """

    if verbose:
        print(f"Removing vectorstore at dir '{dir}'")
    
    # remove existing dir
    if os.path.exists(dir):
        rmtree(dir)
    
    return


def create_vectorstore(splits, dir, openai_key="", embeddings_model_name="", verbose=False):
    """
    Create a local vectorstore database to store splits.

    Parameters
    ----------
    splits : list of Documents
        A list of content splits in Document objects.
    dir : str
        File path to directory where database will be stored.
    openai_key : str
        OpenAI key for embedding models.
    embeddings_model_name : str
        Embeddings model name.
    verbose : bool
        Used for verbose mode. True is ON, False is OFF.

    Returns
    -------
    db : Chroma
        A chroma vectorstore database.
    """

    if not os.path.exists(dir):
        # create directory
        os.mkdir(dir)

    # create word embedding function
    if openai_key:
        embedding = OpenAIEmbeddings(model=embeddings_model_name, chunk_size=1)
    else:
        embedding = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    
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


def update_vectorstore(splits, dir, embeddings_model_name="all-MiniLM-L6-v2", verbose=False):
    """
    Update an existing local vectorstore database.

    Parameters
    ----------
    splits : list of Documents
        A list of content splits in Document objects.
    dir : str
        File path to directory where database will be stored.
    embeddings_model_name : str
        Embeddings model name.
    verbose : bool
        Used for verbose mode. True is ON, False is OFF.

    Returns
    -------
    db : Chroma
        A chroma vectorstore database.
    """

    new_splits = list()

    # create word embedding function
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    
    # define client settings
    settings = Settings(chroma_db_impl='duckdb+parquet', persist_directory=dir, anonymized_telemetry=False)

    if verbose:
        print(f"Updating vectorstore at dir '{dir}'")

    # retrieve vectorstore database
    db = Chroma(persist_directory=dir, embedding_function=embeddings, client_settings=settings)

    # retrieve existing documents in vs to avoid duplicates (needs to be reviewed)
    collection = db.get()
    vs_documents = {metadata["source"] for metadata in collection["metadatas"]}
    new_document_sources = [s.metadata["source"] for s in splits]
    new_splits = [split for doc, split in zip(new_document_sources, splits) if doc not in vs_documents]

    if new_splits:
        if verbose:
            print(f"Adding {len(splits)} new splits...")
        
        # add new splits
        db.add_documents(new_splits)

    if verbose:
        print(f"Done!")
    
    return db