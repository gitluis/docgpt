#!/usr/bin/env python3

# modules
from langchain.text_splitter import RecursiveCharacterTextSplitter


# functions
def split_documents(documents, chunk_size=500, chunk_overlap=50, verbose=False):
    """
    Split documents into multiple texts (or splits).

    Parameters
    ----------
    documents : list of Documents
        List of Document objects.
    chunk_size : int
        Number of characters per split.
    chunk_overlap : int
        //
    verbose : bool
        Used for verbose mode. True is ON, False is OFF.

    Returns
    -------
    splits : list of Documents
        A list of content splits in Document objects.
    """

    splits = list()

    if documents:
        if verbose:
            print(f"Found {len(documents)} documents...")
    
        # split document into multiple texts
        ts = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        splits = ts.split_documents(documents)
    
        if verbose:
            print(f"Split into {len(splits)} splits of texts")
            print(f"Each split has {chunk_size} tokens max.")
            
    else:
        print(f"No documents available to split")
    
    return splits