#!/usr/bin/env python3
""" Split LangChain documents into overlapping chunks for embedding. """

# modules
from langchain.text_splitter import RecursiveCharacterTextSplitter


# functions
def split_documents(documents, chunk_size=500, chunk_overlap=50, verbose=False):
    """
    Split documents into multiple texts (or splits).

    Uses LangChain's `RecursiveCharacterTextSplitter`, which prefers to break
    on paragraphs, then lines, then words. Each split keeps the metadata
    (such as "source") of the document it came from, plus "start_index": the
    character position where the split starts in that document, used to put
    splits back in reading order (see `retrieval.sort_in_reading_order`).

    Parameters
    ----------
    documents : list of Documents
        List of Document objects.
    chunk_size : int
        Maximum number of characters per split.
    chunk_overlap : int
        Number of characters that may overlap per split.
    verbose : bool
        Used for verbose mode. True is ON, False is OFF. The printed
        "tokens" figures are character counts.

    Returns
    -------
    splits : list of Documents
        A list of content splits in Document objects. Empty if `documents`
        is empty.
    """

    splits = list()

    if documents:
        if verbose:
            print(f"Found {len(documents)} documents...")
    
        # split document into multiple texts
        ts = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True)
        splits = ts.split_documents(documents)
    
        if verbose:
            print(f"Split into {len(splits)} splits of texts")
            print(f"Each split with a max. limit of {chunk_size} tokens")

            tokens = 0
            
            # count  real amount of tokens in all splits
            for split in splits:
                tokens += len(split.page_content)
                
            print(f"Real # of Tokens: {tokens:,}")
            
    else:
        print(f"No documents available to split")
    
    return splits