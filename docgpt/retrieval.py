#!/usr/bin/env python3
"""
Choose which chunks DocGPT sends to the model with each question.

Two settings, read from `.env` by the app and the notebooks, control it:

- `TARGET_SOURCES`: how many chunks to send (default 5), or "all" to send
  every chunk (whole-document mode, for summaries and "list everything"
  questions).
- `RETRIEVER_SEARCH_TYPE`: how to pick them. "mmr" (default) takes the most
  relevant chunks while skipping near-duplicates; "similarity" takes the most
  relevant chunks as they are.

This module has no imports from the rest of the package, so it works both
from `docgpt/run.py` (`import retrieval`) and from the notebooks
(`from docgpt import retrieval`). `format_sources` describes where the
retrieved chunks came from, for citing them under an answer.
"""

# modules
import os


# constants
DEFAULT_TARGET_SOURCES = 5
DEFAULT_SEARCH_TYPE = "mmr"
SEARCH_TYPES = ("mmr", "similarity")

# MMR first fetches the `fetch_k` most similar chunks, then picks `k` of them,
# trading relevance for variety. fetch_k is 4 times k, and at least 20.
MMR_FETCH_K_MULTIPLIER = 4
MMR_MIN_FETCH_K = 20

# 1 = rank by relevance only, 0 = rank by variety only
MMR_LAMBDA_MULT = 0.5


# functions
def get_target_sources(value, n_splits):
    """
    Work out how many chunks to retrieve for each question.

    Parameters
    ----------
    value : str, int or None
        Value of `TARGET_SOURCES`: a positive whole number, or "all" to
        retrieve every chunk. Empty or None uses `DEFAULT_TARGET_SOURCES`.
    n_splits : int
        Number of chunks in the vector store.

    Returns
    -------
    k : int
        Number of chunks to retrieve, never more than `n_splits`.

    Raises
    ------
    ValueError
        If `value` is not a positive whole number or "all".
    """

    value = str(value if value is not None else "").strip()

    if not value:
        k = DEFAULT_TARGET_SOURCES
    elif value.lower() == "all":
        k = n_splits
    else:
        try:
            k = int(value)
        except ValueError:
            k = 0

        if k < 1:
            raise ValueError(f"TARGET_SOURCES must be a positive whole number or 'all', not '{value}'")

    return min(k, n_splits)


def get_search_type(value):
    """
    Validate the search type used to pick chunks.

    Parameters
    ----------
    value : str or None
        Value of `RETRIEVER_SEARCH_TYPE`: "mmr" or "similarity", in any
        letter case. Empty or None uses `DEFAULT_SEARCH_TYPE`.

    Returns
    -------
    search_type : str
        "mmr" or "similarity".

    Raises
    ------
    ValueError
        If `value` is not one of `SEARCH_TYPES`.
    """

    value = (value or "").strip().lower()

    if not value:
        return DEFAULT_SEARCH_TYPE

    if value not in SEARCH_TYPES:
        raise ValueError(f"RETRIEVER_SEARCH_TYPE must be one of {', '.join(SEARCH_TYPES)}, not '{value}'")

    return value


def get_search_settings(target_sources, search_type, n_splits):
    """
    Turn the two settings into arguments for `db.as_retriever()`.

    Parameters
    ----------
    target_sources : str, int or None
        Value of `TARGET_SOURCES` (see `get_target_sources`).
    search_type : str or None
        Value of `RETRIEVER_SEARCH_TYPE` (see `get_search_type`).
    n_splits : int
        Number of chunks in the vector store.

    Returns
    -------
    search_type : str
        "mmr" or "similarity". When every chunk is retrieved (`all`, or a
        number at least as large as the store), this is always "similarity":
        there is nothing left to choose between, so MMR would only add work.
    search_kwargs : dict
        `{"k": k}` for similarity search; MMR also gets `fetch_k` and
        `lambda_mult`.

    Raises
    ------
    ValueError
        If either setting is invalid.
    """

    k = get_target_sources(target_sources, n_splits)
    search_type = get_search_type(search_type)

    if search_type == "mmr" and k < n_splits:
        fetch_k = min(n_splits, max(k * MMR_FETCH_K_MULTIPLIER, MMR_MIN_FETCH_K))
        return "mmr", {"k": k, "fetch_k": fetch_k, "lambda_mult": MMR_LAMBDA_MULT}

    return "similarity", {"k": k}


def build_retriever(db, target_sources=None, search_type=None, verbose=False):
    """
    Create a retriever over a Chroma vector store.

    Parameters
    ----------
    db : langchain.vectorstores.Chroma
        The vector store to search. Its chunk count is read from the store,
        so chunks added earlier with `storage.update_vectorstore` count too.
    target_sources : str, int or None
        Value of `TARGET_SOURCES`: a positive whole number, or "all".
        Default: 5.
    search_type : str or None
        Value of `RETRIEVER_SEARCH_TYPE`: "mmr" or "similarity".
        Default: "mmr".
    verbose : bool
        Used for verbose mode. True is ON, False is OFF.

    Returns
    -------
    retriever : langchain.vectorstores.base.VectorStoreRetriever

    Raises
    ------
    ValueError
        If either setting is invalid or the store is empty.
    """

    n_splits = db._collection.count()

    if n_splits < 1:
        raise ValueError("The vector store has no chunks to retrieve from!")

    search_type, search_kwargs = get_search_settings(target_sources, search_type, n_splits)

    if verbose:
        print(f"Retrieving {search_kwargs['k']} of {n_splits} chunks per question using '{search_type}' search")

    return db.as_retriever(search_type=search_type, search_kwargs=search_kwargs)


def format_sources(documents):
    """
    Describe where retrieved chunks came from, for citing them under an answer.

    Chunks from the same file are grouped into one entry, in the order the
    files first appear. PDF pages and CSV rows are numbered from 1.

    Parameters
    ----------
    documents : list of Documents
        Chunks returned by a retriever, such as the "source_documents" key of
        a RetrievalQA output built with `return_source_documents=True`.

    Returns
    -------
    sources : list of str
        One entry per file, e.g. "bitcoin.pdf (pages 3, 4)", "data.csv
        (row 12)" or "notes.txt". Empty if `documents` is empty.
    """

    files = dict()

    for doc in documents or []:
        metadata = doc.metadata or dict()
        name = os.path.basename(str(metadata.get("source", "unknown source")))
        places = files.setdefault(name, {"page": set(), "row": set()})

        # PyMuPDFLoader numbers pages and CSVLoader numbers rows from 0
        for key in ("page", "row"):
            if isinstance(metadata.get(key), int):
                places[key].add(metadata[key] + 1)

    sources = list()

    for name, places in files.items():
        details = list()

        for key in ("page", "row"):
            numbers = sorted(places[key])

            if numbers:
                label = key if len(numbers) == 1 else f"{key}s"
                details.append(f"{label} {', '.join(str(n) for n in numbers)}")

        sources.append(f"{name} ({'; '.join(details)})" if details else name)

    return sources
