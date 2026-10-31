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

When every chunk is sent, they go in reading order (file, page, position),
not in order of similarity to the question, so the model reads each document
as written. `count_tokens` and `get_context_window` tell whether all of them
fit in the chat model's context window.

This module has no imports from the rest of the package, so it works both
from `docgpt/run.py` (`import retrieval`) and from the notebooks
(`from docgpt import retrieval`). `format_sources` describes where the
retrieved chunks came from, for citing them under an answer.
"""

# modules
import os
import re

from typing import List

from langchain.schema import BaseRetriever, Document


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

# Context window, in tokens, of common chat models. Names are matched by
# prefix, in this order, after lowercasing and treating "gpt-35" (Azure's
# spelling) as "gpt-3.5". Set `OPENAI_API_CONTEXT_WINDOW` for anything else.
MODEL_CONTEXT_WINDOWS = (
    ("gpt-4.1", 1_047_576),
    ("gpt-4o", 128_000),
    ("gpt-4-turbo", 128_000),
    ("gpt-4-1106", 128_000),
    ("gpt-4-0125", 128_000),
    ("gpt-4-vision", 128_000),
    ("gpt-4-32k", 32_768),
    ("gpt-4", 8_192),
    ("gpt-3.5-turbo-16k", 16_384),
    ("gpt-3.5-turbo", 4_096),
)

# used when the model is not listed above and no override is set
DEFAULT_CONTEXT_WINDOW = 8_192

# tokens kept free for the model's answer
ANSWER_TOKENS_RESERVED = 1_000

# tokens for the prompt template around the chunks, and per chunk for the
# blank lines that separate them
PROMPT_OVERHEAD_TOKENS = 200
TOKENS_PER_CHUNK_SEPARATOR = 2

# characters per token, used when tiktoken can't load its encoding (it
# downloads it on first use). Kept low so estimates err toward "too big".
CHARS_PER_TOKEN_ESTIMATE = 3.5


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


class WholeDocumentRetriever(BaseRetriever):
    """
    Retriever that returns the same chunks, in the same order, for every
    question. Used to send every chunk in reading order.

    Parameters
    ----------
    documents : list of Documents
        Chunks to return, usually from `get_all_chunks`.
    """

    documents: List[Document]

    def get_relevant_documents(self, query: str) -> List[Document]:
        return list(self.documents)

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        return list(self.documents)


def sort_in_reading_order(documents):
    """
    Sort chunks by file, then page (or CSV row), then position in the page.

    Parameters
    ----------
    documents : list of Documents
        Chunks with "source" and, when known, "page", "row" and "start_index"
        metadata (see `splitters.split_documents`). Chunks missing a value
        keep their relative order.

    Returns
    -------
    documents : list of Documents
        A new, sorted list.
    """

    def key(doc):
        metadata = doc.metadata or dict()
        place = metadata.get("page", metadata.get("row", 0))
        start = metadata.get("start_index", 0)

        return (
            str(metadata.get("source", "")),
            place if isinstance(place, int) else 0,
            start if isinstance(start, int) else 0,
        )

    return sorted(documents, key=key)


def get_all_chunks(db):
    """
    Read every chunk stored in a Chroma vector store, in reading order.

    Parameters
    ----------
    db : langchain.vectorstores.Chroma

    Returns
    -------
    documents : list of Documents
    """

    collection = db.get()
    documents = [
        Document(page_content=text, metadata=metadata or dict())
        for text, metadata in zip(collection["documents"], collection["metadatas"])
    ]

    return sort_in_reading_order(documents)


def count_tokens(texts, model_name=None):
    """
    Count the tokens in some texts, as the chat model would.

    Parameters
    ----------
    texts : list of str
    model_name : str or None
        Chat model name, e.g. "gpt-4-32k". Unknown or Azure-style names use
        the "cl100k_base" encoding shared by GPT-3.5 and GPT-4.

    Returns
    -------
    tokens : int
        Exact count, or an estimate from the number of characters if
        tiktoken can't load its encoding (for example, without internet
        access the first time).
    """

    try:
        import tiktoken

        try:
            encoding = tiktoken.encoding_for_model((model_name or "").replace("gpt-35", "gpt-3.5"))
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")

        return sum(len(encoding.encode(text)) for text in texts)

    except Exception:
        return int(sum(len(text) for text in texts) / CHARS_PER_TOKEN_ESTIMATE) + 1


def get_context_window(model_name=None, override=None):
    """
    Work out how many tokens the chat model accepts (prompt plus answer).

    Parameters
    ----------
    model_name : str or None
        Value of `OPENAI_API_MODEL_NAME`, e.g. "gpt-4-32k" or "gpt-35-turbo".
    override : str, int or None
        Value of `OPENAI_API_CONTEXT_WINDOW`. When set, it is used as is.

    Returns
    -------
    tokens : int
        From `override`, else `MODEL_CONTEXT_WINDOWS`, else
        `DEFAULT_CONTEXT_WINDOW`.

    Raises
    ------
    ValueError
        If `override` is set but is not a positive whole number.
    """

    override = str(override if override is not None else "").strip().replace(",", "").replace("_", "")

    if override:
        try:
            tokens = int(override)
        except ValueError:
            tokens = 0

        if tokens < 1:
            raise ValueError(f"OPENAI_API_CONTEXT_WINDOW must be a positive whole number, not '{override}'")

        return tokens

    name = re.sub(r"^gpt-35", "gpt-3.5", (model_name or "").strip().lower())

    for prefix, tokens in MODEL_CONTEXT_WINDOWS:
        if name.startswith(prefix):
            return tokens

    return DEFAULT_CONTEXT_WINDOW


def get_prompt_tokens(chunk_tokens, n_chunks, question_tokens=0):
    """
    Estimate the size of a "stuff" prompt holding some chunks and a question.

    Parameters
    ----------
    chunk_tokens : int
        Tokens in the chunks' text (see `count_tokens`).
    n_chunks : int
        Number of chunks.
    question_tokens : int
        Tokens in the question.

    Returns
    -------
    tokens : int
        Estimated prompt size. Add `ANSWER_TOKENS_RESERVED` before comparing
        it with the context window.
    """

    return chunk_tokens + n_chunks * TOKENS_PER_CHUNK_SEPARATOR + PROMPT_OVERHEAD_TOKENS + question_tokens


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
    retriever : VectorStoreRetriever or WholeDocumentRetriever
        When every chunk is retrieved (`all`, or a number at least as large
        as the store), a `WholeDocumentRetriever` holding the store's chunks
        in reading order, read once here. Otherwise a retriever that searches
        the store for each question.

    Raises
    ------
    ValueError
        If either setting is invalid or the store is empty.
    """

    n_splits = db._collection.count()

    if n_splits < 1:
        raise ValueError("The vector store has no chunks to retrieve from!")

    search_type, search_kwargs = get_search_settings(target_sources, search_type, n_splits)

    if search_kwargs["k"] >= n_splits:
        if verbose:
            print(f"Retrieving all {n_splits} chunks per question, in reading order")

        return WholeDocumentRetriever(documents=get_all_chunks(db))

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
