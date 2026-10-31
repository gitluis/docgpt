#!/usr/bin/env python3
"""
Build DocGPT's retrieval QA chains from the documents folder.

Settings are read from `.env` when this module is imported (see
`env-example.txt`): `DOCUMENTS_DIR`, `VECTORSTORE_DB_DIR`, `TARGET_SOURCES`,
`RETRIEVER_SEARCH_TYPE` and the Azure OpenAI `OPENAI_*` variables, including
the optional `OPENAI_API_CONTEXT_WINDOW`.

Running this file directly builds the vector store and exits, which is a
quick way to check that documents load and embeddings work:

    python docgpt/run.py
"""

# modules
import os
import openai

import loaders
import retrieval
import routing
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

# tokens the chat model accepts; empty uses a table of known models (see retrieval.py)
llm_context_window = os.environ.get("OPENAI_API_CONTEXT_WINDOW")

embeddings_model_name = os.environ.get("OPENAI_EMBEDDINGS_MODEL_NAME")
# defaults to the model name when not set
embeddings_deployment_name = os.environ.get("OPENAI_EMBEDDINGS_DEPLOYMENT_NAME") or embeddings_model_name


# classes
class DocChains:
    """
    The two QA chains the app chooses between, plus what it needs to choose.

    Both chains share one vector store and one chat model, so having two
    costs no extra embeddings.

    Attributes
    ----------
    top_k : langchain.chains.RetrievalQA
        Sends `k` chunks picked by `RETRIEVER_SEARCH_TYPE`.
    whole : langchain.chains.RetrievalQA
        Sends every chunk, in reading order. The same object as `top_k`
        when `TARGET_SOURCES` already covers every chunk.
    k : int
        Chunks `top_k` sends.
    n_splits : int
        Chunks in the vector store.
    chunk_tokens : int
        Tokens in all chunks (see `retrieval.count_tokens`).
    context_window : int
        Tokens the chat model accepts (see `retrieval.get_context_window`).
    model_name : str or None
        Chat model name, used to count a question's tokens.
    """

    def __init__(self, top_k, whole, k, n_splits, chunk_tokens, context_window, model_name=None):
        self.top_k = top_k
        self.whole = whole
        self.k = k
        self.n_splits = n_splits
        self.chunk_tokens = chunk_tokens
        self.context_window = context_window
        self.model_name = model_name

    @property
    def covers_all(self):
        """ True when top-k mode already sends every chunk (`TARGET_SOURCES` is `all` or large). """

        return self.k >= self.n_splits

    def whole_prompt_tokens(self, question=""):
        """ Estimated prompt size, in tokens, when every chunk is sent with `question`. """

        question_tokens = retrieval.count_tokens([question], self.model_name) if question else 0

        return retrieval.get_prompt_tokens(self.chunk_tokens, self.n_splits, question_tokens)

    def whole_fits(self, question=""):
        """ Whether every chunk, `question` and room for the answer fit in the context window. """

        return self.whole_prompt_tokens(question) + retrieval.ANSWER_TOKENS_RESERVED <= self.context_window

    def choose_mode(self, question, override=routing.AUTO):
        """
        Pick the mode for a question (see `routing.choose_mode`).

        Returns
        -------
        mode : str
            `routing.TOP_K` or `routing.WHOLE`. Always `routing.WHOLE` when
            `covers_all`, since both chains then send the same chunks.
        reason : str or None
        """

        if self.covers_all:
            return routing.WHOLE, "TARGET_SOURCES covers every chunk"

        return routing.choose_mode(question, override, self.whole_fits(question))

    def ask(self, question, mode=routing.TOP_K):
        """
        Answer a question with the chain for `mode`.

        Returns
        -------
        output : dict
            The chain's output: the answer in "result" and the chunks it was
            given in "source_documents".
        """

        chain = self.whole if mode == routing.WHOLE else self.top_k

        return chain(question)


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
    chains : DocChains
        Two "stuff" chains over the same store: `chains.top_k` sends
        `TARGET_SOURCES` chunks (5 by default), picked by
        `RETRIEVER_SEARCH_TYPE`, and `chains.whole` sends every chunk in
        reading order. Call `chains.choose_mode(question)` to pick one and
        `chains.ask(question, mode)` to answer; the answer is in the "result"
        key of the returned dict and the chunks it was given are in
        "source_documents" (see `retrieval.format_sources`).

    Raises
    ------
    ValueError
        If `DOCUMENTS_DIR` has no supported documents with text, or if
        `TARGET_SOURCES`, `RETRIEVER_SEARCH_TYPE` or
        `OPENAI_API_CONTEXT_WINDOW` is invalid.

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
    context_window = retrieval.get_context_window(llm_model_name, llm_context_window)

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
    qa = RetrievalQA.from_chain_type(llm=chat, chain_type="stuff", retriever=retriever, return_source_documents=True)

    # and a second chain over the same store that sends every chunk, in reading order
    whole_retriever = retriever

    if not isinstance(retriever, retrieval.WholeDocumentRetriever):
        whole_retriever = retrieval.build_retriever(db, "all")

    qa_whole = qa

    if whole_retriever is not retriever:
        qa_whole = RetrievalQA.from_chain_type(llm=chat, chain_type="stuff", retriever=whole_retriever, return_source_documents=True)

    # count tokens once, to tell later whether every chunk fits in one prompt
    chunk_tokens = retrieval.count_tokens([doc.page_content for doc in whole_retriever.documents], llm_model_name)
    k = retrieval.get_target_sources(target_sources, len(whole_retriever.documents))

    chains = DocChains(qa, qa_whole, k, len(whole_retriever.documents), chunk_tokens, context_window, llm_model_name)

    print(f"All chunks: ~{chunk_tokens:,} tokens; context window: {context_window:,} tokens; "
          f"whole-document mode {'fits' if chains.whole_fits() else 'does not fit'}")

    return chains


if __name__ == "__main__":
    main()