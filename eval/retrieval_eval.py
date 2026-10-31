#!/usr/bin/env python3
"""
Measure how well DocGPT's retriever finds the right chunks.

For each question in a questions file, every retriever setting under test
(`RETRIEVER_SEARCH_TYPE` x `TARGET_SOURCES`) retrieves chunks, and the script
checks whether any of them comes from the page where the answer is. No chat
model is called, so the only cost is embeddings.

Run it from the repository root:

    python eval/retrieval_eval.py
    python eval/retrieval_eval.py --k 3 5 8 10 --search-types mmr similarity --misses

Embeddings use the same settings as the app (`.env`): Azure OpenAI when
`OPENAI_API_KEY` is set, otherwise the local Hugging Face model in
`EMBEDDINGS_MODEL_NAME`. Pass `--embeddings local` to force the local model.
The index is built in a temporary folder; `VECTORSTORE_DB_DIR` is not touched.

Questions file format (see `eval/questions.json`, written for the sample
`documents/bitcoin.pdf`):

    [{"question": "...", "source": "bitcoin.pdf", "pages": [3]}, ...]

`source` is a file name in the documents folder. `pages` lists the pages
(numbered from 1) that contain the answer; leave it out for files without
pages, and any chunk of that file counts. Questions about files that aren't
in the documents folder are skipped.
"""

# modules
import os
import sys
import csv
import json
import argparse
import tempfile

from shutil import rmtree

# make the app's modules importable (they import each other by bare name)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "docgpt"))

import openai

import run
import loaders
import retrieval
import splitters
import storage

from langchain.embeddings.base import Embeddings


# classes
class CachedEmbeddings(Embeddings):
    """ Embeddings wrapper that embeds each distinct text only once. """

    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.cache = dict()

    def embed_documents(self, texts):
        missing = [t for t in dict.fromkeys(texts) if t not in self.cache]

        if missing:
            self.cache.update(zip(missing, self.embeddings.embed_documents(missing)))

        return [self.cache[t] for t in texts]

    def embed_query(self, text):
        if text not in self.cache:
            self.cache[text] = self.embeddings.embed_query(text)

        return self.cache[text]


# functions
def parse_args():
    """ Read command-line options. """

    here = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(description="Measure how well the retriever finds the right chunks.")
    parser.add_argument("--documents", default=run.documents_dir or "documents",
                        help="Folder with the documents to index (default: DOCUMENTS_DIR).")
    parser.add_argument("--questions", default=os.path.join(here, "questions.json"),
                        help="Questions file (default: eval/questions.json).")
    parser.add_argument("--k", nargs="+", default=["3", "5", "8"],
                        help="TARGET_SOURCES values to compare (default: 3 5 8). 'all' is allowed.")
    parser.add_argument("--search-types", nargs="+", default=list(retrieval.SEARCH_TYPES),
                        help="RETRIEVER_SEARCH_TYPE values to compare (default: mmr similarity).")
    parser.add_argument("--chunk-size", type=int, default=500, help="Characters per chunk (default: 500, as in run.py).")
    parser.add_argument("--chunk-overlap", type=int, default=50, help="Characters of overlap (default: 50, as in run.py).")
    parser.add_argument("--embeddings", choices=("auto", "openai", "local"), default="auto",
                        help="auto: Azure OpenAI if OPENAI_API_KEY is set, otherwise local (default).")
    parser.add_argument("--misses", action="store_true", help="List the questions each setting missed.")
    parser.add_argument("--csv", help="Also write the results table to this CSV file.")

    return parser.parse_args()


def load_questions(path, splits):
    """
    Read the questions file and keep the questions whose file was indexed.

    Returns
    -------
    questions : list of dict
        Each with "question", "source" and "pages" (a set, empty for any page).
    """

    with open(path, encoding="utf-8") as f:
        items = json.load(f)

    indexed = {os.path.basename(s.metadata.get("source", "")) for s in splits}
    questions = list()

    for item in items:
        if item["source"] not in indexed:
            print(f"Skipping question about '{item['source']}' (not in the documents folder): {item['question']}")
            continue

        questions.append({"question": item["question"], "source": item["source"], "pages": set(item.get("pages") or [])})

    return questions


def is_hit(doc, question):
    """ True if a retrieved chunk comes from the file and page that hold the answer. """

    if os.path.basename(doc.metadata.get("source", "")) != question["source"]:
        return False

    if not question["pages"]:
        return True

    page = doc.metadata.get("page")

    return isinstance(page, int) and page + 1 in question["pages"]


def evaluate(db, questions, target_sources, search_type, n_splits):
    """
    Run every question through one retriever setting.

    Returns
    -------
    row : dict
        hit_rate: share of questions with at least one chunk from the right page.
        mrr: mean of 1 / rank of the first right chunk (0 when there is none).
        pages: mean number of distinct pages among the retrieved chunks.
        chars: mean number of characters sent to the model per question.
        misses: the questions with no chunk from the right page.
    """

    used_type, kwargs = retrieval.get_search_settings(target_sources, search_type, n_splits)
    retriever = db.as_retriever(search_type=used_type, search_kwargs=kwargs)

    hits, rr, pages, chars, misses = 0, 0.0, 0, 0, list()

    for q in questions:
        docs = retriever.get_relevant_documents(q["question"])
        ranks = [i for i, d in enumerate(docs, start=1) if is_hit(d, q)]

        if ranks:
            hits += 1
            rr += 1 / ranks[0]
        else:
            misses.append(q["question"])

        pages += len({(d.metadata.get("source"), d.metadata.get("page")) for d in docs})
        chars += sum(len(d.page_content) for d in docs)

    n = len(questions)

    return {
        "search_type": search_type,
        "used": used_type,
        "k": kwargs["k"],
        "hit_rate": hits / n,
        "mrr": rr / n,
        "pages": pages / n,
        "chars": chars / n,
        "misses": misses,
    }


def main():
    """ Index the documents, run every setting and print a comparison table. """

    args = parse_args()

    # ------------------------ #
    # Load and split documents #
    # ------------------------ #

    docs = loaders.load_documents(args.documents)
    splits = splitters.split_documents(docs, args.chunk_size, args.chunk_overlap)

    if not splits:
        sys.exit(f"No supported documents with text found in '{args.documents}'.")

    questions = load_questions(args.questions, splits)

    if not questions:
        sys.exit("None of the questions are about files in the documents folder.")

    # validate every setting before spending requests on embeddings
    for search_type in args.search_types:
        for k in args.k:
            retrieval.get_search_settings(k, search_type, len(splits))

    # ------------------------------ #
    # Build a temporary vector store #
    # ------------------------------ #

    use_openai = args.embeddings == "openai" or (args.embeddings == "auto" and openai.api_key)

    if use_openai:
        if not openai.api_key:
            sys.exit("--embeddings openai needs OPENAI_API_KEY in .env.")

        model = run.embeddings_model_name
        embedding = storage.get_embedding_function(model, openai.api_key, run.embeddings_deployment_name)
    else:
        model = os.environ.get("EMBEDDINGS_MODEL_NAME") or "all-MiniLM-L6-v2"
        embedding = storage.get_embedding_function(model)

    print(f"Indexing {len(splits)} chunks with '{model}' and testing {len(questions)} questions...")

    tmp = tempfile.mkdtemp(prefix="docgpt-eval-")

    try:
        from chromadb.config import Settings
        from langchain.vectorstores import Chroma

        settings = Settings(chroma_db_impl="duckdb+parquet", persist_directory=tmp, anonymized_telemetry=False)
        db = Chroma.from_documents(splits, embedding=CachedEmbeddings(embedding), persist_directory=tmp, client_settings=settings)

        rows = [evaluate(db, questions, k, t, len(splits)) for t in args.search_types for k in args.k]
    finally:
        rmtree(tmp, ignore_errors=True)

    # ------------- #
    # Print results #
    # ------------- #

    print()
    print(f"{'search type':<20} {'k':>4} {'hit rate':>9} {'MRR':>6} {'pages':>6} {'chars':>7}")

    for r in rows:
        label = r["search_type"] if r["used"] == r["search_type"] else f"{r['search_type']} (as {r['used']})"
        print(f"{label:<20} {r['k']:>4} {r['hit_rate']:>9.0%} {r['mrr']:>6.2f} {r['pages']:>6.1f} {r['chars']:>7,.0f}")

    print()
    print("hit rate: questions with at least one chunk from the right page")
    print("MRR: 1 / rank of the first right chunk, averaged (1.00 = always first)")
    print("pages: distinct pages among the chunks sent; chars: characters sent per question")

    if args.misses:
        for r in rows:
            if r["misses"]:
                print(f"\nMissed by {r['search_type']}, k={r['k']}:")

                for m in r["misses"]:
                    print(f"  - {m}")

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["search_type", "used_search_type", "k", "hit_rate", "mrr", "pages", "chars"])

            for r in rows:
                writer.writerow([r["search_type"], r["used"], r["k"], round(r["hit_rate"], 4), round(r["mrr"], 4), round(r["pages"], 2), round(r["chars"])])

        print(f"\nWrote {args.csv}")

    return


if __name__ == "__main__":
    main()
