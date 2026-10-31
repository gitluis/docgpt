#!/usr/bin/env python3
"""
Decide, for each question, whether to send the top chunks or every chunk.

Most questions are about a specific detail, which the few chunks most similar
to the question answer best ("top k" mode, `TARGET_SOURCES` chunks). Summary
questions are about everything, so they get every chunk in reading order
("whole document" mode), as long as all of it fits in the model's context
window.

`is_summary_request` spots summary questions with a list of keywords, in
English and Spanish. `choose_mode` combines that with the app's override
setting and the fit check, and says why it chose what it did.

This module has no imports from the rest of the package, so it works both
from `docgpt/app.py` (`import routing`) and from the notebooks
(`from docgpt import routing`).
"""

# modules
import re


# constants
TOP_K = "top_k"
WHOLE = "whole"

# override settings offered in the app
AUTO = "auto"
OVERRIDES = (AUTO, TOP_K, WHOLE)

# Phrases that ask about the documents as a whole. Matched anywhere in the
# question, ignoring letter case. "resume" counts only when a Spanish word
# follows it ("resume el documento"): on its own it also means a CV or
# "continue" in English.
SUMMARY_PATTERNS = (
    # English
    r"\bsummar(y|ies|ize|ise|ized|ised|izing|ising|ization|isation)\b",
    r"\boverview\b",
    r"\btl;?\s?dr\b",
    r"\bgist\b",
    r"\b(key|main|core) (points?|takeaways?|ideas?|findings?|themes?)\b",
    r"\bwhat(\s+is|'s|’s|s) (this|the|that|it) ((document|doc|file|pdf|paper|report|presentation|deck) )?(all )?about\b",
    r"\bwhat are (these|the) (documents|docs|files|pdfs|papers|reports) (all )?about\b",
    # Spanish
    r"\bres[uú]men(es)?\b",
    r"\bresumir\b",
    r"\bres[uú]me(lo|la|los|las|me|nos)\b",
    r"\bresúme\b",
    r"\bresume (el|la|los|las|este|esta|estos|estas|esto|eso|lo|todo|cada)\b",
    r"\bsintetiza\w*\b",
    r"\bpuntos (clave|principales|importantes)\b",
    r"\bideas (clave|principales)\b",
    r"\bde qu[eé] (se )?trata\b",
)

SUMMARY_REGEX = re.compile("|".join(f"(?:{pattern})" for pattern in SUMMARY_PATTERNS), re.IGNORECASE)


# functions
def is_summary_request(question):
    """
    Tell whether a question asks for a summary or overview of the documents.

    Parameters
    ----------
    question : str

    Returns
    -------
    is_summary : bool
        True if any of `SUMMARY_PATTERNS` appears in the question, e.g.
        "Summarize the paper", "Give me the key points", "TL;DR?",
        "Hazme un resumen" or "¿De qué trata el documento?".
    """

    return bool(SUMMARY_REGEX.search(question or ""))


def choose_mode(question, override=AUTO, whole_fits=True):
    """
    Pick the mode for one question.

    Parameters
    ----------
    question : str
    override : str
        `AUTO` (decide from the question), `TOP_K` or `WHOLE` (always use
        that mode).
    whole_fits : bool
        Whether every chunk plus the question fits in the model's context
        window. When False, whole-document mode is never chosen.

    Returns
    -------
    mode : str
        `TOP_K` or `WHOLE`.
    reason : str
        Why, in a few words, for showing under the answer.

    Raises
    ------
    ValueError
        If `override` is not one of `OVERRIDES`.
    """

    if override not in OVERRIDES:
        raise ValueError(f"override must be one of {', '.join(OVERRIDES)}, not '{override}'")

    if override == TOP_K:
        return TOP_K, "set in the sidebar"

    if override == WHOLE:
        if whole_fits:
            return WHOLE, "set in the sidebar"

        return TOP_K, "the whole document doesn't fit in the model's context window"

    if is_summary_request(question):
        if whole_fits:
            return WHOLE, "summary request detected"

        return TOP_K, "summary request detected, but the whole document doesn't fit in the model's context window"

    return TOP_K, None
