#!/usr/bin/env python3
"""
Load files into LangChain `Document` objects.

The loader is chosen from the file extension using `DOCUMENT_LOADERS`.
Supported extensions: .csv, .pdf, .txt, .pptx, .doc and .docx. Loading .doc
files requires LibreOffice (`soffice`) to be installed.
"""

# modules
import os

from tqdm import tqdm

from langchain.document_loaders import (
    CSVLoader,
    PyMuPDFLoader,
    TextLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)


# constants
DOCUMENT_LOADERS = {
    # map file extensions to document loaders
    ".csv": (CSVLoader, {}),
    ".pdf": (PyMuPDFLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
}


# functions
def load_document(fp):
    """
    Load a single document.

    Parameters
    ----------
    fp : str
        Document file path. Its extension (in any letter case) must be one of
        the keys of `DOCUMENT_LOADERS`.

    Returns
    -------
    list of Documents
        A list of Document objects (for example, one per PDF page or one per
        CSV row).

    Raises
    ------
    ValueError
        If the file's extension is not supported.
    """

    # retrieve file's extension type (e.g. ".pdf"), ignoring letter case
    ext = os.path.splitext(fp)[1].lower()

    # retrieve doc's loader based on its extension type
    if ext in DOCUMENT_LOADERS:
        cls, args = DOCUMENT_LOADERS[ext]
        loader = cls(fp, **args)
    else:
        raise ValueError(f"File type of '{fp}' is not supported!")

    return loader.load()


def is_supported(fp):
    """
    Check whether a file can be loaded.

    Parameters
    ----------
    fp : str
        File path.

    Returns
    -------
    bool
        True if the file's extension (in any letter case) has a loader in
        `DOCUMENT_LOADERS`, False otherwise.
    """

    return os.path.splitext(fp)[1].lower() in DOCUMENT_LOADERS


def load_documents(dir, ignore_files=(), verbose=False):
    """
    Load every supported file in a directory.

    Only the top level of `dir` is read. Subfolders and hidden files (names
    starting with ".", such as ".DS_Store") are skipped silently; other files
    with an unsupported extension are skipped with a message. Files are loaded
    in alphabetical order.

    Parameters
    ----------
    dir : str
        File path to directory containing documents.
    ignore_files : list of str
        Names of files inside `dir` (names, not paths) that will not be
        loaded.
    verbose : bool
        Used for verbose mode. True is ON, False is OFF.

    Returns
    -------
    list of Documents
        A list of Document objects. Empty if `dir` has no supported files.

    Raises
    ------
    ValueError
        If `dir` does not exist or is not a directory.
    """

    files = list()
    loaded_files = list()

    # checks whether directory specified exists or not
    if not os.path.isdir(dir):
        raise ValueError(f"Directory '{dir}' does not exist!")

    # retrieve list of files to be loaded
    for name in sorted(os.listdir(dir)):
        fp = os.path.join(dir, name)

        # skip ignored files, hidden files and subfolders
        if name in ignore_files or name.startswith(".") or not os.path.isfile(fp):
            continue

        # skip files without a document loader
        if not is_supported(fp):
            print(f"Skipping unsupported file '{fp}'")
            continue

        files.append(fp)

    n = len(files)

    # load files
    with tqdm(total=n, desc=f"Loading {n} file(s) from directory '{dir}'", ncols=100) as pbar:
        for file in files:
            doc = load_document(file)
            loaded_files.extend(doc)
            pbar.update()

    if verbose:
        for file in files:
            print(f"Loaded file from '{file}'")

    return loaded_files
