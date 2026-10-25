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
        Document file path. The extension must be one of the keys of
        `DOCUMENT_LOADERS`, in lowercase.

    Returns
    -------
    list of Documents
        A list of Document objects (for example, one per PDF page or one per
        CSV row).

    Raises
    ------
    ValueError
        If the file's extension is not supported.
    KeyError
        If the extension is supported but not lowercase (e.g. ".PDF").
    """

    # retrieve file's extension type
    ext = f".{fp.rsplit('.')[-1]}"

    # retrieve doc's loader based on its extension type
    if ext.lower() in DOCUMENT_LOADERS:
        cls, args = DOCUMENT_LOADERS[ext]
        loader = cls(fp, **args)
    else:
        raise ValueError(f"Document type '(.{ext})' is not supported!")
    
    return loader.load()


def load_documents(dir, ignore_files=[], verbose=False):
    """
    Load every file in a directory.

    Only the top level of `dir` is read; subfolders are not scanned.

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
        A list of Document objects.

    Raises
    ------
    ValueError
        If `dir` does not exist, or if it contains an entry that cannot be
        loaded: an unsupported file, a hidden file such as ".DS_Store", a file
        without an extension, or a subfolder.
    KeyError
        If a file's extension is supported but not lowercase (e.g. ".PDF").
    """

    files = list()
    loaded_files = list()

    # checks whether directory specified exists or not
    if os.path.exists(dir):
        # retrieve list of files to be loaded
        files = [os.path.join(dir, file) for file in os.listdir(dir) if file not in ignore_files]
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
    
    else:
        raise ValueError(f"Directory '{dir}' does not exist!")
    
    return loaded_files
