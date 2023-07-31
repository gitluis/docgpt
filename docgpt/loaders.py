#!/usr/bin/env python3

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
        Document file path.

    Returns
    -------
    list of Documents
        A list of Document objects.
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
    Load multiple documents.

    Parameters
    ----------
    dir : str
        File path to directory containing documents.
    ignore_files : list or str
        A list of paths of files that will not be loaded.
    verbose : bool
        Used for verbose mode. True is ON, False is OFF.

    Returns
    -------
    list of Documents
        A list of Document objects.
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
