"""
Data Connectors for LlamaIndex.

This module contains the data connectors for LlamaIndex. Each connector inherits
from a `BaseReader` class, connects to a data source, and loads Document objects
from that data source.

You may also choose to construct Document objects manually, for instance
in our `Insert How-To Guide <../how_to/insert.html>`_. See below for the API
definition of a Document - the bare minimum is a `text` property.

"""

from llama_index.core.readers.base import ReaderConfig
from llama_index.core.readers.download import download_loader

# readers
from llama_index.core.readers.file.base import (
    SimpleDirectoryReader,
    FileSystemReaderMixin,
    DirectoryReaderArgs,
    LoadFileArgs,
    BaseDirectoryReaderArgs,
)
from llama_index.core.readers.string_iterable import StringIterableReader
from llama_index.core.schema import Document

__all__ = [
    "DirectoryReaderArgs",
    "LoadFileArgs",
    "BaseDirectoryReaderArgs",
    "SimpleDirectoryReader",
    "FileSystemReaderMixin",
    "ReaderConfig",
    "Document",
    "StringIterableReader",
    "download_loader",
]
