"""Test node utils"""

import pytest
from typing import List
from gpt_index.indices.node_utils import get_nodes_from_document
from gpt_index.langchain_helpers.text_splitter import TokenTextSplitter
from gpt_index.readers.schema.base import Document


@pytest.fixture
def text_splitter() -> TokenTextSplitter:
    return TokenTextSplitter(chunk_size=20, chunk_overlap=0)


@pytest.fixture
def documents() -> List[Document]:
    """Get documents."""
    # NOTE: one document for now
    doc_text = (
        "Hello world.\n"
        "This is a test.\n"
        "This is another test.\n"
        "This is a test v2."
    )
    return [Document(doc_text, doc_id='test_doc_id', extra_info={'test_key': 'test_val'})]


def test_get_nodes_from_document(documents: List[Document], text_splitter: TokenTextSplitter):
    nodes = get_nodes_from_document(
        documents[0],
        text_splitter,
        start_idx=0,
        include_extra_info=False,
    )
    assert len(nodes) == 2
    actual_chunk_sizes = [len(text_splitter.tokenizer(node.get_text())) for node in nodes]
    assert all(chunk_size <= text_splitter._chunk_size for chunk_size in actual_chunk_sizes)


def test_get_nodes_from_document_with_extra_info(documents: List[Document], text_splitter: TokenTextSplitter):
    nodes = get_nodes_from_document(
        documents[0],
        text_splitter,
        start_idx=0,
        include_extra_info=True,
    )
    assert len(nodes) == 3
    actual_chunk_sizes = [len(text_splitter.tokenizer(node.get_text())) for node in nodes]
    assert all(chunk_size <= text_splitter._chunk_size for chunk_size in actual_chunk_sizes)

