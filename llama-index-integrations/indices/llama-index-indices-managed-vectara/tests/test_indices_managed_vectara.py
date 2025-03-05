from typing import List
from llama_index.core.schema import Document, Node, MediaResource
from llama_index.core.indices.managed.base import BaseManagedIndex
from llama_index.indices.managed.vectara import VectaraIndex
import pytest
import re

#
# For this test to run properly, please setup as follows:
# 1. Create a Vectara account: sign up at https://console.vectara.com/signup
# 2. Create two corpora with corpus keys "Llamaindex-testing-1" and "llamaindex-testing-2" in your Vectara account with the following filter attributes:
#   "Llamaindex-testing-1":
#   a. doc.test_num (text)
#   b. doc.test_score (integer)
#   c. doc.date (text)
#   d. doc.url (text)
#   "llamaindex-testing-2":
#   a. doc.author (text)
#   b. doc.title (text)
#   c. part.test_num (text)
#   d. part.test_score (integer)
#   e. part.date (text)
# 3. Create an API_KEY for these corpora with permissions for query and indexing
# 4. Setup environment variables:
#    VECTARA_API_KEY, VECTARA_CORPUS_KEY, and OPENAI_API_KEY
#    For VECTARA_CORPUS_KEY, separate the corpus keys for the corpora with a ',' for example: "Llamaindex-testing-1,llamaindex-testing-2".
#


def test_class():
    names_of_base_classes = [b.__name__ for b in VectaraIndex.__mro__]
    assert BaseManagedIndex.__name__ in names_of_base_classes


def get_docs() -> List[Document]:
    inputs = [
        {
            "id": "doc_1",
            "text": "This is test text for Vectara integration with LlamaIndex",
            "metadata": {"test_num": "1", "test_score": 10, "date": "2020-02-25"},
        },
        {
            "id": "doc_2",
            "text": "And now for something completely different",
            "metadata": {"test_num": "2", "test_score": 2, "date": "2015-10-13"},
        },
        {
            "id": "doc_3",
            "text": "when 900 years you will be, look as good you will not",
            "metadata": {"test_num": "3", "test_score": 20, "date": "2023-09-12"},
        },
        {
            "id": "doc_4",
            "text": "when 850 years you will be, look as good you will not",
            "metadata": {"test_num": "4", "test_score": 50, "date": "2022-01-01"},
        },
    ]
    docs: List[Document] = []
    for inp in inputs:
        doc = Document(
            id_=inp["id"],
            text_resource=MediaResource(text=inp["text"]),
            metadata=inp["metadata"],
        )
        docs.append(doc)
    return docs


def get_nodes() -> List[Node]:
    inputs = [
        {
            "text": "This is test text for Vectara integration with LlamaIndex",
            "metadata": {"test_num": "1", "test_score": 10, "date": "2020-02-25"},
        },
        {
            "text": "And now for something completely different",
            "metadata": {"test_num": "2", "test_score": 2, "date": "2015-10-13"},
        },
        {
            "text": "when 900 years you will be, look as good you will not",
            "metadata": {"test_num": "3", "test_score": 20, "date": "2023-09-12"},
        },
        {
            "text": "when 850 years you will be, look as good you will not",
            "metadata": {"test_num": "4", "test_score": 50, "date": "2022-01-01"},
        },
    ]

    nodes: List[Node] = []
    for inp in inputs:
        node = Node(
            text_resource=MediaResource(text=inp["text"]), metadata=inp["metadata"]
        )
        nodes.append(node)
    return nodes


# Normal vectara index fixture
@pytest.fixture()
def vectara1():
    docs = get_docs()
    try:
        vectara1 = VectaraIndex.from_documents(docs)
    except ValueError:
        pytest.skip("Missing Vectara credentials, skipping test")

    yield vectara1

    # Tear down code
    for id in vectara1.doc_ids:
        vectara1.delete_ref_doc(id)


# vectara index fixture, where we specify the base url
@pytest.fixture()
def vectara1_custom():
    docs = get_docs()
    try:
        vectara1_custom = VectaraIndex.from_documents(
            docs, vectara_base_url="https://api.vectara.io"
        )
    except ValueError:
        pytest.skip("Missing Vectara credentials, skipping test")

    yield vectara1_custom

    # Tear down code
    for id in vectara1_custom.doc_ids:
        vectara1_custom.delete_ref_doc(id)


def test_simple_retrieval(vectara1) -> None:
    docs = get_docs()
    qe = vectara1.as_retriever(similarity_top_k=1)
    res = qe.retrieve("Find me something different")
    assert len(res) == 1
    assert res[0].node.get_content() == docs[1].text
    assert res[0].node.node_id == docs[1].doc_id


def test_simple_retrieval_with_custom_base_url(vectara1_custom) -> None:
    docs = get_docs()
    qe = vectara1_custom.as_retriever(similarity_top_k=1)
    res = qe.retrieve("Find me something different")
    assert len(res) == 1
    assert res[0].node.get_content() == docs[1].text
    assert res[0].node.node_id == docs[1].doc_id


def test_mmr_retrieval(vectara1) -> None:
    docs = get_docs()

    # test with diversity bias = 0
    qe = vectara1.as_retriever(
        similarity_top_k=2,
        n_sentences_before=0,
        n_sentences_after=0,
        reranker="mmr",
        rerank_k=10,
        mmr_diversity_bias=0.0,
    )
    res = qe.retrieve("how will I look?")
    assert len(res) == 2
    assert res[0].node.get_content() == docs[2].text
    assert res[1].node.get_content() == docs[3].text

    # test with diversity bias = 1
    qe = vectara1.as_retriever(
        similarity_top_k=2,
        n_sentences_before=0,
        n_sentences_after=0,
        reranker="mmr",
        rerank_k=10,
        mmr_diversity_bias=1.0,
    )
    res = qe.retrieve("how will I look?")
    assert len(res) == 2
    assert res[0].node.get_content() == docs[2].text
    assert res[1].node.get_content() == docs[0].text


def test_retrieval_with_filter(vectara1) -> None:
    docs = get_docs()

    assert isinstance(vectara1, VectaraIndex)
    qe = vectara1.as_retriever(similarity_top_k=1, filter=["doc.test_num = '1'", ""])
    res = qe.retrieve("What does this test?")
    assert len(res) == 1
    assert res[0].node.get_content() == docs[0].text


def test_udf_retrieval(vectara1) -> None:
    docs = get_docs()

    # test with basic math expression
    qe = vectara1.as_retriever(
        similarity_top_k=2,
        n_sentences_before=0,
        n_sentences_after=0,
        reranker="userfn",
        udf_expression="get('$.score') + get('$.document_metadata.test_score')",
    )

    res = qe.retrieve("What will the future look like?")
    assert len(res) == 2
    assert res[0].node.get_content() == docs[3].text
    assert res[1].node.get_content() == docs[2].text

    # test with dates: Weight of score subtracted by number of years from current date
    qe = vectara1.as_retriever(
        similarity_top_k=2,
        n_sentences_before=0,
        n_sentences_after=0,
        reranker="userfn",
        udf_expression="max(0, 5 * get('$.score') - (to_unix_timestamp(now()) - to_unix_timestamp(datetime_parse(get('$.document_metadata.date'), 'yyyy-MM-dd'))) / 31536000)",
    )

    res = qe.retrieve("What will the future look like?")
    assert len(res) == 2
    assert res[0].node.get_content() == docs[2].text
    assert res[1].node.get_content() == docs[3].text


def test_chain_rerank_retrieval(vectara1) -> None:
    docs = get_docs()

    # Test basic chain
    qe = vectara1.as_retriever(
        similarity_top_k=2,
        n_sentences_before=0,
        n_sentences_after=0,
        reranker="chain",
        rerank_chain=[{"type": "slingshot"}, {"type": "mmr", "diversity_bias": 0.4}],
    )

    res = qe.retrieve("What's this all about?")
    assert len(res) == 2
    assert res[0].node.get_content() == docs[0].text
    assert res[1].node.get_content() == docs[2].text

    # Test chain with UDF and limit
    qe = vectara1.as_retriever(
        similarity_top_k=4,
        n_sentences_before=0,
        n_sentences_after=0,
        reranker="chain",
        rerank_chain=[
            {"type": "slingshot"},
            {"type": "mmr"},
            {
                "type": "userfn",
                "user_function": "5 * get('$.score') + get('$.document_metadata.test_score') / 2",
                "limit": 2,
            },
        ],
    )

    res = qe.retrieve("What's this all about?")
    assert len(res) == 2
    assert res[0].node.get_content() == docs[3].text
    assert res[1].node.get_content() == docs[2].text

    # Test chain with cutoff
    qe = vectara1.as_retriever(
        similarity_top_k=4,
        n_sentences_before=0,
        n_sentences_after=0,
        reranker="chain",
        rerank_chain=[
            {"type": "slingshot"},
            {"type": "mmr", "diversity_bias": 0.4, "cutoff": 0.75},
        ],
    )

    res = qe.retrieve("What's this all about?")
    assert len(res) == 1
    assert res[0].node.get_content() == docs[0].text

    # Second query with same retriever to ensure rerank chain configuration remains the same
    res = qe.retrieve("How will I look when I'm older?")
    assert qe._rerank_chain[0].get("type") == "customer_reranker"
    assert qe._rerank_chain[0].get("reranker_name") == "Rerank_Multilingual_v1"
    assert qe._rerank_chain[1].get("type") == "mmr"
    assert res[0].node.get_content() == docs[2].text


def test_custom_prompt(vectara1) -> None:
    qe = vectara1.as_query_engine(
        similarity_top_k=3,
        n_sentences_before=0,
        n_sentences_after=0,
        reranker="mmr",
        mmr_diversity_bias=0.2,
        summary_enabled=True,
        prompt_text='[\n  {"role": "system", "content": "You are an expert in summarizing the future of Vectara\'s inegration with LlamaIndex. Your summaries are insightful, concise, and highlight key innovations and changes."},\n  #foreach ($result in $vectaraQueryResults)\n    {"role": "user", "content": "What are the key points in result number $vectaraIdxWord[$foreach.index] about Vectara\'s LlamaIndex integration?"},\n    {"role": "assistant", "content": "In result number $vectaraIdxWord[$foreach.index], the key points are: ${result.getText()}"},\n  #end\n  {"role": "user", "content": "Can you generate a comprehensive summary on \'Vectara\'s LlamaIndex Integration\' incorporating all the key points discussed?"}\n]\n',
    )

    res = qe.query("How does Vectara's integration with llamaindex work?")
    assert "integration" in str(res).lower()
    assert "llamaindex" in str(res).lower()
    assert "vectara" in str(res).lower()


def test_update_doc(vectara1) -> None:
    docs = get_docs()

    vectara1.update_ref_doc(
        document=docs[1], corpus_key="Llamaindex-testing-1", metadata={"test_score": 14}
    )

    qe = vectara1.as_retriever(similarity_top_k=1)

    res = qe.retrieve("Find me something completely different.")
    assert len(res) == 1
    assert res[0].node.get_content() == docs[1].text
    assert res[0].node.metadata["document"]["test_score"] == 14


@pytest.fixture()
def vectara2():
    try:
        vectara2 = VectaraIndex()
    except ValueError:
        pytest.skip("Missing Vectara credentials, skipping test")

    file_path = "docs/docs/examples/data/paul_graham/paul_graham_essay.txt"
    id = vectara2.insert_file(
        file_path,
        metadata={"url": "https://www.paulgraham.com/worked.html"},
        corpus_key="llamaindex-testing-2",
    )

    yield vectara2

    # Tear down code
    vectara2.delete_ref_doc(id, corpus_key="llamaindex-testing-2")


def test_file_upload(vectara2) -> None:
    # test query with Vectara summarization (default)
    query_engine = vectara2.as_query_engine(similarity_top_k=3)
    res = query_engine.query("What software did Paul Graham write?")
    assert "paul graham" in str(res).lower() and "software" in str(res).lower()
    assert "fcs" in res.metadata
    assert res.metadata["fcs"] >= 0

    # test query with Vectara summarization (streaming)
    query_engine = vectara2.as_query_engine(similarity_top_k=3, streaming=True)
    res = query_engine.query("What software did Paul Graham write?")
    summary = str(res)

    assert "paul graham" in summary.lower() and "software" in summary.lower()
    assert res.metadata["fcs"] >= 0
    assert len(res.source_nodes) > 0

    # test query with VectorStoreQuery (using OpenAI for summarization)
    query_engine = vectara2.as_query_engine(similarity_top_k=3, summary_enabled=False)
    res = query_engine.query("What software did Paul Graham write?")
    assert "paul graham" in str(res).lower() and "software" in str(res).lower()

    # test query with Vectara summarization (default)
    query_engine = vectara2.as_query_engine(similarity_top_k=3)
    res = query_engine.query("How is Paul related to Reddit?")
    summary = res.response
    assert "paul" in summary.lower() and "reddit" in summary.lower()
    assert "https://www.paulgraham.com/worked.html" in str(res.source_nodes)


def test_knee_reranker(vectara2) -> None:
    query_engine = vectara2.as_query_engine(
        rerank_k=50,
        similarity_top_k=50,
        reranker="chain",
        rerank_chain=[
            {"type": "slingshot"},
            {"type": "userfn", "user_function": "knee()"},
        ],
    )

    # test query with knee reranker (should return less results than rerank_k)
    res = query_engine.query("How is Paul related to Reddit?")
    summary = res.response
    assert "paul" in summary.lower() and "reddit" in summary.lower()
    assert "https://www.paulgraham.com/worked.html" in str(res.source_nodes)
    assert len(res.source_nodes) > 0 and len(res.source_nodes) < 20


def test_citations(vectara2) -> None:
    # test markdown citations
    query_engine = vectara2.as_query_engine(
        similarity_top_k=10,
        summary_num_results=7,
        summary_prompt_name="vectara-summary-ext-24-05-med-omni",
        citations_style="markdown",
        citations_url_pattern="{doc.url}",
        citations_text_pattern="(source)",
    )
    res = query_engine.query("What colleges has Paul attended?")
    summary = res.response
    assert "(source)" in summary
    assert "https://www.paulgraham.com/worked.html" in summary

    # test numeric citations
    query_engine = vectara2.as_query_engine(
        similarity_top_k=10,
        summary_num_results=7,
        summary_prompt_name="mockingbird-1.0-2024-07-16",
        citations_style="numeric",
    )
    res = query_engine.query("What colleges has Paul attended?")
    summary = res.response
    assert re.search(r"\[\d+\]", summary)

    # test citations with url pattern only (no text pattern)
    query_engine = vectara2.as_query_engine(
        similarity_top_k=10,
        summary_num_results=7,
        summary_prompt_name="vectara-summary-ext-24-05-med-omni",
        citations_style="markdown",
        citations_url_pattern="{doc.url}",
    )
    res = query_engine.query("What colleges has Paul attended?")
    summary = res.response
    assert "https://www.paulgraham.com/worked.html" in summary
    assert re.search(r"\[\d+\]", summary)


def test_chat(vectara2) -> None:
    # Test chat initialization
    chat_engine = vectara2.as_chat_engine(
        reranker="chain",
        rerank_k=30,
        rerank_chain=[{"type": "slingshot"}, {"type": "mmr", "diversity_bias": 0.2}],
    )
    res = chat_engine.chat("What grad schools did Paul apply to?")
    summary = res.response

    assert all(s in summary.lower() for s in ["mit", "yale", "harvard"])
    assert res.metadata["fcs"] > 0
    chat_id = chat_engine.conv_id
    assert chat_id is not None

    # Test chat follow up
    res = chat_engine.chat("What did he learn at the graduate school he selected?")
    summary = res.response

    assert "paul" in summary.lower()
    assert "harvard" in summary.lower()
    assert res.metadata["fcs"] > 0
    assert chat_engine.conv_id == chat_id

    # Test chat follow up with streaming
    res = chat_engine.stream_chat(
        "How did attending graduate school help him in his career?"
    )
    summary = str(res)

    assert len(res.source_nodes) > 0
    assert chat_engine.conv_id == chat_id

    # Test chat initialization with streaming
    chat_engine = vectara2.as_chat_engine(
        reranker="chain",
        rerank_k=30,
        rerank_chain=[
            {"type": "slingshot", "cutoff": 0.25},
            {"type": "mmr", "diversity_bias": 0.2},
        ],
    )
    res = chat_engine.stream_chat("How did Paul feel when Yahoo bought his company?")
    summary = str(res)

    assert "yahoo" in summary.lower()
    assert "felt" in summary.lower()
    assert chat_engine._retriever._conv_id is not None
    assert chat_engine._retriever._conv_id != chat_id
    assert len(res.source_nodes) > 0


@pytest.fixture()
def vectara3():
    nodes = get_nodes()
    try:
        vectara3 = VectaraIndex()
        vectara3.add_nodes(
            nodes,
            document_id="doc_1",
            document_metadata={"author": "Vectara", "title": "LlamaIndex Integration"},
            corpus_key="llamaindex-testing-2",
        )
    except ValueError:
        pytest.skip("Missing Vectara credentials, skipping test")

    yield vectara3

    # Tear down code
    for id in vectara3.doc_ids:
        vectara3.delete_ref_doc(id, corpus_key="llamaindex-testing-2")


def test_simple_retrieval_with_nodes(vectara3) -> None:
    nodes = get_nodes()
    qe = vectara3.as_retriever(
        similarity_top_k=1, n_sentences_before=0, n_sentences_after=0
    )
    res = qe.retrieve("Find me something different")
    assert len(res) == 1
    assert res[0].node.metadata["document"]["author"] == "Vectara"
    assert res[0].node.metadata["document"]["title"] == "LlamaIndex Integration"
    assert res[0].node.get_content() == nodes[1].text_resource.text
    assert res[0].node.metadata["test_score"] == 2


def test_filter_with_nodes(vectara3) -> None:
    qe = vectara3.as_retriever(
        similarity_top_k=2,
        n_sentences_before=0,
        n_sentences_after=0,
        lambda_val=[0.2, 0.01],
        filter=["", "doc.author = 'Vectara' AND part.test_score > 10"],
    )

    res = qe.retrieve("How will I look when I'm older?")
    assert len(res) == 2
    assert "look as good you will not" in res[0].node.get_content()
    assert "look as good you will not" in res[1].node.get_content()
    assert res[0].node.get_content() != res[1].node.get_content()
