# LlamaIndex Managed Integration: Vectara

The Vectara Index provides a simple implementation to Vectara's end-to-end RAG pipeline,
including data ingestion, document retrieval, reranking results, summary generation, and hallucination evaluation.

Please note that this documentation applies to versions >= 0.4.0 and will not be the same as for earlier versions of Vectara `ManagedIndex`.

## 📌 Setup

First, make sure you have the latest LlamaIndex version installed.

```
pip install -U llama-index
```

Next, install the Vectara Index:

```
pip install -U llama-index-indices-managed-vectara
```

Finally, set up your Vectara corpus. If you don't have a Vectara account, you can [sign up](https://vectara.com/integrations/llamaindex) and follow our [Quick Start](https://docs.vectara.com/docs/quickstart) guide to create a corpus and an API key (make sure the api_key has both indexing and query permissions, or use your personal API key).

Once you have your API key, export it as an environment variable:

```python
import os

os.environ["VECTARA_API_KEY"] = "<YOUR_VECTARA_API_KEY>"
os.environ["VECTARA_CORPUS_KEY"] = "<YOUR_VECTARA_CORPUS_KEY>"
```

## 🚀 Usage

### 1. Index Documents

Create an index and add some sample documents:

```python
from llama_index.indices.managed.vectara import VectaraIndex
from llama_index.core.schema import Document, MediaResource

docs = [
    Document(
        id_="doc1",
        text_resource=MediaResource(
            text="""
            This is test text for Vectara integration with LlamaIndex.
            Users should love their experience with this integration
            """,
        ),
    ),
    Document(
        id_="doc2",
        text_resource=MediaResource(
            text="""
            The Vectara index integration with LlamaIndex implements Vectara's RAG pipeline.
            It can be used both as a retriever and query engine.
            """,
        ),
    ),
]
index = VectaraIndex.from_documents(docs)
```

Make sure to always specify a unique `id_` for every document you add to your index.
If you don't specify this parameter, a random id will be generated and the document will be separately added to your corpus every time you run your code.

You can now use this index to retrieve documents.

### 2. Retrieve Documents

Retrieve the top 2 most relevant document for a query:

```python
# Retrieves the top search result
retriever = index.as_retriever(similarity_top_k=2)

results = retriever.retrieve("How will users feel about this new tool?")
print(results[0])
```

### 3. Use as a Query Engine

Generate a summary of retrieved results:

```python
query_engine = index.as_query_engine()

results = query_engine.query(
    "Which company has partnered with Vectara to implement their RAG pipeline as an index?"
)
print(f"Generated summary: {results.response}\n")
print("Top sources:")
for node in results.source_nodes[:2]:
    print(node)
```

## 📂 Understanding `source_nodes` structure

Each node object in `source_nodes` contains a `NodeWithScore` object with:

- `text`: The matched text snippet.
- `id_`: The unique identifier of the document.
- `metadata`: A dictionary containing:
  - Key-value pairs from the matched part of the document.
  - A `document` key that stores all document-level metadata.
- `score`: The relevance score of the match.

Example Output:

```
NodeWithScore(
    node=Node(
        text_resource=MediaResource(
            text="This is a test text for Vectara integration with LlamaIndex."
        ),
        id_="doc1",
        metadata={
            "category": "AI",
            "page": 23,
            "document": {
                "url": "https://www.vectara.com/developers/build/integrations/llamaindex",
                "title": "LlamaIndex + Vectara Integration",
                "author": "Ofer Mendelevitch",
                "date": "2025-03-01"
            }
        }
    ),
    score=0.89
)
```

If you want to see the full features and capabilities of `VectaraIndex`, check out this Jupyter [notebook](https://github.com/vectara/example-notebooks/blob/main/notebooks/using-vectara-with-llamaindex.ipynb).
