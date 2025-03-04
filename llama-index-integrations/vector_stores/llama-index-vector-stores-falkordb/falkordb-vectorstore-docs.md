# FalkorDB Vector Store Integration for LlamaIndex

## Overview

The FalkorDB Vector Store integration for LlamaIndex allows you to use FalkorDB as a vector store backend for your LlamaIndex applications. This integration supports efficient storage and retrieval of vector embeddings, enabling fast similarity searches and other vector operations.

## Installation

To use the FalktorDB vector store with LlamaIndex, you need to install the necessary package:

```bash
pip install llama-index-vector-stores-falkordb
```

## Usage

### Initializing the FalkorDB Vector Store

To use the FalkorDB vector store, you first need to initialize it with your FalkorDB connection details:

```python
from llama_index.vector_stores.falkordb import FalkorDBVectorStore

vector_store = FalkorDBVectorStore(
    url="falkor://localhost:7687",
    database="your_database_name",
    index_name="your_index_name",
    node_label="YourNodeLabel",
    embedding_node_property="embedding",
    text_node_property="text",
    distance_strategy="cosine",
    embedding_dimension=1536,
)
```

### Adding Documents

You can add documents to the vector store using the `add` method:

```python
from llama_index.core.schema import Document

documents = [
    Document("This is the first document."),
    Document("This is the second document."),
]

vector_store.add(documents)
```

### Querying the Vector Store

To perform a similarity search, use the `query` method:

```python
from llama_index.core.vector_stores.types import VectorStoreQuery

query_embedding = [0.1, 0.2, 0.3, ...]  # Your query embedding
query = VectorStoreQuery(query_embedding=query_embedding, similarity_top_k=5)
results = vector_store.query(query)

for node, score in zip(results.nodes, results.similarities):
    print(f"Text: {node.text}, Score: {score}")
```

## Advanced Features

### Creating a New Index

If you need to create a new vector index, you can use the `create_new_index` method:

```python
vector_store.create_new_index()
```

This method will create a new vector index in FalkorDB based on the parameters you provided when initializing the `FalkorDBVectorStore`.

### Retrieving Existing Index

To check if an index already exists and retrieve its information:

```python
exists = vector_store.retrieve_existing_index()
if exists:
    print("Index exists with the following properties:")
    print(f"Node Label: {vector_store.node_label}")
    print(f"Embedding Property: {vector_store.embedding_node_property}")
    print(f"Embedding Dimension: {vector_store.embedding_dimension}")
    print(f"Distance Strategy: {vector_store.distance_strategy}")
else:
    print("Index does not exist")
```

### Deleting Documents

To delete documents from the vector store:

```python
ref_doc_id = "your_document_id"
vector_store.delete(ref_doc_id)
```

### Using Metadata Filters

You can use metadata filters when querying the vector store:

```python
from llama_index.core.vector_stores.types import (
    MetadataFilters,
    MetadataFilter,
)

filters = MetadataFilters(
    filters=[
        MetadataFilter(
            key="category", value="science", operator=FilterOperator.EQ
        )
    ]
)

query = VectorStoreQuery(
    query_embedding=query_embedding, similarity_top_k=5, filters=filters
)

results = vector_store.query(query)
```

## Best Practices

1. **Connection Management**: Ensure that you properly manage your FalkorDB connections, especially in production environments.
2. **Index Naming**: Use descriptive names for your indexes to easily identify them in your FalkorDB instance.
3. **Error Handling**: Implement proper error handling to manage potential issues with connections or queries.
4. **Performance Tuning**: Adjust the `embedding_dimension` and `distance_strategy` parameters based on your specific use case and performance requirements.

## Troubleshooting

If you encounter issues:

1. Check your FalkorDB connection details (URL, database name).
2. Ensure that the FalkorDB server is running and accessible.
3. Verify that the index exists and has the correct properties.
4. Check the FalkorDB logs for any error messages.

For more information on FalkorDB and its capabilities, refer to the [official FalkorDB documentation](https://falkordb.com/docs/).
