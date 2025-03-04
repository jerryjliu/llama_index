# LlamaIndex Readers Integration:

## ChatGPT Conversations Reader and Message Node Parser

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A custom reader and node parser for processing exported ChatGPT conversation JSON files using [LlamaIndex](https://github.com/jerryjliu/llama_index). This package allows you to load and parse your ChatGPT conversation data, enabling advanced querying and analysis using LlamaIndex's indexing and querying capabilities.

## **Table of Contents**

- [Installation](#installation)
- [Features](#features)
- [Usage](#usage)
  - [Loading Conversations](#loading-conversations)
  - [Parsing Messages](#parsing-messages)
  - [Building an Index](#building-an-index)
- [Examples](#examples)
- [Tests](#tests)
- [Contributing](#contributing)
- [License](#license)

## **Installation**

```bash
pip install llama-index-readers-chatgpt-conversations
```

## **Features**

- **ChatGPT Conversation Reader**: Load exported ChatGPT conversation JSON files into LlamaIndex `Document` objects.
- **Message Parser**: Parse conversation documents into structured nodes with speaker identities and metadata.
- **Markdown Support**: Process Markdown content within messages, including headers, code blocks, and inline code.
- **Node Relationships**: Maintain relationships between nodes for contextual understanding.

## **Usage**

### **1. Loading Conversations**

Use the `ChatGPTConversationsReader` to load your exported ChatGPT conversation JSON file.

```python
from llama_index.readers.chatgpt_conversation_json import (
    ChatGPTConversationsReader,
)

# Initialize the reader with the path to your conversations.json file
reader = ChatGPTConversationsReader(input_file="path/to/conversations.json")

# Load documents
documents = reader.load_data()
print(f"Loaded {len(documents)} documents.")
```

### **2. Parsing Messages**

Use the `ChatGPTMessageNodeParser` to parse the loaded documents into nodes.

```python
from llama_index.readers.chatgpt_conversation_json import (
    ChatGPTMessageNodeParser,
)

# Initialize the message parser
parser = ChatGPTMessageNodeParser()

# Parse documents into nodes
nodes = parser(documents)
print(f"Parsed {len(nodes)} nodes.")
```

### **3. Building an Index**

Leverage LlamaIndex to build an index over the parsed nodes for querying.

```python
from llama_index import VectorStoreIndex, StorageContext

# Create a storage context (e.g., using a default in-memory store)
storage_context = StorageContext.from_defaults()

# Build the index
index = VectorStoreIndex(nodes, storage_context=storage_context)

# (Optional) Persist the index for later use
storage_context.persist(persist_dir="./storage")
```

## **Examples**

Here's a full example combining all steps:

```python
from llama_index.readers.chatgpt_conversation_json import (
    ChatGPTConversationsReader,
    ChatGPTMessageNodeParser,
)
from llama_index.core import VectorStoreIndex, StorageContext

# Step 1: Load conversations
reader = ChatGPTConversationsReader(input_file="path/to/conversations.json")
documents = reader.load_data()

# Step 2: Parse messages into nodes
parser = ChatGPTMessageNodeParser()
nodes = parser(documents)

# Step 3: Build an index over the nodes
storage_context = StorageContext.from_defaults()
index = VectorStoreIndex(nodes, storage_context=storage_context)

# Step 4: Query the index
query_engine = index.as_query_engine()
response = query_engine.query("What did I ask about data export?")
print(response)
```

## **Tests**

To run the tests, navigate to the project root directory and execute:

```bash
pytest tests
```

## **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## **Additional Notes**

- **Dependencies**:

  - `llama_index_core`
  - `markdown-it-py`

- **Compatibility**: The package is compatible with Python 3.8 and above.

- **Publishing**: Update the `pyproject.toml` with the appropriate package name, version, and author information before publishing to PyPI.

---

## **Happy Coding!**

Feel free to reach out if you have any questions or need further assistance.
