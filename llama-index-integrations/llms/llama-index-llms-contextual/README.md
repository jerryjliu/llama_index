# Contextual LLM Integration for LlamaIndex

This package provides a Contextual LLM integration for LlamaIndex.

## Installation

```bash
pip install llama-index-llms-contextual
```

## Usage

```python
from llama_index.llms.contextual import Contextual

llm = Contextual(model="contextual-clm", api_key="your_api_key")

response = llm.complete("Explain the importance of Grounded Language Models.")
```
