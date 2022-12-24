"""Init file of GPT Index."""

from pathlib import Path

with open(Path(__file__).absolute().parents[0] / "VERSION") as _f:
    __version__ = _f.read().strip()


# embeddings
from gpt_index.embeddings.langchain import LangchainEmbedding
from gpt_index.embeddings.openai import OpenAIEmbedding
from gpt_index.indices.keyword_table.base import GPTKeywordTableIndex
from gpt_index.indices.keyword_table.rake_base import GPTRAKEKeywordTableIndex
from gpt_index.indices.keyword_table.simple_base import GPTSimpleKeywordTableIndex
from gpt_index.indices.list.base import GPTListIndex

# indices
from gpt_index.indices.keyword_table import (
    GPTKeywordTableIndex,
    GPTRAKEKeywordTableIndex,
    GPTSimpleKeywordTableIndex,
)
from gpt_index.indices.list import GPTListIndex
from gpt_index.indices.tree import GPTTreeIndex
from gpt_index.indices.vector_store import GPTFaissIndex

# langchain helper
from gpt_index.langchain_helpers.chain_wrapper import LLMPredictor

# prompts
from gpt_index.prompts.base import Prompt
from gpt_index.prompts.prompts import (
    KeywordExtractPrompt,
    QueryKeywordExtractPrompt,
    QuestionAnswerPrompt,
    RefinePrompt,
    SummaryPrompt,
    TreeInsertPrompt,
    TreeSelectMultiplePrompt,
    TreeSelectPrompt,
)

# readers
from gpt_index.readers.file import SimpleDirectoryReader
from gpt_index.readers.google.gdocs import GoogleDocsReader
from gpt_index.readers.mongo import SimpleMongoReader
from gpt_index.readers.notion import NotionPageReader

# allow importing Document at the top-level
from gpt_index.readers.schema.base import Document
from gpt_index.readers.slack import SlackReader
from gpt_index.readers.weaviate import WeaviateReader
from gpt_index.readers.wikipedia import WikipediaReader

# token predictor
from gpt_index.token_predictor.mock_chain_wrapper import MockLLMPredictor

__all__ = [
    "GPTKeywordTableIndex",
    "GPTSimpleKeywordTableIndex",
    "GPTRAKEKeywordTableIndex",
    "GPTListIndex",
    "GPTTreeIndex",
    "GPTFaissIndex",
    "Prompt",
    "LangchainEmbedding",
    "OpenAIEmbedding",
    "SummaryPrompt",
    "TreeInsertPrompt",
    "TreeSelectPrompt",
    "TreeSelectMultiplePrompt",
    "RefinePrompt",
    "QuestionAnswerPrompt",
    "KeywordExtractPrompt",
    "QueryKeywordExtractPrompt",
    "WikipediaReader",
    "Document",
    "SimpleDirectoryReader",
    "SimpleMongoReader",
    "NotionPageReader",
    "GoogleDocsReader",
    "SlackReader",
    "WeaviateReader",
    "LLMPredictor",
    "MockLLMPredictor",
]
