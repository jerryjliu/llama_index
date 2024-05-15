import re
import string
from typing import Any, Callable, List, Optional, Sequence, TypedDict

import nltk
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


from llama_index.core.node_parser.interface import TextSplitter
from llama_index.core.node_parser.interface import NodeParser


import numpy as np
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.bridge.pydantic import Field
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.node_parser import NodeParser
from llama_index.core.node_parser.interface import NodeParser
from llama_index.core.node_parser.node_utils import (
    build_nodes_from_splits,
    default_id_func,
)
from llama_index.core.node_parser.text.utils import split_by_sentence_tokenizer
from llama_index.core.schema import BaseNode, Document
from llama_index.core.utils import get_tqdm_iterable

nltk.download("punkt")
nltk.download("stopwords")

DEFAULT_OG_TEXT_METADATA_KEY = "original_text"

#TODO add more models
LANGUAGES : list[str] = ["english"]
LANGUAGE_MODELS : dict[str, list[str]] = {"english" : ["en_core_web_md"]}



class LanguageConfig:

    def __init__(self, language : str = "english", spacy_model : str = "en_core_web_md", model_validation : bool = True):

        if language not in LANGUAGES:
            raise ValueError(f"{language} language is not supported yet! Avaiable languages: {LANGUAGES}")
        
        if spacy_model not in LANGUAGE_MODELS[language] and model_validation:
            raise ValueError(f"{spacy_model} model is not avaiable")
        
        self.language = language
        self.nlp = spacy.load(spacy_model)
        self.stopwords = set(stopwords.words(language))
            


class SemanticDoubleMergingSplitterNodeParser(NodeParser):

    """Semantic double merging text splitter.

    Splits a document into Nodes, with each node being a group of semantically related sentences.

    Args:
        # buffer_size (int): number of sentences to group together when evaluating semantic similarity
        # embed_model: (BaseEmbedding): embedding model to use
        sentence_splitter (Optional[Callable]): splits text into sentences
        include_metadata (bool): whether to include metadata in nodes
        include_prev_next_rel (bool): whether to include prev/next relationships
    """

    language_config: LanguageConfig = Field(
        default=LanguageConfig(),
        description="Config that selects language and spacy model for chunking"
    )

    initial_threshold: float = Field(
        default=0.6,
        description=(
            "The value of semantic similarity that must be exceeded between two"
            "sentences to create a new chunk.  The bigger this "
            "value is, the more nodes will be generated. Range is from 0 to 1."
        ),
    )

    appending_treshold: float = Field(
        default=0.8,
        description=(
            "The value of semantic similarity that must be exceeded between a "
            "chunk and new sentence to add this sentence to existing chunk.  The bigger this "
            "value is, the more nodes will be generated. Range is from 0 to 1."
        ),
    )

    merging_threshold: float = Field(
        default=0.8,
        description=(
            "The value of semantic similarity that must be exceeded between two chunks "
            "to form a bigger chunk.  The bigger this value is,"
            "the more nodes will be generated. Range is from 0 to 1."
        ),
    )

    max_chunk_size: int = Field(
        default=1000,
        description="Maximum size of returned chunk (number of characters)"
    )

    sentence_splitter: Callable[[str], List[str]] = Field(
        default_factory=split_by_sentence_tokenizer,
        description="The text splitter to use when splitting documents.",
        exclude=True,
    )


    @classmethod
    def class_name(cls) -> str:
        return "SemanticDoubleMergingSplitterNodeParser"
    
    
    @classmethod
    def from_defaults(
        cls,
        language_config: Optional[LanguageConfig] = LanguageConfig(),
        initial_threshold: Optional[float] = 0.6,
        appending_threshold: Optional[float] = 0.8,
        merging_threshold: Optional[float] = 0.8,
        max_chunk_size: Optional[int] = 1000,
        sentence_splitter: Optional[Callable[[str], List[str]]] = None,
        original_text_metadata_key: str = DEFAULT_OG_TEXT_METADATA_KEY,
        include_metadata: bool = True,
        include_prev_next_rel: bool = True,
        callback_manager: Optional[CallbackManager] = None,
        id_func: Optional[Callable[[int, Document], str]] = None,
    ) -> "SemanticDoubleMergingSplitterNodeParser":
        callback_manager = callback_manager or CallbackManager([])

        sentence_splitter = sentence_splitter or split_by_sentence_tokenizer()

        id_func = id_func or default_id_func

        return cls(
            language_config=language_config,
            initial_threshold=initial_threshold,
            appending_threshold=appending_threshold,
            merging_threshold=merging_threshold,
            max_chunk_size=max_chunk_size,
            sentence_splitter=sentence_splitter,
            original_text_metadata_key=original_text_metadata_key,
            include_metadata=include_metadata,
            include_prev_next_rel=include_prev_next_rel,
            callback_manager=callback_manager,
            id_func=id_func,
        )


    def _parse_nodes(
        self,
        nodes: Sequence[BaseNode],
        show_progress: bool = False,
        **kwargs: Any,
    ) -> List[BaseNode]:
        #TODO
        pass

