"""Token splitter."""

import logging
from typing import Callable, List, Optional

from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.callbacks.schema import CBEventType, EventPayload
from llama_index.core.constants import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE
from llama_index.core.node_parser.interface import MetadataAwareTextSplitter
from llama_index.core.node_parser.node_utils import default_id_func
from llama_index.core.node_parser.text.utils import split_by_char, split_by_sep
from llama_index.core.schema import Document
from llama_index.core.utils import get_tokenizer

_logger = logging.getLogger(__name__)

# NOTE: this is the number of tokens we reserve for metadata formatting
DEFAULT_METADATA_FORMAT_LEN = 2


class TokenTextSplitter(MetadataAwareTextSplitter):
    """Implementation of splitting text that looks at word tokens."""

    chunk_size: int = Field(
        default=DEFAULT_CHUNK_SIZE,
        description="The token chunk size for each chunk.",
        gt=0,
    )
    chunk_overlap: int = Field(
        default=DEFAULT_CHUNK_OVERLAP,
        description="The token overlap of each chunk when splitting.",
        ge=0,
    )
    separator: str = Field(
        default=" ", description="Default separator for splitting into words"
    )
    backup_separators: List = Field(
        default_factory=list, description="Additional separators for splitting."
    )

    keep_whitespaces: bool = Field(
        default=False,
        description="Whether to keep leading/trailing whitespaces in the chunk.",
    )

    _tokenizer: Callable = PrivateAttr()
    _split_fns: List[Callable] = PrivateAttr()

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        tokenizer: Optional[Callable] = None,
        callback_manager: Optional[CallbackManager] = None,
        separator: str = " ",
        backup_separators: Optional[List[str]] = ["\n"],
        keep_whitespaces: bool = False,
        include_metadata: bool = True,
        include_prev_next_rel: bool = True,
        id_func: Optional[Callable[[int, Document], str]] = None,
    ):
        """Initialize with parameters."""
        if chunk_overlap > chunk_size:
            raise ValueError(
                f"Got a larger chunk overlap ({chunk_overlap}) than chunk size "
                f"({chunk_size}), should be smaller."
            )
        callback_manager = callback_manager or CallbackManager([])
        id_func = id_func or default_id_func
        super().__init__(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separator=separator,
            backup_separators=backup_separators,
            keep_whitespaces=keep_whitespaces,
            callback_manager=callback_manager,
            include_metadata=include_metadata,
            include_prev_next_rel=include_prev_next_rel,
            id_func=id_func,
        )
        self._tokenizer = tokenizer or get_tokenizer()
        all_seps = [separator] + (backup_separators or [])
        self._split_fns = [split_by_sep(sep) for sep in all_seps] + [split_by_char()]

    @classmethod
    def from_defaults(
        cls,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        separator: str = " ",
        backup_separators: Optional[List[str]] = ["\n"],
        callback_manager: Optional[CallbackManager] = None,
        keep_whitespaces: bool = False,
        include_metadata: bool = True,
        include_prev_next_rel: bool = True,
        id_func: Optional[Callable[[int, Document], str]] = None,
    ) -> "TokenTextSplitter":
        """Initialize with default parameters."""
        callback_manager = callback_manager or CallbackManager([])
        return cls(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separator=separator,
            backup_separators=backup_separators,
            keep_whitespaces=keep_whitespaces,
            callback_manager=callback_manager,
            include_metadata=include_metadata,
            include_prev_next_rel=include_prev_next_rel,
            id_func=id_func,
        )

    @classmethod
    def class_name(cls) -> str:
        return "TokenTextSplitter"

    def split_text_metadata_aware(self, text: str, metadata_str: str) -> List[str]:
        """Split text into chunks, reserving space required for metadata str."""
        metadata_len = len(self._tokenizer(metadata_str)) + DEFAULT_METADATA_FORMAT_LEN
        effective_chunk_size = self.chunk_size - metadata_len
        if effective_chunk_size <= 0:
            raise ValueError(
                f"Metadata length ({metadata_len}) is longer than chunk size "
                f"({self.chunk_size}). Consider increasing the chunk size or "
                "decreasing the size of your metadata to avoid this."
            )
        elif effective_chunk_size < 50:
            print(
                f"Metadata length ({metadata_len}) is close to chunk size "
                f"({self.chunk_size}). Resulting chunks are less than 50 tokens. "
                "Consider increasing the chunk size or decreasing the size of "
                "your metadata to avoid this.",
                flush=True,
            )

        return self._split_text(text, chunk_size=effective_chunk_size)

    def split_text(self, text: str) -> List[str]:
        """Split text into chunks."""
        return self._split_text(text, chunk_size=self.chunk_size)

    def _split_text(self, text: str, chunk_size: int) -> List[str]:
        """Split text into chunks up to chunk_size."""
        if text == "":
            return [text]

        with self.callback_manager.event(
            CBEventType.CHUNKING, payload={EventPayload.CHUNKS: [text]}
        ) as event:
            splits = self._split(text, chunk_size)
            chunks = self._merge(splits, chunk_size)

            event.on_end(
                payload={EventPayload.CHUNKS: chunks},
            )

        return chunks

    def _split(self, text: str, chunk_size: int) -> List[str]:
        """Break text into splits that are smaller than chunk size.

        The order of splitting is:
        1. split by separator
        2. split by backup separators (if any)
        3. split by characters

        NOTE: the splits contain the separators.
        """
        if len(self._tokenizer(text)) <= chunk_size:
            return [text]

        for split_fn in self._split_fns:
            splits = split_fn(text)
            if len(splits) > 1:
                break

        new_splits = []
        for split in splits:
            split_len = len(self._tokenizer(split))
            if split_len <= chunk_size:
                new_splits.append(split)
            else:
                # recursively split
                new_splits.extend(self._split(split, chunk_size=chunk_size))
        return new_splits

    def _merge(self, splits: List[str], chunk_size: int) -> List[str]:
        """Merge splits into chunks.

        The high-level idea is to keep adding splits to a chunk until we
        exceed the chunk size, then we start a new chunk with overlap.

        When we start a new chunk, we pop off the first element of the previous
        chunk until the total length is less than the chunk size.
        """
        chunks: List[str] = []

        cur_chunk: List[str] = []
        cur_len = 0
        for split in splits:
            split_len = len(self._tokenizer(split))
            if split_len > chunk_size:
                _logger.warning(
                    f"Got a split of size {split_len}, ",
                    f"larger than chunk size {chunk_size}.",
                )

            # if we exceed the chunk size after adding the new split, then
            # we need to end the current chunk and start a new one
            if cur_len + split_len > chunk_size:
                # end the previous chunk
                chunk = (
                    "".join(cur_chunk)
                    if self.keep_whitespaces
                    else "".join(cur_chunk).strip()
                )
                if chunk:
                    chunks.append(chunk)

                # start a new chunk with overlap
                # keep popping off the first element of the previous chunk until:
                #   1. the current chunk length is less than chunk overlap
                #   2. the total length is less than chunk size
                while cur_len > self.chunk_overlap or cur_len + split_len > chunk_size:
                    # pop off the first element
                    first_chunk = cur_chunk.pop(0)
                    cur_len -= len(self._tokenizer(first_chunk))

            cur_chunk.append(split)
            cur_len += split_len

        # handle the last chunk
        chunk = (
            "".join(cur_chunk) if self.keep_whitespaces else "".join(cur_chunk).strip()
        )
        if chunk:
            chunks.append(chunk)

        return chunks
