"""
Callback handler for storing trace data in-memory in OpenInference format.
"""

import hashlib
import importlib
import os
import sys
from collections import OrderedDict
from dataclasses import dataclass, field, fields
from datetime import datetime
from types import ModuleType
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    TypeAlias,
    TypeVar,
)

from llama_index.callbacks.base import BaseCallbackHandler
from llama_index.callbacks.schema import CBEventType, EventPayload

if TYPE_CHECKING:
    from pandas import DataFrame


OPENINFERENCE_COLUMN_NAME = "openinference_column_name"
OpenInferenceDataType = TypeVar("OpenInferenceDataType", bound="OpenInferenceData")
Embedding: TypeAlias = List[float]


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _generate_random_id() -> str:
    return _hash_bytes(os.urandom(32))


@dataclass
class OpenInferenceData:
    id: str = field(
        default_factory=_generate_random_id,
        metadata={OPENINFERENCE_COLUMN_NAME: ":id.str:"},
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            field.metadata.get(OPENINFERENCE_COLUMN_NAME, field.name): getattr(
                self, field.name
            )
            for field in fields(self)
        }


@dataclass
class OpenInferenceQueryData(OpenInferenceData):
    timestamp: Optional[str] = field(
        default=None, metadata={OPENINFERENCE_COLUMN_NAME: ":timestamp.iso_8601:"}
    )
    query_text: Optional[str] = field(
        default=None, metadata={OPENINFERENCE_COLUMN_NAME: ":feature.text:query"}
    )
    query_embedding: Optional[Embedding] = field(
        default=None, metadata={OPENINFERENCE_COLUMN_NAME: ":feature.vector:query"}
    )
    response_text: Optional[str] = field(
        default=None, metadata={OPENINFERENCE_COLUMN_NAME: ":prediction.text:response"}
    )
    document_ids: List[str] = field(
        default_factory=list,
        metadata={OPENINFERENCE_COLUMN_NAME: "TODO: document_ids column"},
    )
    document_hashes: List[str] = field(
        default_factory=list,
        metadata={OPENINFERENCE_COLUMN_NAME: "TODO: document_hashes column"},
    )
    scores: List[float] = field(
        default_factory=list,
        metadata={OPENINFERENCE_COLUMN_NAME: "TODO: scores column"},
    )


@dataclass
class OpenInferenceDocumentData(OpenInferenceData):
    document_text: Optional[str] = field(
        default=None, metadata={OPENINFERENCE_COLUMN_NAME: ":feature.text:document"}
    )
    document_embedding: Optional[Embedding] = field(
        default=None, metadata={OPENINFERENCE_COLUMN_NAME: ":feature.vector:document"}
    )
    document_hash: Optional[str] = field(
        default=None, metadata={OPENINFERENCE_COLUMN_NAME: "TODO: document_hash column"}
    )


@dataclass
class TraceData:
    trace_id: Optional[str] = field(default_factory=_generate_random_id)
    start_timestamp: Optional[datetime] = None
    end_timestamp: Optional[datetime] = None
    query_data: OpenInferenceQueryData = field(default_factory=OpenInferenceQueryData)
    document_datas: List[OpenInferenceDocumentData] = field(default_factory=list)


def _import_package(package_name: str) -> ModuleType:
    try:
        package = importlib.import_module(package_name)
    except ImportError:
        raise ImportError(f"The {package_name} package must be installed.")
    return package


class OpenInferenceDataBuffer(Generic[OpenInferenceDataType]):
    def __init__(self, max_size_in_bytes: Optional[int] = None) -> None:
        self._id_to_data_map: OrderedDict[str, OpenInferenceData] = OrderedDict()
        self._max_size_in_bytes = max_size_in_bytes
        self._size_in_bytes = 0

    def __len__(self) -> int:
        return len(self._id_to_data_map)

    def add(self, data: OpenInferenceData) -> None:
        if data.id in self._id_to_data_map:
            self._size_in_bytes -= sys.getsizeof(self._id_to_data_map.pop(data.id))
        self._id_to_data_map[data.id] = data
        self._size_in_bytes += sys.getsizeof(data)

        if self._max_size_in_bytes is not None:
            while self._size_in_bytes > self._max_size_in_bytes:
                _, data = self._id_to_data_map.popitem(last=False)
                self._size_in_bytes -= sys.getsizeof(data)

    def clear(self) -> None:
        self._id_to_data_map.clear()
        self._size_in_bytes = 0

    @property
    def dataframe(self) -> "DataFrame":
        pandas = _import_package("pandas")
        return pandas.DataFrame(
            [data.to_dict() for data in self._id_to_data_map.values()]
        )

    @property
    def size_in_bytes(self) -> int:
        return self._size_in_bytes


class OpenInferenceCallbackHandler(BaseCallbackHandler):
    def __init__(
        self,
        max_size_in_bytes: Optional[int] = None,
        callback: Optional[
            Callable[[OpenInferenceDataBuffer[OpenInferenceQueryData]], None]
        ] = None,
    ) -> None:
        super().__init__(event_starts_to_ignore=[], event_ends_to_ignore=[])
        self._callback = callback
        self._trace_data = TraceData()
        self._query_data_buffer = OpenInferenceDataBuffer[OpenInferenceQueryData](
            max_size_in_bytes=max_size_in_bytes
        )
        self._document_data_buffer = OpenInferenceDataBuffer[OpenInferenceDocumentData](
            max_size_in_bytes=max_size_in_bytes,
        )

    def start_trace(self, trace_id: Optional[str] = None) -> None:
        if trace_id == "query":
            trace_start_time = datetime.now()
            self._trace_data = TraceData(start_timestamp=trace_start_time)
            self._trace_data.query_data.timestamp = trace_start_time.isoformat()
            self._trace_data.query_data.id = _generate_random_id()

    def end_trace(
        self,
        trace_id: Optional[str] = None,
        trace_map: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        if trace_id == "query":
            self._trace_data.end_timestamp = datetime.now()
            self._query_data_buffer.add(self._trace_data.query_data)
            for document_data in self._trace_data.document_datas:
                self._document_data_buffer.add(document_data)
            if self._callback is not None:
                self._callback(self._query_data_buffer)

    def on_event_start(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> str:
        if payload is not None:
            if event_type is CBEventType.QUERY:
                query_text = payload[EventPayload.QUERY_STR]
                self._trace_data.query_data.id = _hash_bytes(query_text.encode())
                self._trace_data.query_data.query_text = query_text
        return event_id

    def on_event_end(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        if payload is not None:
            # if event_type is CBEventType.QUERY:
            #     self._trace_data.response_text = payload[EventPayload.RESPONSE]
            if event_type is CBEventType.RETRIEVE:
                for node_with_score in payload[EventPayload.NODES]:
                    node = node_with_score.node
                    score = node_with_score.score
                    self._trace_data.query_data.document_hashes.append(node.hash)
                    self._trace_data.query_data.document_ids.append(node.id_)
                    self._trace_data.query_data.scores.append(score)
                    self._trace_data.document_datas.append(
                        OpenInferenceDocumentData(
                            id=node.id_,
                            document_text=node.text,
                            document_hash=node.hash,
                        )
                    )
            elif event_type is CBEventType.LLM:
                self._trace_data.query_data.response_text = payload[
                    EventPayload.RESPONSE
                ]
            elif event_type is CBEventType.EMBEDDING:
                self._trace_data.query_data.query_embedding = payload[
                    EventPayload.EMBEDDINGS
                ][
                    0
                ]  # when does this list have more than one element?

    @property
    def query_dataframe(self) -> "DataFrame":
        return self._query_data_buffer.dataframe

    @property
    def document_dataframe(self) -> "DataFrame":
        return self._document_data_buffer.dataframe
