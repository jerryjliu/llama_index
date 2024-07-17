import pytest
import inspect

from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler, CBEventType
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.nvidia import NVIDIAEmbedding

from openai import AuthenticationError

from pytest_httpx import HTTPXMock


@pytest.fixture()
def mock_integration_api(httpx_mock: HTTPXMock):
    BASE_URL = "https://integrate.api.nvidia.com/v1"
    NVIDIA_API_KEY = "dummy"  # Replace with your actual API key

    mock_response = {"object": "list", "data": [{"index": 0, "embedding": ""}]}

    httpx_mock.add_response(
        method="POST",
        url=f"{BASE_URL}/embeddings",
        json=mock_response,
        headers={"Content-Type": "application/json"},
        status_code=200,
    )

    return BASE_URL, NVIDIA_API_KEY


def test_embedding_class():
    emb = NVIDIAEmbedding()
    assert isinstance(emb, BaseEmbedding)


def test_nvidia_embedding_param_setting():
    emb = NVIDIAEmbedding(
        model="test-model",
        truncate="END",
        timeout=20,
        max_retries=10,
        embed_batch_size=15,
    )

    assert emb.model == "test-model"
    assert emb.truncate == "END"
    assert emb._client.timeout == 20
    assert emb._client.max_retries == 10
    assert emb._aclient.timeout == 20
    assert emb._aclient.max_retries == 10
    assert emb.embed_batch_size == 15


def test_nvidia_embedding_throws_on_batches_larger_than_259():
    with pytest.raises(ValueError):
        NVIDIAEmbedding(embed_batch_size=300)


def test_nvidia_embedding_async():
    emb = NVIDIAEmbedding()

    assert inspect.iscoroutinefunction(emb._aget_query_embedding)
    query_emb = emb._aget_query_embedding("hi")
    assert inspect.isawaitable(query_emb)
    query_emb.close()

    assert inspect.iscoroutinefunction(emb._aget_text_embedding)
    text_emb = emb._aget_text_embedding("hi")
    assert inspect.isawaitable(text_emb)
    text_emb.close()

    assert inspect.iscoroutinefunction(emb._aget_text_embeddings)
    text_embs = emb._aget_text_embeddings(["hi", "hello"])
    assert inspect.isawaitable(text_embs)
    text_embs.close()


def test_nvidia_embedding_callback(mock_integration_api):
    llama_debug = LlamaDebugHandler(print_trace_on_end=False)
    assert len(llama_debug.get_events()) == 0

    callback_manager = CallbackManager([llama_debug])
    emb = NVIDIAEmbedding(api_key="dummy", callback_manager=callback_manager)

    try:
        emb.get_text_embedding("hi")
    except AuthenticationError:
        pass

    assert len(llama_debug.get_events(CBEventType.EMBEDDING)) > 0


def test_nvidia_embedding_throws_with_invalid_key(mock_integration_api):
    emb = NVIDIAEmbedding(api_key="invalid")
    emb.get_text_embedding("hi")
