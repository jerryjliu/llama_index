from llama_index.core.multi_modal_llms.base import MultiModalLLM
from llama_index.multi_modal_llms.nvidia import NVIDIAMultiModal
from llama_index.multi_modal_llms.nvidia.utils import (
    NVIDIA_MULTI_MODAL_MODELS,
    DEFAULT_MODEL,
)
import base64
import os
from typing import Any, Dict, List, Union

import pytest
import requests
from llama_index.core.base.llms.types import (
    CompletionResponse,
    ChatMessage,
    ChatResponse,
)
from llama_index.core.schema import ImageDocument
import numpy as np
from PIL import Image
import tempfile

# TODO: multiple texts
# TODO: accuracy tests

#
# API Specification -
#
#  - User message may contain 1 or more image_url
#  - url is either a url to an image or base64 encoded image
#  - format for base64 is "data:image/png;{type}},..."
#  - supported image types are png, jpeg (or jpg), webp, gif (non-animated)
#

#
# note: differences between api catalog and openai api
#  - openai api supports server-side image download, api catalog does not consistently
#   - NVIDIAMultiModal does client side download to simulate the same behavior
#  - NVIDIAMultiModal will automatically read local files and convert them to base64
#  - openai api always uses {"image_url": {"url": "..."}}
#     where api catalog sometimes uses {"image_url": "..."}
#

image_urls = [
    "https://res.cloudinary.com/hello-tickets/image/upload/c_limit,f_auto,q_auto,w_1920/v1640835927/o3pfl41q7m5bj8jardk0.jpg",
    "https://www.visualcapitalist.com/wp-content/uploads/2023/10/US_Mortgage_Rate_Surge-Sept-11-1.jpg",
    "https://www.sportsnet.ca/wp-content/uploads/2023/11/CP1688996471-1040x572.jpg",
    # Add yours here!
]

MODELS = list(NVIDIA_MULTI_MODAL_MODELS.keys())


def test_embedding_class():
    names_of_base_classes = [b.__name__ for b in NVIDIAMultiModal.__mro__]
    assert MultiModalLLM.__name__ in names_of_base_classes


def test_init():
    m = NVIDIAMultiModal(max_tokens=400)
    assert m.max_tokens == 400


def urlToBase64(url):
    return base64.b64encode(requests.get(url).content).decode("utf-8")


@pytest.fixture(scope="session")
def temp_image_path(suffix: str):
    # Create a white square image
    white_square = np.ones((100, 100, 3), dtype=np.uint8) * 255
    image = Image.fromarray(white_square)

    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix=f".{suffix}", delete=False) as temp_file:
        image.save(temp_file, format=suffix.upper())
        temp_path = temp_file.name

    yield temp_path

    # Clean up the temporary file after the test
    os.unlink(temp_path)


@pytest.fixture(scope="session")
def get_asset_id():
    content_type = "image/jpg"
    description = "example-image-from-lc-nv-ai-e-notebook"

    create_response = requests.post(
        "https://api.nvcf.nvidia.com/v2/nvcf/assets",
        headers={
            "Authorization": f"Bearer {os.environ['NVIDIA_API_KEY']}",
            "accept": "application/json",
            "Content-Type": "application/json",
        },
        json={"contentType": content_type, "description": description},
    )
    create_response.raise_for_status()

    upload_response = requests.put(
        create_response.json()["uploadUrl"],
        headers={
            "Content-Type": content_type,
            "x-amz-meta-nvcf-asset-description": description,
        },
        data=requests.get(image_urls[0]).content,
    )
    upload_response.raise_for_status()

    return create_response.json()["assetId"]


def test_class():
    emb = NVIDIAMultiModal(api_key="BOGUS")
    assert isinstance(emb, MultiModalLLM)


@pytest.mark.parametrize(
    "content",
    [
        [ImageDocument(image_url=image_urls[0])],
        [ImageDocument(image=urlToBase64(image_urls[0]), mimetype="jpeg")],
    ],
)
@pytest.mark.parametrize(
    "func",
    ["invoke", "stream"],
)
def test_vlm_input_style(
    vlm_model: str,
    content: List[ImageDocument],
    func: str,
) -> None:
    llm = NVIDIAMultiModal(model=vlm_model)
    assert vlm_model in MODELS
    if func == "invoke":
        response = llm.complete(prompt="Describe the Image.", image_documents=content)
        assert isinstance(response, CompletionResponse)
    if func == "stream":
        for token in llm.stream_complete(
            prompt="Describe the Image.", image_documents=content
        ):
            assert isinstance(token.text, str)


@pytest.mark.parametrize(
    "suffix",
    ["jpeg", "png", "webp", "gif"],
    scope="session",
)
def test_vlm_image_type(
    suffix: str,
    temp_image_path: str,
    vlm_model: str,
) -> None:
    llm = NVIDIAMultiModal(model=vlm_model)
    response = llm.complete(
        "Describe image", image_documents=[ImageDocument(image_path=temp_image_path)]
    )
    assert isinstance(response, CompletionResponse)
    assert isinstance(response.text, str)


pytest.mark.skipif(os.path.isfile("data/nvidia-picasso-large.png"))


def test_vlm_image_large(
    vlm_model: str,
) -> None:
    chat = NVIDIAMultiModal(model=vlm_model)
    response = chat.complete(
        prompt="Describe image",
        image_documents=[ImageDocument(image_path="data/nvidia-picasso-large.png")],
    )
    assert isinstance(response, CompletionResponse)
    assert isinstance(response.text, str)


@pytest.mark.parametrize(
    "suffix",
    ["jpeg", "png", "webp", "gif"],
    scope="session",
)
def test_vlm_two_images(
    suffix: str,
    temp_image_path: str,
    vlm_model: str,
) -> None:
    chat = NVIDIAMultiModal(model=vlm_model)
    response = chat.complete(
        prompt="Describe image",
        image_documents=[
            ImageDocument(image_path=temp_image_path),
            ImageDocument(image_path=temp_image_path),
        ],
    )
    assert isinstance(response, CompletionResponse)
    assert isinstance(response.text, str)


@pytest.mark.parametrize(
    "content",
    [
        [ImageDocument(metadata={"asset_id": ""})],
    ],
)
@pytest.mark.parametrize(
    "func",
    ["invoke", "stream"],
)
def test_vlm_asset_id(
    vlm_model: str,
    content: Union[str, List[Union[str, Dict[str, Any]]]],
    func: str,
    get_asset_id: str,
) -> None:
    assert isinstance(content[0], ImageDocument)
    content[0].metadata["asset_id"] = get_asset_id

    assert content[0].metadata["asset_id"] != ""

    chat = NVIDIAMultiModal(model=vlm_model)
    if func == "invoke":
        response = chat.complete(prompt="Describe image", image_documents=content)
        assert isinstance(response, CompletionResponse)
        assert isinstance(response.text, str)
    if func == "stream":
        for token in chat.stream_complete(
            prompt="Describe image", image_documents=content
        ):
            assert isinstance(token.text, str)


## ------------------------- chat/stream_chat test cases ------------------------- ##


@pytest.mark.parametrize(
    "func",
    ["chat", "stream_chat"],
)
def test_stream_chat_multiple_messages(vlm_model: str, func: str) -> None:
    """Test streaming chat with multiple messages and images."""
    llm = NVIDIAMultiModal(model=vlm_model)

    messages = [
        ChatMessage(
            role="user",
            content=[
                {"type": "text", "text": "Describe the first image:"},
                {"type": "image_url", "image_url": image_urls[0]},
            ],
        ),
        ChatMessage(
            role="assistant", content="This is a response about the first image."
        ),
        ChatMessage(
            role="user",
            content=[
                {"type": "text", "text": "Now describe this second image:"},
                {"type": "image_url", "image_url": image_urls[1]},
            ],
        ),
    ]

    if func == "chat":
        response = llm.chat(messages)
        assert isinstance(response, ChatResponse)
        assert isinstance(response.delta, str)
    if func == "stream_chat":
        for token in llm.stream_chat(messages):
            assert isinstance(token.delta, str)


@pytest.mark.parametrize(
    "content",
    [
        """<img src="data:image/jpg;asset_id,{asset_id}"/>""",
        [
            {
                "type": "image_url",
                "image_url": "data:image/jpg;asset_id,{asset_id}",
            }
        ],
        [
            {"type": "text", "text": "Describe this image:"},
            {"type": "image_url", "image_url": image_urls[1]},
        ],
    ],
)
@pytest.mark.parametrize(
    "func",
    ["chat", "stream_chat"],
)
def test_vlm_asset_id_chat(
    vlm_model: str,
    content: Union[str, List[Union[str, Dict[str, Any]]]],
    func: str,
    get_asset_id: str,
) -> None:
    def fill(
        item: Any,
        asset_id: str,
    ) -> Union[str, Any]:
        # do not mutate item, mutation will cause cross test contamination
        result: Any
        if isinstance(item, str):
            result = item.format(asset_id=asset_id)
        elif isinstance(item, ChatMessage):
            result = item.model_copy(update={"content": fill(item.content, asset_id)})
        elif isinstance(item, list):
            result = [fill(sub_item, asset_id) for sub_item in item]
        elif isinstance(item, dict):
            result = {key: fill(value, asset_id) for key, value in item.items()}
        return result

    asset_id = get_asset_id
    assert asset_id != ""
    content = fill(content, asset_id)

    llm = NVIDIAMultiModal(model=vlm_model)
    if func == "chat":
        response = llm.chat([ChatMessage(role="user", content=content)])
        assert isinstance(response, ChatResponse)
        assert isinstance(response.delta, str)
    if func == "stream_chat":
        for token in llm.stream_chat([ChatMessage(role="user", content=content)]):
            assert isinstance(token.delta, str)


@pytest.mark.parametrize(
    "func",
    ["chat", "stream_chat"],
    scope="session",
)
@pytest.mark.parametrize(
    "suffix",
    ["jpeg", "png", "webp", "gif"],
    scope="session",
)
def test_vlm_image_type_chat(
    suffix: str, temp_image_path: str, vlm_model: str, func: str
) -> None:
    llm = NVIDIAMultiModal(model=vlm_model)
    if func == "chat":
        response = llm.chat(
            [ChatMessage(content=[{"type": "image_url", "image_url": temp_image_path}])]
        )
        assert isinstance(response, ChatResponse)
        assert isinstance(response.delta, str)
    if func == "stream_chat":
        for token in llm.stream_chat(
            [ChatMessage(content=[{"type": "image_url", "image_url": temp_image_path}])]
        ):
            assert isinstance(token, ChatResponse)


## ------------------------- Async test cases ------------------------- ##


@pytest.mark.parametrize(
    "content",
    [
        [ImageDocument(image_url=image_urls[0])],
        [ImageDocument(image=urlToBase64(image_urls[0]), mimetype="jpeg")],
    ],
)
@pytest.mark.asyncio()
async def test_vlm_input_style_async(
    vlm_model: str,
    content: List[ImageDocument],
) -> None:
    llm = NVIDIAMultiModal(model=vlm_model)
    assert vlm_model in MODELS

    # Await the completion of the async call
    response = await llm.acomplete(
        prompt="Describe the Image.", image_documents=content
    )

    # Ensure the response is a valid CompletionResponse
    assert isinstance(response, CompletionResponse)
    assert isinstance(response.text, str)


@pytest.mark.asyncio()
async def test_vlm_chat_async(vlm_model: str) -> None:
    llm = NVIDIAMultiModal(model=vlm_model)
    messages = [
        ChatMessage(
            role="user",
            content=[
                {"type": "text", "text": "Describe the first image:"},
                {"type": "image_url", "image_url": image_urls[0]},
            ],
        ),
        ChatMessage(
            role="assistant", content="This is a response about the first image."
        ),
        ChatMessage(
            role="user",
            content=[
                {"type": "text", "text": "Now describe this second image:"},
                {"type": "image_url", "image_url": image_urls[1]},
            ],
        ),
    ]
    response = await llm.achat(messages)
    assert isinstance(response, ChatResponse)
    assert isinstance(response.delta, str)


@pytest.mark.asyncio()
async def test_vlm_chat_async_stream(vlm_model: str) -> None:
    llm = NVIDIAMultiModal(model=vlm_model)
    messages = [
        ChatMessage(
            role="user",
            content=[
                {"type": "text", "text": "Describe the first image:"},
                {"type": "image_url", "image_url": image_urls[0]},
            ],
        ),
        ChatMessage(
            role="assistant", content="This is a response about the first image."
        ),
        ChatMessage(
            role="user",
            content=[
                {"type": "text", "text": "Now describe this second image:"},
                {"type": "image_url", "image_url": image_urls[1]},
            ],
        ),
    ]
    async for token in await llm.astream_chat(messages):
        assert isinstance(token, ChatResponse)
        assert isinstance(token.delta, str)


def test_default_known() -> None:
    """
    Test that a model in the model table will be accepted.
    """
    # check if default model is getting set
    with pytest.warns(UserWarning) as record:
        x = NVIDIAMultiModal()
        assert x.model == DEFAULT_MODEL
    assert len(record) == 1
    assert f"Default model is set as: {DEFAULT_MODEL}" in str(record[0].message)
