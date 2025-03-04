import base64
import filetype
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from llama_index.core.schema import ImageDocument
import json
import os
import re
import urllib
from llama_index.core.base.llms.types import ChatMessage

DEFAULT_MODEL = "nvidia/neva-22b"
BASE_URL = "https://ai.api.nvidia.com/v1/"

KNOWN_URLS = [
    BASE_URL,
    "https://integrate.api.nvidia.com/v1",
]

NVIDIA_MULTI_MODAL_MODELS = {
    "adept/fuyu-8b": {"endpoint": f"{BASE_URL}vlm/adept/fuyu-8b", "type": "nv-vlm"},
    "google/deplot": {"endpoint": f"{BASE_URL}vlm/google/deplot", "type": "nv-vlm"},
    "microsoft/kosmos-2": {
        "endpoint": f"{BASE_URL}vlm/microsoft/kosmos-2",
        "type": "nv-vlm",
    },
    "nvidia/neva-22b": {"endpoint": f"{BASE_URL}vlm/nvidia/neva-22b", "type": "nv-vlm"},
    "google/paligemma": {
        "endpoint": f"{BASE_URL}vlm/google/paligemma",
        "type": "nv-vlm",
    },
    "microsoft/phi-3-vision-128k-instruct": {
        "endpoint": f"{BASE_URL}vlm/microsoft/phi-3-vision-128k-instruct",
        "type": "vlm",
    },
    "microsoft/phi-3.5-vision-instruct": {
        "endpoint": f"{BASE_URL}microsoft/microsoft/phi-3_5-vision-instruct",
        "type": "nv-vlm",
    },
    "nvidia/vila": {"endpoint": f"{BASE_URL}vlm/nvidia/vila", "type": "nv-vlm"},
    "meta/llama-3.2-11b-vision-instruct": {
        "endpoint": f"{BASE_URL}gr/meta/llama-3.2-11b-vision-instruct/chat/completions",
        "type": "vlm",
    },
    "meta/llama-3.2-90b-vision-instruct": {
        "endpoint": f"{BASE_URL}/gr/meta/llama-3.2-90b-vision-instruct/chat/completions",
        "type": "vlm",
    },
}


def infer_image_mimetype_from_base64(base64_string) -> str:
    # Decode the base64 string
    decoded_data = base64.b64decode(base64_string)

    # Use filetype to guess the MIME type
    kind = filetype.guess(decoded_data)

    # Return the MIME type if detected, otherwise return None
    return kind.mime if kind is not None else None


def infer_image_mimetype_from_file_path(image_file_path: str) -> str:
    # Get the file extension
    file_extension = image_file_path.split(".")[-1].lower()

    # Map file extensions to mimetypes
    # Claude 3 support the base64 source type for images, and the image/jpeg, image/png, image/gif, and image/webp media types.
    # https://docs.anthropic.com/claude/reference/messages_post
    if file_extension in ["jpg", "jpeg", "png", "webp", "gif"]:
        return file_extension
    return "png"
    # Add more mappings for other image types if needed

    # If the file extension is not recognized


# Function to encode the image to base64 content
def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def create_image_content(image_document) -> Optional[Dict[str, Any]]:
    """
    Create the image content based on the provided image document.
    """
    if image_document.image:
        mimetype = (
            image_document.mimetype
            if image_document.mimetype
            else infer_image_mimetype_from_base64(image_document.image)
        )
        return {
            "type": "text",
            "text": f'<img src="data:image/{mimetype};base64,{image_document.image}" />',
        }, ""

    elif "asset_id" in image_document.metadata:
        asset_id = image_document.metadata["asset_id"]
        mimetype = image_document.mimetype if image_document.mimetype else "jpeg"
        return {
            "type": "text",
            "text": f'<img src="data:image/{mimetype};asset_id,{asset_id}" />',
        }, asset_id

    elif image_document.image_url and image_document.image_url != "":
        mimetype = infer_image_mimetype_from_file_path(image_document.image_url)
        return {
            "type": "image_url",
            "image_url": image_document.image_url,
        }, ""
    elif (
        "file_path" in image_document.metadata
        and image_document.metadata["file_path"] != ""
    ):
        mimetype = infer_image_mimetype_from_file_path(
            image_document.metadata["file_path"]
        )
        base64_image = encode_image(image_document.metadata["file_path"])
        return {
            "type": "text",
            "text": f'<img src="data:image/{mimetype};base64,{base64_image}" />',
        }, ""

    return None, None


def generate_nvidia_multi_modal_chat_message(
    model: str,
    prompt: Optional[str] = None,
    inputs: Optional[List[ChatMessage]] = [],
    image_documents: Optional[Sequence[ImageDocument]] = [],
) -> List[Dict[str, Any]]:
    # If image_documents is None, return a text-only chat message
    completion_content = []
    asset_ids = []
    extra_headers = {}
    model_type = NVIDIA_MULTI_MODAL_MODELS[model]["type"]

    for input in inputs:
        if input.content:
            asset_ids.extend(_nv_vlm_get_asset_ids(input.content))

    # Process each image document
    for image_document in image_documents:
        image_content, asset_id = create_image_content(image_document)
        if image_content:
            completion_content.append(image_content)
        if asset_id:
            asset_ids.append(asset_id)

    if len(asset_ids) > 0:
        extra_headers["NVCF-INPUT-ASSET-REFERENCES"] = ",".join(asset_ids)

    # Append the text prompt to the completion content
    if prompt:
        completion_content.append({"type": "text", "text": prompt})
        return completion_content, extra_headers

    inputs = [
        {
            "role": message.role,
            "content": _nv_vlm_adjust_input(message, model_type).content,
        }
        for message in inputs
    ]
    return inputs, extra_headers


def process_response(response) -> List[dict]:
    """General-purpose response processing for single responses and streams."""
    if hasattr(response, "json"):  ## For single response (i.e. non-streaming)
        try:
            return [response.json()]
        except json.JSONDecodeError:
            response = str(response.__dict__)
    if isinstance(response, str):  ## For set of responses (i.e. streaming)
        msg_list = []
        for msg in response.split("\n\n"):
            if "{" not in msg:
                continue
            msg_list += [json.loads(msg[msg.find("{") :])]
        return msg_list
    raise ValueError(f"Received ill-formed response: {response}")


def aggregate_msgs(msg_list: Sequence[dict]) -> Tuple[dict, bool]:
    """Dig out relevant details of aggregated message."""
    content_buffer: Dict[str, Any] = {}
    content_holder: Dict[Any, Any] = {}
    usage_holder: Dict[Any, Any] = {}  ####
    finish_reason_holder: Optional[str] = None
    is_stopped = False
    for msg in msg_list:
        usage_holder = msg.get("usage", {})  ####
        if "choices" in msg:
            ## Tease out ['choices'][0]...['delta'/'message']
            # when streaming w/ usage info, we may get a response
            #  w/ choices: [] that includes final usage info
            choices = msg.get("choices", [{}])
            msg = choices[0] if choices else {}
            # TODO: this needs to be fixed, the fact we only
            #       use the first choice breaks the interface
            finish_reason_holder = msg.get("finish_reason", None)
            is_stopped = finish_reason_holder == "stop"
            msg = msg.get("delta", msg.get("message", msg.get("text", "")))
            if not isinstance(msg, dict):
                msg = {"content": msg}
        elif "data" in msg:
            ## Tease out ['data'][0]...['embedding']
            msg = msg.get("data", [{}])[0]
        content_holder = msg
        for k, v in msg.items():
            if k in ("content",) and k in content_buffer:
                content_buffer[k] += v
            else:
                content_buffer[k] = v
        if is_stopped:
            break
    content_holder = {
        **content_holder,
        **content_buffer,
        "text": content_buffer["content"],
    }
    if usage_holder:
        content_holder.update(token_usage=usage_holder)  ####
    if finish_reason_holder:
        content_holder.update(finish_reason=finish_reason_holder)
    return content_holder, is_stopped


def _nv_vlm_adjust_input(message: ChatMessage, model_type: str) -> ChatMessage:
    """
    This function converts the OpenAI VLM API input message to
    NVIDIA VLM API input message, in place.

    The NVIDIA VLM API input message.content:
        {
            "role": "user",
            "content": [
                ...,
                {
                    "type": "image_url",
                    "image_url": "{data}"
                },
                ...
            ]
        }
    where OpenAI VLM API input message.content:
        {
            "role": "user",
            "content": [
                ...,
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "{url | data}"
                    }
                },
                ...
            ]
        }

    In the process, it accepts a url or file and converts them to
    data urls.
    """
    if content := message.content:
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and "image_url" in part:
                    if (
                        isinstance(part["image_url"], dict)
                        and "url" in part["image_url"]
                    ):
                        url = _url_to_b64_string(part["image_url"]["url"])
                        if model_type == "nv-vlm":
                            part["image_url"] = url
                        else:
                            part["image_url"]["url"] = url
    return message


def _nv_vlm_get_asset_ids(
    content: Union[str, List[Union[str, Dict[str, Any]]]],
) -> List[str]:
    """
    Extracts asset IDs from the message content.

    VLM APIs accept asset IDs as input in two forms:
     - content = [{"image_url": {"url": "data:image/{type};asset_id,{asset_id}"}}*]
     - content = .*<img src="data:image/{type};asset_id,{asset_id}"/>.*
    """

    def extract_asset_id(data: str) -> List[str]:
        pattern = re.compile(r'data:image/[^;]+;asset_id,([^"\'\s]+)')
        return pattern.findall(data)

    asset_ids = []
    if isinstance(content, str):
        asset_ids.extend(extract_asset_id(content))
    elif isinstance(content, list):
        for part in content:
            if isinstance(part, str):
                asset_ids.extend(extract_asset_id(part))
            elif isinstance(part, dict) and "image_url" in part:
                image_url = part["image_url"]
                if isinstance(image_url, str):
                    asset_ids.extend(extract_asset_id(image_url))
            elif isinstance(part, dict) and "text" in part:
                image_url = part["text"]
                if isinstance(image_url, str):
                    asset_ids.extend(extract_asset_id(image_url))

    return asset_ids


def _is_url(s: str) -> bool:
    try:
        result = urllib.parse.urlparse(s)
        return all([result.scheme, result.netloc])
    except Exception as e:
        raise f"Unable to parse URL: {e}"
        return False


def _url_to_b64_string(image_source: str) -> str:
    try:
        if _is_url(image_source):
            return image_source
        elif image_source.startswith("data:image"):
            return image_source
        elif os.path.exists(image_source):
            encoded = encode_image(image_source)
            image_type = infer_image_mimetype_from_base64(encoded)
            return f"data:image/{image_type};base64,{encoded}"
        else:
            raise ValueError(
                "The provided string is not a valid URL, base64, or file path."
            )
    except Exception as e:
        raise ValueError(f"Unable to process the provided image source: {e}")
