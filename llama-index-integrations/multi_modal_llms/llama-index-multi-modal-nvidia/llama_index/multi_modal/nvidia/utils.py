import base64
import filetype
from typing import Any, Dict, List, Optional, Sequence, Tuple
from llama_index.core.schema import ImageDocument
import json

DEFAULT_MODEL = "google/deplot"
BASE_URL = "https://ai.api.nvidia.com/v1/"

KNOWN_URLS = [
    BASE_URL,
    "https://integrate.api.nvidia.com/v1",
]

NVIDIA_MULTI_MODAL_MODELS = {
    "adept/fuyu-8b": {"endpoint": f"{BASE_URL}vlm/adept/fuyu-8b"},
    "google/deplot": {"endpoint": f"{BASE_URL}vlm/google/deplot"},
    "microsoft/kosmos-2": {"endpoint": f"{BASE_URL}vlm/microsoft/kosmos-2"},
    "nvidia/neva-22b": {"endpoint": f"{BASE_URL}vlm/nvidia/neva-22b"},
    "google/paligemma": {"endpoint": f"{BASE_URL}vlm/google/paligemma"},
    "microsoft/phi-3-vision-128k-instruct": {
        "endpoint": f"{BASE_URL}vlm/microsoft/phi-3-vision-128k-instruct"
    },
    "microsoft/phi-3.5-vision-instruct": {
        "endpoint": f"{BASE_URL}microsoft/microsoft/phi-3_5-vision-instruct"
    },
    "nvidia/vila": {"endpoint": f"{BASE_URL}vlm/nvidia/vila"},
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
    if file_extension in ["jpg", "jpeg", "png"]:
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
        mimetype = image_document.mimetype if image_document.mimetype else "jpeg"
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
    prompt: str,
    role: str,
    image_documents: Optional[Sequence[ImageDocument]] = None,
) -> List[Dict[str, Any]]:
    # If image_documents is None, return a text-only chat message
    completion_content = []
    asset_ids = []
    extra_headers = {}

    # Process each image document
    for image_document in image_documents:
        image_content, asset_id = create_image_content(image_document)
        if image_content:
            completion_content.append(image_content)
        if asset_id:
            asset_ids.append(asset_id)

    # Append the text prompt to the completion content
    completion_content.append({"type": "text", "text": prompt})

    if asset_ids:
        extra_headers["NVCF-INPUT-ASSET-REFERENCES"] = ",".join(asset_ids)

    return [{"role": role, "content": completion_content}], extra_headers


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
