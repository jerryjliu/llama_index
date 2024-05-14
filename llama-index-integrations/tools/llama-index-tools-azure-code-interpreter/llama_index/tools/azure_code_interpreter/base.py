"""Azure Dynamic Sessions tool spec."""

import base64
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import importlib.metadata
from io import BufferedReader, BytesIO
from typing import Any, Callable, List, Optional
import urllib
from uuid import uuid4
import re
import os

from azure.identity import DefaultAzureCredential
from azure.core.credentials import AccessToken
import requests

from llama_index.core.tools.tool_spec.base import BaseToolSpec


@dataclass
class RemoteFileMetadata:
    """Metadata for a file in the session."""

    filename: str
    """The filename relative to `/mnt/data`."""

    size_in_bytes: int
    """The size of the file in bytes."""

    file_full_path: str
    """The full path of the file."""

    @staticmethod
    def from_dict(data: dict) -> "RemoteFileMetadata":
        """Create a RemoteFileMetadata object from a dictionary."""
        filename = data["filename"]
        if filename.startswith("/mnt/data"):
            file_full_path = filename
        else:
            file_full_path = f"/mnt/data/{filename}"
        return RemoteFileMetadata(
            filename=filename,
            size_in_bytes=data["size"],
            file_full_path=file_full_path,
        )


def _sanitize_input(query: str) -> str:
    """Sanitize input and remove whitespace, backtick, and markdown.

    Args:
        query: The query to sanitize

    Returns:
        str: The sanitized query
    """
    # Removes `, whitespace & python from start
    query = re.sub(r"^(\s|`)*(?i:python)?\s*", "", query)
    # Removes whitespace & ` from end
    query = re.sub(r"(\s|`)*$", "", query)
    # Add new line if no new line was appended at the end of the query
    if not query.endswith("\n"):
        query += "\n"
    return query


class AzureCodeInterpreterToolSpec(BaseToolSpec):
    """Azure Code Interpreter tool spec.

    Leverages Azure Dynamic Sessions to execute Python code.
    """

    spec_functions = ["code_interpreter", "list_files"]

    def __init__(
        self,
        pool_managment_endpoint: str,
        session_id: Optional[str] = None,
        local_save_path: Optional[str] = None,
        sanitize_input: bool = True,
    ) -> None:
        """Initialize with parameters."""
        self.pool_management_endpoint: str = pool_managment_endpoint
        self.access_token: Optional[AccessToken] = None

        def _access_token_provider_factory() -> Callable[[], Optional[str]]:
            def access_token_provider() -> Optional[str]:
                """Create a function that returns an access token."""
                if self.access_token is None or datetime.fromtimestamp(
                    self.access_token.expires_on, timezone.utc
                ) < (datetime.now(timezone.utc) + timedelta(minutes=5)):
                    credential = DefaultAzureCredential()
                    self.access_token = credential.get_token(
                        "https://dynamicsessions.io/.default"
                    )
                return self.access_token.token

            return access_token_provider

        self.access_token_provider: Callable[
            [], Optional[str]
        ] = _access_token_provider_factory()
        """A function that returns the access token to use for the session pool."""

        self.session_id: str = session_id or str(uuid4())
        """The session ID to use for the session pool. Defaults to a random UUID."""

        self.sanitize_input: bool = sanitize_input
        """Whether to sanitize input before executing it."""

        if local_save_path:
            if not os.path.exists(local_save_path):
                raise Exception(f"Local save path {local_save_path} does not exist.")

        self.local_save_path: Optional[str] = local_save_path
        """The local path to save files generated by Python interpreter."""

        try:
            _package_version = importlib.metadata.version(
                "llamaindex-azure-code-interpreter"
            )
        except importlib.metadata.PackageNotFoundError:
            _package_version = "0.0.0"

        self.user_agent = (
            f"llamaindex-azure-code-interpreter/{_package_version} (Language=Python)"
        )

    def _build_url(self, path: str) -> str:
        pool_management_endpoint = self.pool_management_endpoint
        if not pool_management_endpoint:
            raise ValueError("pool_management_endpoint is not set")

        if not pool_management_endpoint.endswith("/"):
            pool_management_endpoint += "/"

        encoded_session_id = urllib.parse.quote(self.session_id)
        query = f"identifier={encoded_session_id}&api-version=2024-02-02-preview"
        query_separator = "&" if "?" in pool_management_endpoint else "?"

        return pool_management_endpoint + path + query_separator + query

    def code_interpreter(self, python_code: str) -> dict:
        """
        This tool is used to execute python commands when you need to perform calculations or computations in a Session.
        Input should be a valid python command. The tool returns the result, stdout, and stderr.

        Args:
            python_code (str): Python code to be executed generated by llm.
        """
        if self.sanitize_input:
            python_code = _sanitize_input(python_code)

        access_token = self.access_token_provider()
        api_url = self._build_url("code/execute")
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "User-Agent": self.user_agent,
        }
        body = {
            "properties": {
                "codeInputType": "inline",
                "executionType": "synchronous",
                "code": python_code,
            }
        }

        response = requests.post(api_url, headers=headers, json=body)
        response.raise_for_status()
        response_json = response.json()
        if "properties" in response_json:
            if (
                "result" in response_json["properties"]
                and response_json["properties"]["result"]
            ):
                if isinstance(response_json["properties"]["result"], dict):
                    if "base64_data" in response_json["properties"]["result"]:
                        base64_encoded_data = response_json["properties"]["result"][
                            "base64_data"
                        ]
                        if self.local_save_path:
                            file_path = f"{self.local_save_path}/{self.session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{response_json['properties']['result']['format']}"
                            decoded_data = base64.b64decode(base64_encoded_data)
                            with open(file_path, "wb") as f:
                                f.write(decoded_data)
                            # Check if file is written to the file path successfully. if so, update the response_json
                            response_json["properties"]["result"][
                                "saved_to_local_path"
                            ] = response_json["properties"]["result"].pop("base64_data")
                            if os.path.exists(file_path):
                                response_json["properties"]["result"][
                                    "saved_to_local_path"
                                ] = True
                            else:
                                response_json["properties"]["result"][
                                    "saved_to_local_path"
                                ] = False
                        else:
                            response_json["properties"]["result"]["base64_data"] = ""
        return response_json

    def upload_file(
        self,
        data: Optional[Any] = None,
        local_file_path: Optional[str] = None,
    ) -> List[RemoteFileMetadata]:
        """Upload a file to the session under the path /mnt/data.

        Args:
            data: The data to upload.
            local_file_path: The path to the local file to upload.

        Returns:
            List[RemoteFileMetadata]: The list of metadatas for the uploaded files.
        """
        if data and local_file_path:
            raise ValueError("data and local_file_path cannot be provided together")

        if local_file_path:
            remote_file_path = f"/mnt/data/{os.path.basename(local_file_path)}"
            data = open(local_file_path, "rb")

        access_token = self.access_token_provider()
        if not remote_file_path.startswith("/mnt/data"):
            remote_file_path = f"/mnt/data/{remote_file_path}"
        api_url = self._build_url("files/upload")
        headers = {
            "Authorization": f"Bearer {access_token}",
        }

        files = [("file", (remote_file_path, data, "application/octet-stream"))]

        response = requests.request("POST", api_url, headers=headers, files=files)
        response.raise_for_status()

        response_json = response.json()
        remote_files_metadatas = []
        for entry in response_json["value"]:
            if "properties" in entry:
                remote_files_metadatas.append(
                    RemoteFileMetadata.from_dict(entry["properties"])
                )
        return remote_files_metadatas

    def download_file_to_local(
        self, remote_file_path: str, local_file_path: Optional[str] = None
    ) -> Optional[BufferedReader]:
        """Download a file from the session back to your local environment.

        Args:
            remote_file_path: The path to download the file from, relative to `/mnt/data`.
            local_file_path: The path to save the downloaded file to. If not provided, the file is returned as a BufferedReader.

        Returns:
            BufferedReader: The data of the downloaded file.
        """
        access_token = self.access_token_provider()
        # In case if the file path LLM provides is absolute, remove the /mnt/data/ prefix
        remote_file_path = remote_file_path.replace("/mnt/data/", "")
        api_url = self._build_url(f"files/content/{remote_file_path}")
        headers = {
            "Authorization": f"Bearer {access_token}",
        }

        response = requests.get(api_url, headers=headers)
        response.raise_for_status()

        if local_file_path:
            with open(local_file_path, "wb") as f:
                f.write(response.content)
            return None

        return BytesIO(response.content)

    def list_files(self) -> List[RemoteFileMetadata]:
        """List the files in the session.

        Returns:
            List[RemoteFileMetadata]: The metadata for the files in the session
        """
        access_token = self.access_token_provider()
        api_url = self._build_url("files")
        headers = {
            "Authorization": f"Bearer {access_token}",
        }

        response = requests.get(api_url, headers=headers)
        response.raise_for_status()

        response_json = response.json()
        return [
            RemoteFileMetadata.from_dict(entry["properties"])
            for entry in response_json["value"]
        ]
