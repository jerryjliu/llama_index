"""
GCS file and directory reader.

A loader that fetches a file or iterates through a directory on Google Cloud Storage (GCS).

"""

import json
import logging
from typing import Callable, Dict, List, Optional, Union
from typing_extensions import Annotated
from datetime import datetime
from pathlib import Path

from google.oauth2 import service_account
from google.auth.exceptions import DefaultCredentialsError
from llama_index.core.readers import SimpleDirectoryReader, FileSystemReaderMixin
from llama_index.core.readers.base import (
    BasePydanticReader,
    ResourcesReaderMixin,
    BaseReader,
)
from llama_index.core.schema import Document
from llama_index.core.bridge.pydantic import Field, WithJsonSchema

# Set up logging
logger = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/devstorage.read_only"]


FileMetadataCallable = Annotated[
    Callable[[str], Dict],
    WithJsonSchema({"type": "string"}),
]


class GCSReader(BasePydanticReader, ResourcesReaderMixin, FileSystemReaderMixin):
    """
    A reader for Google Cloud Storage (GCS) files and directories.

    This class allows reading files from GCS, listing resources, and retrieving resource information.
    It supports authentication via service account keys and implements various reader mixins.

    Attributes:
        bucket (str): The name of the GCS bucket.
        key (Optional[str]): The specific file key to read. If None, the entire bucket is parsed.
        prefix (Optional[str]): The prefix to filter by when iterating through the bucket.
        recursive (bool): Whether to recursively search in subdirectories.
        file_extractor (Optional[Dict[str, Union[str, BaseReader]]]): Custom file extractors.
        required_exts (Optional[List[str]]): List of required file extensions.
        filename_as_id (bool): Whether to use the filename as the document ID.
        num_files_limit (Optional[int]): Maximum number of files to read.
        file_metadata (Optional[Callable[[str], Dict]]): Function to extract metadata from filenames.
        service_account_key (Optional[Dict[str, str]]): Service account key as a dictionary.
        service_account_key_json (Optional[str]): Service account key as a JSON string.
        service_account_key_path (Optional[str]): Path to the service account key file.
    """

    is_remote: bool = True

    bucket: str
    key: Optional[str] = None
    prefix: Optional[str] = ""
    recursive: bool = True
    file_extractor: Optional[Dict[str, Union[str, BaseReader]]] = Field(
        default=None, exclude=True
    )
    required_exts: Optional[List[str]] = None
    filename_as_id: bool = True
    num_files_limit: Optional[int] = None
    file_metadata: Optional[FileMetadataCallable] = Field(default=None, exclude=True)
    service_account_key: Optional[Dict[str, str]] = None
    service_account_key_json: Optional[str] = None
    service_account_key_path: Optional[str] = None

    @classmethod
    def class_name(cls) -> str:
        """Return the name of the class."""
        return "GCSReader"

    def _get_gcsfs(self):
        """
        Create and return a GCSFileSystem object.

        This method handles authentication using the provided service account credentials.

        Returns:
            GCSFileSystem: An authenticated GCSFileSystem object.

        Raises:
            ValueError: If no valid authentication method is provided.
            DefaultCredentialsError: If there's an issue with the provided credentials.
        """
        from gcsfs import GCSFileSystem

        try:
            if self.service_account_key is not None:
                creds = service_account.Credentials.from_service_account_info(
                    self.service_account_key, scopes=SCOPES
                )
            elif self.service_account_key_json is not None:
                creds = service_account.Credentials.from_service_account_info(
                    json.loads(self.service_account_key_json), scopes=SCOPES
                )
            elif self.service_account_key_path is not None:
                creds = service_account.Credentials.from_service_account_file(
                    self.service_account_key_path, scopes=SCOPES
                )
            else:
                logger.warning(
                    "No explicit credentials provided. Falling back to default credentials."
                )
                creds = None  # This will use default credentials

            return GCSFileSystem(token=creds)
        except DefaultCredentialsError as e:
            logger.error(f"Failed to authenticate with GCS: {e!s}")
            raise

    def _get_simple_directory_reader(self) -> SimpleDirectoryReader:
        """
        Create and return a SimpleDirectoryReader for GCS.

        This method sets up a SimpleDirectoryReader with the appropriate GCS filesystem
        and other configured parameters.

        Returns:
            SimpleDirectoryReader: A configured SimpleDirectoryReader for GCS.
        """
        gcsfs = self._get_gcsfs()

        input_dir = self.bucket
        input_files = None

        if self.key:
            input_files = [f"{self.bucket}/{self.key}"]
        elif self.prefix:
            input_dir = f"{input_dir}/{self.prefix}"

        return SimpleDirectoryReader(
            input_dir=input_dir,
            input_files=input_files,
            recursive=self.recursive,
            file_extractor=self.file_extractor,
            required_exts=self.required_exts,
            filename_as_id=self.filename_as_id,
            num_files_limit=self.num_files_limit,
            file_metadata=self.file_metadata,
            fs=gcsfs,
        )

    def load_data(self) -> List[Document]:
        """
        Load data from the specified GCS bucket or file.

        Returns:
            List[Document]: A list of loaded documents.

        Raises:
            Exception: If there's an error loading the data.
        """
        try:
            logger.info(f"Loading data from GCS bucket: {self.bucket}")
            return self._get_simple_directory_reader().load_data()
        except Exception as e:
            logger.error(f"Error loading data from GCS: {e!s}")
            raise

    def list_resources(self, **kwargs) -> List[str]:
        """
        List resources in the specified GCS bucket or directory.

        Args:
            **kwargs: Additional arguments to pass to the underlying list_resources method.

        Returns:
            List[str]: A list of resource identifiers.

        Raises:
            Exception: If there's an error listing the resources.
        """
        try:
            logger.info(f"Listing resources in GCS bucket: {self.bucket}")
            return self._get_simple_directory_reader().list_resources(**kwargs)
        except Exception as e:
            logger.error(f"Error listing resources in GCS: {e!s}")
            raise

    def get_resource_info(self, resource_id: str, **kwargs) -> Dict:
        """
        Get information about a specific GCS resource.

        Args:
            resource_id (str): The identifier of the resource.
            **kwargs: Additional arguments to pass to the underlying info method.

        Returns:
            Dict: A dictionary containing resource information.

        Raises:
            Exception: If there's an error retrieving the resource information.
        """
        try:
            logger.info(f"Getting info for resource: {resource_id}")
            gcsfs = self._get_gcsfs()
            info_result = gcsfs.info(resource_id)

            info_dict = {
                "file_path": info_result.get("name"),
                "file_size": info_result.get("size"),
                "last_modified_date": info_result.get("updated"),
                "content_hash": info_result.get("md5Hash"),
                "content_type": info_result.get("contentType"),
                "storage_class": info_result.get("storageClass"),
                "etag": info_result.get("etag"),
                "generation": info_result.get("generation"),
                "created_date": info_result.get("timeCreated"),
            }

            # Convert datetime objects to ISO format strings
            for key in ["last_modified_date", "created_date"]:
                if isinstance(info_dict.get(key), datetime):
                    info_dict[key] = info_dict[key].isoformat()

            return {k: v for k, v in info_dict.items() if v is not None}
        except Exception as e:
            logger.error(f"Error getting resource info from GCS: {e!s}")
            raise

    def load_resource(self, resource_id: str, **kwargs) -> List[Document]:
        """
        Load a specific resource from GCS.

        Args:
            resource_id (str): The identifier of the resource to load.
            **kwargs: Additional arguments to pass to the underlying load_resource method.

        Returns:
            List[Document]: A list containing the loaded document.

        Raises:
            Exception: If there's an error loading the resource.
        """
        try:
            logger.info(f"Loading resource: {resource_id}")
            return self._get_simple_directory_reader().load_resource(
                resource_id, **kwargs
            )
        except Exception as e:
            logger.error(f"Error loading resource from GCS: {e!s}")
            raise

    def read_file_content(self, resource_id: str, **kwargs) -> bytes:
        """
        Read the content of a specific file from GCS.

        Args:
            resource_id (str): The resource ID to read.
            **kwargs: Additional arguments to pass to the underlying read_file_content method.

        Returns:
            bytes: The content of the file.

        Raises:
            Exception: If there's an error reading the file content.
        """
        try:
            logger.info(f"Reading file content: {resource_id}")
            return self._get_simple_directory_reader().read_file_content(
                Path(resource_id), **kwargs
            )
        except Exception as e:
            logger.error(f"Error reading file content from GCS: {e!s}")
            raise

    def get_file_name(self, resource_id: str) -> str:
        return Path(resource_id).name

    def get_file_path(self, resource_id: str) -> str:
        return resource_id
