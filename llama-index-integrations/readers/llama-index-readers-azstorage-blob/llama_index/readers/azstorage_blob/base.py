"""
Azure Storage Blob file and directory reader.

A loader that fetches a file or iterates through a directory from Azure Storage Blob.

"""
import logging
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Dict, List, Optional, Union

from azure.storage.blob import ContainerClient

from llama_index.core.bridge.pydantic import Field
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.core.readers.base import BaseReader, BasePydanticReader
from llama_index.core.schema import Document
from llama_index.core.readers import SimpleDirectoryReader, FileSystemReaderMixin
from llama_index.core.readers.base import (
    BaseReader,
    BasePydanticReader,
    ResourcesReaderMixin,
)

logger = logging.getLogger(__name__)


class AzStorageBlobReader(
    BasePydanticReader, ResourcesReaderMixin, FileSystemReaderMixin
):
    """
    General reader for any Azure Storage Blob file or directory.

    Args:
        container_name (str): name of the container for the blob.
        blob (Optional[str]): name of the file to download. If none specified
            this loader will iterate through list of blobs in the container.
        name_starts_with (Optional[str]): filter the list of blobs to download
            to only those whose names begin with the specified string.
        include: (Union[str, List[str], None]): Specifies one or more additional
            datasets to include in the response. Options include: 'snapshots',
            'metadata', 'uncommittedblobs', 'copy', 'deleted',
            'deletedwithversions', 'tags', 'versions', 'immutabilitypolicy',
            'legalhold'.
        file_extractor (Optional[Dict[str, Union[str, BaseReader]]]): A mapping of file
            extension to a BaseReader class that specifies how to convert that file
            to text. See `SimpleDirectoryReader` for more details, or call this path ```llama_index.readers.file.base.DEFAULT_FILE_READER_CLS```.
        connection_string (str): A connection string which can be found under a storage account's "Access keys" security tab. This parameter
        can be used in place of both the account URL and credential.
        account_url (str): URI to the storage account, may include SAS token.
        credential (Union[str, Dict[str, str], AzureNamedKeyCredential, AzureSasCredential, TokenCredential, None] = None):
            The credentials with which to authenticate. This is optional if the account URL already has a SAS token.
    """

    container_name: str
    prefix: Optional[str] = ""
    blob: Optional[str] = None
    name_starts_with: Optional[str] = None
    include: Optional[Any] = None
    file_extractor: Optional[Dict[str, Union[str, BaseReader]]] = Field(
        default=None, exclude=True
    )
    connection_string: Optional[str] = None
    account_url: Optional[str] = None
    credential: Optional[Any] = None
    is_remote: bool = True

    # Not in use. As part of the TODO below. Is part of the kwargs.
    # self.preloaded_data_path = kwargs.get('preloaded_data_path', None)

    @classmethod
    def class_name(cls) -> str:
        return "AzStorageBlobReader"

    def _get_container_client(self):
        if self.connection_string:
            return ContainerClient.from_connection_string(
                conn_str=self.connection_string,
                container_name=self.container_name,
            )
        return ContainerClient(
            self.account_url, self.container_name, credential=self.credential
        )

    def _download_files_and_extract_metadata(self, temp_dir: str) -> Dict[str, Any]:
        """Download files from Azure Storage Blob and extract metadata."""
        container_client = self._get_container_client()
        blob_meta = {}

        if self.blob:
            blobs_list = [self.blob]
        else:
            blobs_list = container_client.list_blobs(
                self.name_starts_with, self.include
            )

        for obj in blobs_list:
            sanitized_file_name = obj.name.replace("/", "-") if not self.blob else obj
            download_file_path = os.path.join(temp_dir, sanitized_file_name)
            logger.info(f"Start download of {sanitized_file_name}")
            start_time = time.time()
            blob_client = container_client.get_blob_client(obj)
            stream = blob_client.download_blob()
            with open(file=download_file_path, mode="wb") as download_file:
                stream.readinto(download_file)
            blob_meta[sanitized_file_name] = blob_client.get_blob_properties()
            end_time = time.time()
            logger.debug(
                f"{sanitized_file_name} downloaded in {end_time - start_time} seconds."
            )

        return blob_meta

    def _extract_blob_metadata(self, file_metadata: Dict[str, Any]) -> Dict[str, Any]:
        meta: dict = file_metadata

        creation_time = meta.get("creation_time")
        creation_time = creation_time.strftime("%Y-%m-%d") if creation_time else None

        last_modified = meta.get("last_modified")
        last_modified = last_modified.strftime("%Y-%m-%d") if last_modified else None

        last_accessed_on = meta.get("last_accessed_on")
        last_accessed_on = (
            last_accessed_on.strftime("%Y-%m-%d") if last_accessed_on else None
        )

        extracted_meta = {
            "file_name": meta.get("name"),
            "file_type": meta.get("content_settings", {}).get("content_type"),
            "file_size": meta.get("size"),
            "creation_date": creation_time,
            "last_modified_date": last_modified,
            "last_accessed_date": last_accessed_on,
            "container": meta.get("container"),
        }

        extracted_meta.update(meta.get("metadata") or {})
        extracted_meta.update(meta.get("tags") or {})

        return extracted_meta

    def _load_documents_with_metadata(
        self, files_metadata: Dict[str, Any], temp_dir: str
    ) -> List[Document]:
        """Load documents from a directory and extract metadata."""

        def get_metadata(file_name: str) -> Dict[str, Any]:
            return files_metadata.get(file_name, {})

        loader = SimpleDirectoryReader(
            temp_dir, file_extractor=self.file_extractor, file_metadata=get_metadata
        )

        return loader.load_data()

    def list_resources(self, *args: Any, **kwargs: Any) -> List[str]:
        """List all the blobs in the container."""
        blobs_list = self._get_container_client().list_blobs(
            name_starts_with=self.name_starts_with, include=self.include
        )

        return [blob.name for blob in blobs_list]

    def get_resource_info(self, resource_id: str, **kwargs: Any) -> Dict:
        """Get metadata for a specific blob."""
        container_client = self._get_container_client()
        blob_client = container_client.get_blob_client(resource_id)
        blob_meta = blob_client.get_blob_properties()

        info_dict = {
            **self._extract_blob_metadata(blob_meta),
            "file_path": str(resource_id).replace(":", "/"),
        }

        return {
            meta_key: meta_value
            for meta_key, meta_value in info_dict.items()
            if meta_value is not None
        }

    def load_resource(self, resource_id: str, **kwargs: Any) -> List[Document]:
        try:
            container_client = self._get_container_client()
            blob_client = container_client.get_blob_client(resource_id)
            stream = blob_client.download_blob()
            with tempfile.TemporaryDirectory() as temp_dir:
                download_file_path = os.path.join(
                    temp_dir, resource_id.replace("/", "-")
                )
                with open(file=download_file_path, mode="wb") as download_file:
                    stream.readinto(download_file)
                return self._load_documents_with_metadata(
                    {resource_id: blob_client.get_blob_properties()}, temp_dir
                )
        except Exception as e:
            logger.error(
                f"Error loading resource {resource_id} from AzStorageBlob: {e}"
            )
            raise

    def read_file_content(self, resource_id: str, **kwargs: Any) -> bytes:
        """Read the content of a file from Azure Storage Blob."""
        container_client = self._get_container_client()
        blob_client = container_client.get_blob_client(resource_id)
        stream = blob_client.download_blob()
        return stream.readall()

    def load_data(self) -> List[Document]:
        """Load file(s) from Azure Storage Blob."""
        total_download_start_time = time.time()

        with tempfile.TemporaryDirectory() as temp_dir:
            files_metadata = self._download_files_and_extract_metadata(temp_dir)

            total_download_end_time = time.time()

            total_elapsed_time = math.ceil(
                total_download_end_time - total_download_start_time
            )

            logger.info(
                f"Downloading completed in approximately {total_elapsed_time // 60}min"
                f" {total_elapsed_time % 60}s."
            )

            logger.info("Document creation starting")

            return self._load_documents_with_metadata(files_metadata, temp_dir)

    def get_file_name(self, resource_id: str) -> str:
        return Path(resource_id).name

    def get_file_path(self, resource_id: str) -> str:
        return resource_id
