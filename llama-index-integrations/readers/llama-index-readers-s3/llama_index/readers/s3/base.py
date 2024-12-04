"""
S3 file and directory reader.

A loader that fetches a file or iterates through a directory on AWS S3.

"""

import warnings
from typing import Callable, Dict, List, Optional
from typing_extensions import Annotated, Unpack
from datetime import datetime, timezone
from pathlib import Path

from llama_index.core.readers import (
    SimpleDirectoryReader,
    FileSystemReaderMixin,
    DirectoryReaderArgs,
)
from llama_index.core.readers.base import (
    BasePydanticReader,
    ResourcesReaderMixin,
)
from llama_index.core.schema import Document
from llama_index.core.bridge.pydantic import WithJsonSchema


FileMetadataCallable = Annotated[
    Callable[[str], Dict],
    WithJsonSchema({"type": "string"}),
]


class S3Reader(BasePydanticReader, ResourcesReaderMixin, FileSystemReaderMixin):
    """
    General reader for any S3 file or directory.

    If key is not set, the entire bucket (filtered by prefix) is parsed.

    Args:
    bucket (str): the name of your S3 bucket
    key (Optional[str]): the name of the specific file. If none is provided,
        this loader will iterate through the entire bucket.
    prefix (Optional[str]): the prefix to filter by in the case that the loader
        iterates through the entire bucket. Defaults to empty string.
    aws_access_id (Optional[str]): provide AWS access key directly.
    aws_access_secret (Optional[str]): provide AWS access key directly.
    s3_endpoint_url (Optional[str]): provide S3 endpoint URL directly.
    **kwargs: (DirectoryReaderArgs): Additional arguments to pass to the directory reader.
    """

    is_remote: bool = True

    bucket: str
    key: Optional[str] = None
    prefix: Optional[str] = ""
    aws_access_id: Optional[str] = None
    aws_access_secret: Optional[str] = None
    aws_session_token: Optional[str] = None
    s3_endpoint_url: Optional[str] = None
    custom_reader_path: Optional[str] = None
    invalidate_s3fs_cache: bool = True

    def __init__(
        self,
        bucket: str,
        key: Optional[str] = None,
        prefix: Optional[str] = "",
        aws_access_id: Optional[str] = None,
        aws_access_secret: Optional[str] = None,
        aws_session_token: Optional[str] = None,
        s3_endpoint_url: Optional[str] = None,
        custom_reader_path: Optional[str] = None,
        invalidate_s3fs_cache: bool = True,
        **kwargs: Unpack[DirectoryReaderArgs],
    ) -> None:
        super().__init__(
            bucket=bucket,
            key=key,
            prefix=prefix,
            aws_access_id=aws_access_id,
            aws_access_secret=aws_access_secret,
            aws_session_token=aws_session_token,
            s3_endpoint_url=s3_endpoint_url,
            custom_reader_path=custom_reader_path,
            invalidate_s3fs_cache=invalidate_s3fs_cache,
            **kwargs,
        )

    @classmethod
    def class_name(cls) -> str:
        return "S3Reader"

    def _get_s3fs(self):
        from s3fs import S3FileSystem

        s3fs = S3FileSystem(
            key=self.aws_access_id,
            endpoint_url=self.s3_endpoint_url,
            secret=self.aws_access_secret,
            token=self.aws_session_token,
        )
        if self.invalidate_s3fs_cache:
            s3fs.invalidate_cache()

        return s3fs

    def _get_simple_directory_reader(self) -> SimpleDirectoryReader:
        # we don't want to keep the reader as a field in the class to keep it serializable
        s3fs = self._get_s3fs()

        input_dir = self.bucket
        input_files = None

        if self.key:
            input_files = [f"{self.bucket}/{self.key}"]
        elif self.prefix:
            input_dir = f"{input_dir}/{self.prefix}"

        return SimpleDirectoryReader(
            input_dir=input_dir, input_files=input_files, fs=s3fs, **self.model_dump()
        )

    def _load_s3_files_as_docs(self) -> List[Document]:
        """Load file(s) from S3."""
        loader = self._get_simple_directory_reader()
        return loader.load_data()

    async def _aload_s3_files_as_docs(self) -> List[Document]:
        """Asynchronously load file(s) from S3."""
        loader = self._get_simple_directory_reader()
        return await loader.aload_data()

    def _adjust_documents(self, documents: List[Document]) -> List[Document]:
        for doc in documents:
            if self.s3_endpoint_url:
                doc.id_ = self.s3_endpoint_url + "_" + doc.id_
            else:
                doc.id_ = "s3_" + doc.id_
        return documents

    def load_data(self, custom_temp_subdir: str = None) -> List[Document]:
        """
        Load the file(s) from S3.

        Args:
            custom_temp_subdir (str, optional): This parameter is deprecated and unused. Defaults to None.

        Returns:
            List[Document]: A list of documents loaded from S3.
        """
        if custom_temp_subdir is not None:
            warnings.warn(
                "The `custom_temp_subdir` parameter is deprecated and unused. Please remove it from your code.",
                DeprecationWarning,
            )

        documents = self._load_s3_files_as_docs()
        return self._adjust_documents(documents)

    async def aload_data(self, custom_temp_subdir: str = None) -> List[Document]:
        """
        Asynchronously load the file(s) from S3.

        Args:
            custom_temp_subdir (str, optional): This parameter is deprecated and unused. Defaults to None.

        Returns:
            List[Document]: A list of documents loaded from S3.
        """
        if custom_temp_subdir is not None:
            warnings.warn(
                "The `custom_temp_subdir` parameter is deprecated and unused. Please remove it from your code.",
                DeprecationWarning,
            )

        documents = await self._aload_s3_files_as_docs()
        return self._adjust_documents(documents)

    def list_resources(self, **kwargs) -> List[str]:
        simple_directory_reader = self._get_simple_directory_reader()
        return simple_directory_reader.list_resources(**kwargs)

    def get_resource_info(self, resource_id: str, **kwargs) -> Dict:
        # can't use SimpleDirectoryReader.get_resource_info because it lacks some fields
        fs = self._get_s3fs()
        info_result = fs.info(resource_id)

        last_modified_date = info_result.get("LastModified")
        if last_modified_date and isinstance(last_modified_date, datetime):
            last_modified_date = last_modified_date.astimezone(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )
        else:
            last_modified_date = None

        info_dict = {
            "file_path": str(resource_id),
            "file_size": info_result.get("size"),
            "last_modified_date": last_modified_date,
            "content_hash": info_result.get("ETag"),
        }

        # Ignore None values
        return {
            meta_key: meta_value
            for meta_key, meta_value in info_dict.items()
            if meta_value is not None
        }

    def load_resource(self, resource_id: str, **kwargs) -> List[Document]:
        simple_directory_reader = self._get_simple_directory_reader()
        docs = simple_directory_reader.load_resource(resource_id, **kwargs)
        return self._adjust_documents(docs)

    def read_file_content(self, input_file: Path, **kwargs) -> bytes:
        simple_directory_reader = self._get_simple_directory_reader()
        return simple_directory_reader.read_file_content(input_file, **kwargs)
