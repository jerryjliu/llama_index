"""Google Drive files reader."""

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from google.auth.transport.requests import Request
from google.oauth2 import service_account
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.readers import SimpleDirectoryReader, FileSystemReaderMixin
from llama_index.core.readers.base import (
    BasePydanticReader,
    BaseReader,
    ResourcesReaderMixin,
)
from llama_index.core.schema import Document

logger = logging.getLogger(__name__)

# Scope for reading and downloading google drive files
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]


class GoogleDriveReader(
    BasePydanticReader, ResourcesReaderMixin, FileSystemReaderMixin
):
    """Google Drive Reader.

    Reads files from Google Drive. Credentials passed directly to the constructor
    will take precedence over those passed as file paths.

    Args:
        drive_id (Optional[str]): Drive id of the shared drive in google drive.
        folder_id (Optional[str]): Folder id of the folder in google drive.
        file_ids (Optional[str]): File ids of the files in google drive.
        query_string: A more generic query string to filter the documents, e.g. "name contains 'test'".
            It gives more flexibility to filter the documents. More info: https://developers.google.com/drive/api/v3/search-files
        is_cloud (Optional[bool]): Whether the reader is being used in
            a cloud environment. Will not save credentials to disk if so.
            Defaults to False.
        credentials_path (Optional[str]): Path to client config file.
            Defaults to None.
        token_path (Optional[str]): Path to authorized user info file. Defaults
            to None.
        service_account_key_path (Optional[str]): Path to service account key
            file. Defaults to None.
        client_config (Optional[dict]): Dictionary containing client config.
            Defaults to None.
        authorized_user_info (Optional[dict]): Dicstionary containing authorized
            user info. Defaults to None.
        service_account_key (Optional[dict]): Dictionary containing service
            account key. Defaults to None.
        file_extractor (Optional[Dict[str, BaseReader]]): A mapping of file
            extension to a BaseReader class that specifies how to convert that
            file to text. See `SimpleDirectoryReader` for more details.
    """

    drive_id: Optional[str] = None
    folder_id: Optional[str] = None
    file_ids: Optional[List[str]] = None
    query_string: Optional[str] = None
    client_config: Optional[dict] = None
    authorized_user_info: Optional[dict] = None
    service_account_key: Optional[dict] = None
    token_path: Optional[str] = None
    file_extractor: Optional[Dict[str, Union[str, BaseReader]]] = Field(
        default=None, exclude=True
    )

    _is_cloud: bool = PrivateAttr(default=False)
    _creds: Credentials = PrivateAttr()
    _mimetypes: dict = PrivateAttr()

    def __init__(
        self,
        drive_id: Optional[str] = None,
        folder_id: Optional[str] = None,
        file_ids: Optional[List[str]] = None,
        query_string: Optional[str] = None,
        is_cloud: Optional[bool] = False,
        credentials_path: str = "credentials.json",
        token_path: str = "token.json",
        service_account_key_path: str = "service_account_key.json",
        client_config: Optional[dict] = None,
        authorized_user_info: Optional[dict] = None,
        service_account_key: Optional[dict] = None,
        file_extractor: Optional[Dict[str, Union[str, BaseReader]]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize with parameters."""
        # Read the file contents so they can be serialized and stored.
        if client_config is None and os.path.isfile(credentials_path):
            with open(credentials_path, encoding="utf-8") as json_file:
                client_config = json.load(json_file)

        if authorized_user_info is None and os.path.isfile(token_path):
            with open(token_path, encoding="utf-8") as json_file:
                authorized_user_info = json.load(json_file)

        if service_account_key is None and os.path.isfile(service_account_key_path):
            with open(service_account_key_path, encoding="utf-8") as json_file:
                service_account_key = json.load(json_file)

        if (
            client_config is None
            and service_account_key is None
            and authorized_user_info is None
        ):
            raise ValueError(
                "Must specify `client_config` or `service_account_key` or `authorized_user_info`."
            )

        super().__init__(
            drive_id=drive_id,
            folder_id=folder_id,
            file_ids=file_ids,
            query_string=query_string,
            client_config=client_config,
            authorized_user_info=authorized_user_info,
            service_account_key=service_account_key,
            token_path=token_path,
            file_extractor=file_extractor,
            **kwargs,
        )

        self._creds = None
        self._is_cloud = is_cloud
        # Download Google Docs/Slides/Sheets as actual files
        # See https://developers.google.com/drive/v3/web/mime-types
        self._mimetypes = {
            "application/vnd.google-apps.document": {
                "mimetype": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "extension": ".docx",
            },
            "application/vnd.google-apps.spreadsheet": {
                "mimetype": (
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                ),
                "extension": ".xlsx",
            },
            "application/vnd.google-apps.presentation": {
                "mimetype": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
                "extension": ".pptx",
            },
        }

    @classmethod
    def class_name(cls) -> str:
        return "GoogleDriveReader"

    def _get_credentials(self) -> Tuple[Credentials]:
        """Authenticate with Google and save credentials.
        Download the service_account_key.json file with these instructions: https://cloud.google.com/iam/docs/keys-create-delete.

        IMPORTANT: Make sure to share the folders / files with the service account. Otherwise it will fail to read the docs

        Returns:
            credentials
        """
        # First, we need the Google API credentials for the app
        creds = None

        if self.authorized_user_info is not None:
            creds = Credentials.from_authorized_user_info(
                self.authorized_user_info, SCOPES
            )
        elif self.service_account_key is not None:
            return service_account.Credentials.from_service_account_info(
                self.service_account_key, scopes=SCOPES
            )

        # If there are no (valid) credentials available, let the user log in.
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_config(self.client_config, SCOPES)
                creds = flow.run_local_server(port=0)

            # Save the credentials for the next run
            if not self._is_cloud:
                with open(self.token_path, "w", encoding="utf-8") as token:
                    token.write(creds.to_json())

        return creds

    def _get_drive_link(self, file_id: str) -> str:
        return f"https://drive.google.com/file/d/{file_id}/view"

    def _get_relative_path(
        self, service, file_id: str, root_folder_id: Optional[str] = None
    ) -> str:
        """Get the relative path from root_folder_id to file_id."""
        try:
            # Get file details including parents
            file = (
                service.files()
                .get(fileId=file_id, supportsAllDrives=True, fields="name, parents")
                .execute()
            )

            path_parts = [file["name"]]

            if not root_folder_id:
                return file["name"]

            # Traverse up through parents until we reach root_folder_id or can't access anymore
            try:
                current_parent = file.get("parents", [None])[0]
                while current_parent:
                    # If we reach the root folder, stop
                    if current_parent == root_folder_id:
                        break

                    try:
                        parent = (
                            service.files()
                            .get(
                                fileId=current_parent,
                                supportsAllDrives=True,
                                fields="name, parents",
                            )
                            .execute()
                        )
                        path_parts.insert(0, parent["name"])
                        current_parent = parent.get("parents", [None])[0]
                    except Exception as e:
                        logger.debug(f"Stopped at parent {current_parent}: {e!s}")
                        break

            except Exception as e:
                logger.debug(f"Could not access parents for {file_id}: {e!s}")

            return "/".join(path_parts)

        except Exception as e:
            logger.warning(f"Could not get path for file {file_id}: {e}")
            return file["name"]

    def _get_fileids_meta(
        self,
        drive_id: Optional[str] = None,
        folder_id: Optional[str] = None,
        file_id: Optional[str] = None,
        mime_types: Optional[List[str]] = None,
        query_string: Optional[str] = None,
        current_path: Optional[str] = None,
    ) -> List[List[str]]:
        """Get file ids present in folder/ file id
        Args:
            drive_id: Drive id of the shared drive in google drive.
            folder_id: folder id of the folder in google drive.
            file_id: file id of the file in google drive
            mime_types: The mimeTypes you want to allow e.g.: "application/vnd.google-apps.document"
            query_string: A more generic query string to filter the documents, e.g. "name contains 'test'".

        Returns:
            metadata: List of metadata of filde ids.
        """
        from googleapiclient.discovery import build

        try:
            service = build("drive", "v3", credentials=self._creds)
            fileids_meta = []

            if folder_id and not file_id:
                try:
                    folder = (
                        service.files()
                        .get(fileId=folder_id, supportsAllDrives=True, fields="name")
                        .execute()
                    )
                    current_path = (
                        f"{current_path}/{folder['name']}"
                        if current_path
                        else folder["name"]
                    )
                except Exception as e:
                    logger.warning(f"Could not get folder name: {e}")

                folder_mime_type = "application/vnd.google-apps.folder"
                query = "('" + folder_id + "' in parents)"

                # Add mimeType filter to query
                if mime_types:
                    if folder_mime_type not in mime_types:
                        mime_types.append(folder_mime_type)  # keep the recursiveness
                    mime_query = " or ".join(
                        [f"mimeType='{mime_type}'" for mime_type in mime_types]
                    )
                    query += f" and ({mime_query})"

                # Add query string filter
                if query_string:
                    # to keep the recursiveness, we need to add folder_mime_type to the mime_types
                    query += (
                        f" and ((mimeType='{folder_mime_type}') or ({query_string}))"
                    )

                items = []
                page_token = ""
                # get files taking into account that the results are paginated
                while True:
                    if drive_id:
                        results = (
                            service.files()
                            .list(
                                q=query,
                                driveId=drive_id,
                                corpora="drive",
                                includeItemsFromAllDrives=True,
                                supportsAllDrives=True,
                                fields="*",
                                pageToken=page_token,
                            )
                            .execute()
                        )
                    else:
                        results = (
                            service.files()
                            .list(
                                q=query,
                                includeItemsFromAllDrives=True,
                                supportsAllDrives=True,
                                fields="*",
                                pageToken=page_token,
                            )
                            .execute()
                        )
                    items.extend(results.get("files", []))
                    page_token = results.get("nextPageToken", None)
                    if page_token is None:
                        break

                for item in items:
                    item_path = (
                        f"{current_path}/{item['name']}"
                        if current_path
                        else item["name"]
                    )

                    if item["mimeType"] == folder_mime_type:
                        if drive_id:
                            fileids_meta.extend(
                                self._get_fileids_meta(
                                    drive_id=drive_id,
                                    folder_id=item["id"],
                                    mime_types=mime_types,
                                    query_string=query_string,
                                    current_path=current_path,
                                )
                            )
                        else:
                            fileids_meta.extend(
                                self._get_fileids_meta(
                                    folder_id=item["id"],
                                    mime_types=mime_types,
                                    query_string=query_string,
                                    current_path=current_path,
                                )
                            )
                    else:
                        # Check if file doesn't belong to a Shared Drive. "owners" doesn't exist in a Shared Drive
                        is_shared_drive = "driveId" in item
                        author = (
                            item["owners"][0]["displayName"]
                            if not is_shared_drive
                            else "Shared Drive"
                        )
                        fileids_meta.append(
                            (
                                item["id"],
                                author,
                                item_path,
                                item["mimeType"],
                                item["createdTime"],
                                item["modifiedTime"],
                                self._get_drive_link(item["id"]),
                            )
                        )
            else:
                # Get the file details
                file = (
                    service.files()
                    .get(fileId=file_id, supportsAllDrives=True, fields="*")
                    .execute()
                )
                # Get metadata of the file
                is_shared_drive = "driveId" in file
                author = (
                    file["owners"][0]["displayName"]
                    if not is_shared_drive
                    else "Shared Drive"
                )

                # Get the full file path
                file_path = self._get_relative_path(
                    service, file_id, folder_id or self.folder_id
                )

                fileids_meta.append(
                    (
                        file["id"],
                        author,
                        file_path,
                        file["mimeType"],
                        file["createdTime"],
                        file["modifiedTime"],
                        self._get_drive_link(file["id"]),
                    )
                )
            return fileids_meta

        except Exception as e:
            logger.error(
                f"An error occurred while getting fileids metadata: {e}", exc_info=True
            )

    def _download_file(self, fileid: str, filename: str) -> str:
        """Download the file with fileid and filename
        Args:
            fileid: file id of the file in google drive
            filename: filename with which it will be downloaded
        Returns:
            The downloaded filename, which which may have a new extension.
        """
        from io import BytesIO

        from googleapiclient.discovery import build
        from googleapiclient.http import MediaIoBaseDownload

        try:
            # Get file details
            service = build("drive", "v3", credentials=self._creds)
            file = service.files().get(fileId=fileid, supportsAllDrives=True).execute()

            if file["mimeType"] in self._mimetypes:
                download_mimetype = self._mimetypes[file["mimeType"]]["mimetype"]
                download_extension = self._mimetypes[file["mimeType"]]["extension"]
                new_file_name = filename + download_extension

                # Download and convert file
                request = service.files().export_media(
                    fileId=fileid, mimeType=download_mimetype
                )
            else:
                new_file_name = filename

                # Download file without conversion
                request = service.files().get_media(fileId=fileid)

            # Download file data
            file_data = BytesIO()
            downloader = MediaIoBaseDownload(file_data, request)
            done = False

            while not done:
                status, done = downloader.next_chunk()

            # Save the downloaded file
            with open(new_file_name, "wb") as f:
                f.write(file_data.getvalue())

            return new_file_name
        except Exception as e:
            logger.error(
                f"An error occurred while downloading file: {e}", exc_info=True
            )

    def _load_data_fileids_meta(self, fileids_meta: List[List[str]]) -> List[Document]:
        """Load data from fileids metadata
        Args:
            fileids_meta: metadata of fileids in google drive.

        Returns:
            Lis[Document]: List of Document of data present in fileids.
        """
        try:
            with tempfile.TemporaryDirectory() as temp_dir:

                def get_metadata(filename):
                    return metadata[filename]

                temp_dir = Path(temp_dir)
                metadata = {}

                for fileid_meta in fileids_meta:
                    # Download files and name them with their fileid
                    fileid = fileid_meta[0]
                    filepath = os.path.join(temp_dir, fileid)
                    final_filepath = self._download_file(fileid, filepath)

                    # Add metadata of the file to metadata dictionary
                    metadata[final_filepath] = {
                        "file id": fileid_meta[0],
                        "author": fileid_meta[1],
                        "file path": fileid_meta[2],
                        "mime type": fileid_meta[3],
                        "created at": fileid_meta[4],
                        "modified at": fileid_meta[5],
                    }
                loader = SimpleDirectoryReader(
                    temp_dir,
                    file_extractor=self.file_extractor,
                    file_metadata=get_metadata,
                )
                documents = loader.load_data()
                for doc in documents:
                    doc.id_ = doc.metadata.get("file id", doc.id_)

            return documents
        except Exception as e:
            logger.error(
                f"An error occurred while loading data from fileids meta: {e}",
                exc_info=True,
            )

    def _load_from_file_ids(
        self,
        drive_id: Optional[str],
        file_ids: List[str],
        mime_types: Optional[List[str]],
        query_string: Optional[str],
    ) -> List[Document]:
        """Load data from file ids
        Args:
            file_ids: File ids of the files in google drive.
            mime_types: The mimeTypes you want to allow e.g.: "application/vnd.google-apps.document"
            query_string: List of query strings to filter the documents, e.g. "name contains 'test'".

        Returns:
            Document: List of Documents of text.
        """
        try:
            fileids_meta = []
            for file_id in file_ids:
                fileids_meta.extend(
                    self._get_fileids_meta(
                        drive_id=drive_id,
                        file_id=file_id,
                        mime_types=mime_types,
                        query_string=query_string,
                    )
                )
            return self._load_data_fileids_meta(fileids_meta)
        except Exception as e:
            logger.error(
                f"An error occurred while loading with fileid: {e}", exc_info=True
            )

    def _load_from_folder(
        self,
        drive_id: Optional[str],
        folder_id: str,
        mime_types: Optional[List[str]],
        query_string: Optional[str],
    ) -> List[Document]:
        """Load data from folder_id.

        Args:
            drive_id: Drive id of the shared drive in google drive.
            folder_id: folder id of the folder in google drive.
            mime_types: The mimeTypes you want to allow e.g.: "application/vnd.google-apps.document"
            query_string: A more generic query string to filter the documents, e.g. "name contains 'test'".

        Returns:
            Document: List of Documents of text.
        """
        try:
            fileids_meta = self._get_fileids_meta(
                drive_id=drive_id,
                folder_id=folder_id,
                mime_types=mime_types,
                query_string=query_string,
            )
            return self._load_data_fileids_meta(fileids_meta)
        except Exception as e:
            logger.error(
                f"An error occurred while loading from folder: {e}", exc_info=True
            )

    def load_data(
        self,
        drive_id: Optional[str] = None,
        folder_id: Optional[str] = None,
        file_ids: Optional[List[str]] = None,
        mime_types: Optional[List[str]] = None,  # Deprecated
        query_string: Optional[str] = None,
    ) -> List[Document]:
        """Load data from the folder id or file ids.

        Args:
            drive_id: Drive id of the shared drive in google drive.
            folder_id: Folder id of the folder in google drive.
            file_ids: File ids of the files in google drive.
            mime_types: The mimeTypes you want to allow e.g.: "application/vnd.google-apps.document"
            query_string: A more generic query string to filter the documents, e.g. "name contains 'test'".
                It gives more flexibility to filter the documents. More info: https://developers.google.com/drive/api/v3/search-files

        Returns:
            List[Document]: A list of documents.
        """
        self._creds = self._get_credentials()

        # If no arguments are provided to load_data, default to the object attributes
        if drive_id is None:
            drive_id = self.drive_id
        if folder_id is None:
            folder_id = self.folder_id
        if file_ids is None:
            file_ids = self.file_ids
        if query_string is None:
            query_string = self.query_string

        if folder_id:
            return self._load_from_folder(drive_id, folder_id, mime_types, query_string)
        elif file_ids:
            return self._load_from_file_ids(
                drive_id, file_ids, mime_types, query_string
            )
        else:
            logger.warning("Either 'folder_id' or 'file_ids' must be provided.")
            return []

    def list_resources(self, **kwargs) -> List[str]:
        """List resources in the specified Google Drive folder or files."""
        self._creds = self._get_credentials()

        drive_id = kwargs.get("drive_id", self.drive_id)
        folder_id = kwargs.get("folder_id", self.folder_id)
        file_ids = kwargs.get("file_ids", self.file_ids)
        query_string = kwargs.get("query_string", self.query_string)

        if folder_id:
            fileids_meta = self._get_fileids_meta(
                drive_id, folder_id, query_string=query_string
            )
        elif file_ids:
            fileids_meta = []
            for file_id in file_ids:
                fileids_meta.extend(
                    self._get_fileids_meta(
                        drive_id, file_id=file_id, query_string=query_string
                    )
                )
        else:
            raise ValueError("Either 'folder_id' or 'file_ids' must be provided.")

        return [meta[0] for meta in fileids_meta]  # Return list of file IDs

    def get_resource_info(self, resource_id: str, **kwargs) -> Dict:
        """Get information about a specific Google Drive resource."""
        self._creds = self._get_credentials()

        fileids_meta = self._get_fileids_meta(file_id=resource_id)
        if not fileids_meta:
            raise ValueError(f"Resource with ID {resource_id} not found.")

        meta = fileids_meta[0]
        return {
            "file_path": meta[2],
            "file_size": None,
            "last_modified_date": meta[5],
            "content_hash": None,
            "content_type": meta[3],
            "author": meta[1],
            "created_date": meta[4],
            "drive_link": meta[6],
        }

    def load_resource(self, resource_id: str, **kwargs) -> List[Document]:
        """Load a specific resource from Google Drive."""
        return self._load_from_file_ids(
            self.drive_id, [resource_id], None, self.query_string
        )

    def read_file_content(self, resource_id: str, **kwargs) -> bytes:
        """Read the content of a specific file from Google Drive."""
        self._creds = self._get_credentials()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file = os.path.join(temp_dir, "temp_file")
            downloaded_file = self._download_file(resource_id, temp_file)
            with open(downloaded_file, "rb") as file:
                return file.read()

    def get_file_name(self, resource_id: str) -> str:
        """Get the file name from the resource ID."""
        from googleapiclient.discovery import build

        self._creds = self._get_credentials()

        service = build("drive", "v3", credentials=self._creds)
        file = service.files().get(fileId=resource_id, supportsAllDrives=True).execute()
        return file["name"]

    def get_file_path(self, resource_id: str) -> str:
        """Get the file path from the resource ID."""
        from googleapiclient.discovery import build

        self._creds = self._get_credentials()

        service = build("drive", "v3", credentials=self._creds)

        return self._get_relative_path(
            service, file_id=resource_id, folder_id=self.folder_id
        )
