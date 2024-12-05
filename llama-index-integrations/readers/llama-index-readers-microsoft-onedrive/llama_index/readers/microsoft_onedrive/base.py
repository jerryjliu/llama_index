"""OneDrive files reader."""

import logging
import os
import tempfile
import time
from typing import Any, Dict, List, Optional, Union

import requests
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.core.readers.base import BaseReader, BasePydanticReader
from llama_index.core.schema import Document
from llama_index.core.bridge.pydantic import PrivateAttr, Field, BaseModel
from llama_index.core.readers import FileSystemReaderMixin
from llama_index.core.readers.base import (
    BaseReader,
    BasePydanticReader,
    ResourcesReaderMixin,
)

logger = logging.getLogger(__name__)

# Scope for reading and downloading OneDrive files
SCOPES = ["Files.Read.All"]
CLIENTCREDENTIALSCOPES = ["https://graph.microsoft.com/.default"]


class _OneDriveResourcePayload(BaseModel):
    resource_info: Dict[str, Any]
    downloaded_file_path: Optional[str]


class OneDriveReader(BasePydanticReader, ResourcesReaderMixin, FileSystemReaderMixin):
    """
    Microsoft OneDrive reader.

    Initializes a new instance of the OneDriveReader.

    :param client_id: The Application (client) ID for the app registered in the Azure Entra (formerly Azure Active directory) portal with MS Graph permission "Files.Read.All".
    :param tenant_id: The Directory (tenant) ID of the Azure Active Directory (AAD) tenant the app is registered with.
                      Defaults to "consumers" for multi-tenant applications and OneDrive personal.
    :param client_secret: The Application Secret for the app registered in the Azure portal.
                          If provided, the MSAL client credential flow will be used for authentication (ConfidentialClientApplication).
                          If not provided, interactive authentication will be used (Not recommended for CI/CD or scenarios where manual interaction for authentication is not feasible).
                          Required for App authentication.
    :param userprinciplename: The user principal name (normally organization provided email) whose OneDrive will be accessed. Required for App authentication. Will be used if the
                              parameter is not provided when calling load_data().
    :param folder_id: The folder ID of the folder to fetch from OneDrive. Will be used if the parameter is not provided when calling load_data().
    :param file_ids: A list of file IDs of files to fetch from OneDrive. Will be used if the parameter is not provided when calling load_data().
    :param folder_path (str, optional): The relative path of the OneDrive folder to download. If provided, files within the folder are downloaded.  Will be used if the parameter is
                                        not provided when calling load_data().
    :param file_paths (List[str], optional): List of specific file paths to download. Will be used if the parameter is not provided when calling load_data().
    :param file_extractor (Optional[Dict[str, BaseReader]]): A mapping of file extension to a BaseReader class that specifies how to convert that file to text.
                                                             See `SimpleDirectoryReader` for more details.
    :param required_exts (Optional[List[str]]): List of required extensions. Default is None.


    For interactive authentication to work, a browser is used to authenticate, hence the registered application should have a redirect URI set to 'https://localhost'
    for mobile and native applications.
    """

    client_id: str
    client_secret: Optional[str] = None
    tenant_id: Optional[str] = None
    userprincipalname: Optional[str] = None
    folder_id: Optional[str] = None
    file_ids: Optional[List[str]] = None
    folder_path: Optional[str] = None
    file_paths: Optional[List[str]] = None
    required_exts: Optional[List[str]] = None
    file_extractor: Optional[Dict[str, Union[str, BaseReader]]] = Field(
        default=None, exclude=True
    )
    attach_permission_metadata: bool = False

    _is_interactive_auth = PrivateAttr(False)
    _authority = PrivateAttr()

    def __init__(
        self,
        client_id: str,
        client_secret: Optional[str] = None,
        tenant_id: Optional[str] = "consumers",
        userprincipalname: Optional[str] = None,
        folder_id: Optional[str] = None,
        file_ids: Optional[List[str]] = None,
        folder_path: Optional[str] = None,
        file_paths: Optional[List[str]] = None,
        file_extractor: Optional[Dict[str, Union[str, BaseReader]]] = None,
        attach_permission_metadata: bool = False,
        required_exts: Optional[List[str]] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            client_id=client_id,
            client_secret=client_secret,
            tenant_id=tenant_id,
            userprincipalname=userprincipalname,
            folder_id=folder_id,
            file_ids=file_ids,
            folder_path=folder_path,
            file_paths=file_paths,
            file_extractor=file_extractor,
            attach_permission_metadata=attach_permission_metadata,
            required_exts=required_exts,
            **kwargs,
        )
        self._is_interactive_auth = not client_secret
        self._authority = f"https://login.microsoftonline.com/{tenant_id}/"

    def _authenticate_with_msal(self) -> Any:
        """Authenticate with MSAL.

        For interactive authentication to work, a browser is used to authenticate, hence the registered application should have a redirect URI set to 'localhost'
        for mobile and native applications.
        """
        import msal

        result = None

        if self._is_interactive_auth:
            logger.debug("Starting user authentication...")
            app = msal.PublicClientApplication(
                self.client_id, authority=self._authority
            )

            # The acquire_token_interactive method will open the default web browser
            # for the interactive part of the OAuth2 flow. The registered application should have a redirect URI set to 'https://localhost'
            # under mobile and native applications.
            result = app.acquire_token_interactive(SCOPES)
        else:
            logger.debug("Starting app authentication...")
            app = msal.ConfidentialClientApplication(
                self.client_id,
                authority=self._authority,
                client_credential=self.client_secret,
            )

            result = app.acquire_token_for_client(scopes=CLIENTCREDENTIALSCOPES)

        if "access_token" in result:
            logger.debug("Authentication is successful...")
            return result["access_token"]
        else:
            logger.error(result.get("error"))
            logger.error(result.get("error_description"))
            logger.error(result.get("correlation_id"))
            raise Exception(result.get("error"))

    def _construct_endpoint(
        self,
        item_ref: str,
        isRelativePath: bool,
        isFile: bool,
        userprincipalname: Optional[str] = None,
    ) -> str:
        """
        Constructs the appropriate OneDrive API endpoint based on the provided parameters.

        Parameters:
            item_ref (str): The reference to the item; could be an item ID or a relative path.
            isRelativePath (bool): A boolean indicating whether the item_ref is a relative path.
            isFile (bool): A boolean indicating whether the target is a file.
            userprincipalname (str, optional): The user principal name; used if authentication is not interactive. Defaults to None.

        Returns:
            str: A string representing the constructed endpoint.
        """
        if not self._is_interactive_auth and not userprincipalname:
            raise Exception(
                "userprincipalname cannot be empty for App authentication. Provide the userprincipalname (usually email) of the user whose OneDrive will be accessed."
            )

        endpoint = "https://graph.microsoft.com/v1.0/"

        # Update the base endpoint based on the authentication method
        if self._is_interactive_auth:
            endpoint += "me/drive"
        else:
            endpoint += f"users/{userprincipalname}/drive"

        # Update the endpoint for relative paths or item IDs
        if isRelativePath:
            endpoint += f"/root:/{item_ref}"
        else:
            endpoint += f"/items/{item_ref}"

        # If the target is not a file, adjust the endpoint to retrieve children of a folder
        if not isFile:
            endpoint += ":/children" if isRelativePath else "/children"

        logger.info(f"API Endpoint determined: {endpoint}")
        return endpoint

    def _get_items_in_drive_with_maxretries(
        self,
        access_token: str,
        item_ref: Optional[str] = "root",
        max_retries: int = 3,
        userprincipalname: Optional[str] = None,
        isFile: bool = False,
        isRelativePath=False,
    ) -> Any:
        """
        Retrieves items from a drive using Microsoft Graph API.

        Parameters:
        access_token (str): Access token for API calls.
        item_ref (Optional[str]): Specific item ID/path or root for root folder.
        max_retries (int): Max number of retries on rate limit or server errors.
        userprincipalname: str value indicating the userprincipalname (usually organization-provided email) whose OneDrive will be accessed. Required for App authentication.
        isFile: bool value to indicate if to query file or folder
        isRelativePath: bool value to indicate if to query file or folder using relative path
        Returns:
        dict/None: JSON response or None after max retries.

        Raises:
        Exception: On non-retriable status code.
        """
        endpoint = self._construct_endpoint(
            item_ref, isRelativePath, isFile, userprincipalname
        )
        headers = {"Authorization": f"Bearer {access_token}"}
        retries = 0

        while retries < max_retries:
            response = requests.get(endpoint, headers=headers)
            if response.status_code == 200:
                return response.json()
            # Check for Ratelimit error, this can happen if you query endpoint recursively
            # very frequently for large amount of file
            elif response.status_code in (429, *range(500, 600)):
                logger.warning(
                    f"Retrying {retries+1} in {retries+1} secs. Status code: {response.status_code}"
                )
                retries += 1
                time.sleep(retries)  # Exponential back-off
            else:
                raise Exception(
                    f"API request to download {item_ref} failed with status code: {response.status_code}, message: {response.content}"
                )

        logger.error(f"Failed after {max_retries} attempts.")
        return None

    def _download_file_by_url(self, item: Dict[str, Any], local_dir: str) -> str:
        """
        Downloads a file from OneDrive using the provided item's download URL.

        Parameters:
        - item (Dict[str, str]): Dictionary containing file metadata and download URL.
        - local_dir (str): Local directory where the file should be saved.

        Returns:
        - str: The file path of the download file

        """
        # Extract download URL and filename from the provided item.
        file_download_url = item["@microsoft.graph.downloadUrl"]
        file_name = item["name"]

        # Download the file.
        file_data = requests.get(file_download_url)

        # Save the downloaded file to the specified local directory.
        file_path = os.path.join(local_dir, file_name)
        with open(file_path, "wb") as f:
            f.write(file_data.content)

        return file_path

    def _extract_metadata_for_file(
        self, item: Dict[str, Any], access_token: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Extracts metadata related to the file.

        Parameters:
        - item (Dict[str, str]): Dictionary containing file metadata.

        Returns:
        - Dict[str, str]: A dictionary containing the extracted metadata.
        """
        # Extract the required metadata for file.
        created_by = item.get("createdBy", {})
        modified_by = item.get("lastModifiedBy", {})
        parent_dir = item.get("parentReference", {}).get("path")
        file_name = item.get("name")
        file_path = file_name
        if parent_dir and file_name:
            file_path = os.path.join(parent_dir, file_name)
            file_path = file_path.replace("/drive/root:", "")

        permission = {}

        if self.attach_permission_metadata:
            permission = self._get_permissions_info(
                item, self.userprincipalname, access_token
            )

        return {
            "file_id": item.get("id"),
            "file_name": file_name,
            "file_size": item.get("size"),
            "file_path": file_path,
            "created_by_user": created_by.get("user", {}).get("displayName"),
            "created_by_app": created_by.get("application", {}).get("displayName"),
            "created_dateTime": item.get("createdDateTime"),
            "last_modified_by_user": modified_by.get("user", {}).get("displayName"),
            "last_modified_by_app": modified_by.get("application", {}).get(
                "displayName"
            ),
            "last_modified_datetime": item.get("lastModifiedDateTime"),
            **permission,
        }

    def _check_approved_mimetype_and_download_file(
        self,
        item: Dict[str, Any],
        local_dir: Optional[str] = None,
        mime_types: Optional[List[str]] = None,
        access_token: Optional[str] = None,
    ) -> _OneDriveResourcePayload:
        """
        Checks files based on MIME types and download the accepted files.

        :param item: dict, a dictionary representing a file item, must contain 'file' and 'mimeType' keys.
        :param local_dir: str, the local directory to download files to.
        :param mime_types: list, a list of accepted MIME types. If None or empty, all file types are accepted.
        :return: dict, a dictionary containing metadata of downloaded files.
        """
        resource_info = {}
        downloaded_file_path = None

        # Convert accepted MIME types to lowercase for case-insensitive comparison
        accepted_mimetypes = (
            [mimetype.lower() for mimetype in mime_types] if mime_types else ["*"]
        )

        # Check if the item's MIME type is among the accepted MIME types
        is_accepted_mimetype = (
            "*" in accepted_mimetypes
            or item["file"]["mimeType"].lower() in accepted_mimetypes
        )

        if is_accepted_mimetype:
            # It's a file with an accepted MIME type; download and extract metadata
            if local_dir:
                downloaded_file_path = self._download_file_by_url(
                    item, local_dir
                )  # Assuming this method is implemented
            resource_info = self._extract_metadata_for_file(
                item, access_token
            )  # Assuming this method is implemented
        else:
            # Log a debug message for files that are ignored due to an invalid MIME type
            logger.debug(
                f"Ignoring file '{item['name']}' as its MIME type does not match the accepted types."
            )

        return _OneDriveResourcePayload(
            resource_info=resource_info, downloaded_file_path=downloaded_file_path
        )

    def _connect_download_and_return_metadata(
        self,
        access_token: str,
        local_dir: Optional[str] = None,
        item_id: str = None,
        include_subfolders: bool = True,
        mime_types: Optional[List[str]] = None,
        userprincipalname: Optional[str] = None,
        isRelativePath=False,
    ) -> List[_OneDriveResourcePayload]:
        """
        Recursively download files from OneDrive, starting from the specified item_id or the root.

        Parameters:
        - access_token (str): Token for authorization.
        - local_dir (str, optional): Local directory to store downloaded files.
        - item_id (str, optional): ID of the specific item (folder/file) to start from. If None, starts from the root.
        - include_subfolders (bool, optional): Whether to include subfolders. Defaults to True.
        - mime_types(List[str], optional): the mimeTypes you want to allow e.g.: "application/pdf", default is None which loads all files
        - userprincipalname (str): The userprincipalname(normally organization provided email id) whose ondrive needs to be accessed. Mandatory for App authentication scenarios.
        - isRelativePath (bool): Value to indicate if to query file/folder using relative path

        Returns:
        - dict: Dictionary of file paths and their corresponding metadata.

        Raises:
        - Exception: If items can't be retrieved for the current item.
        """
        data = self._get_items_in_drive_with_maxretries(
            access_token,
            item_id,
            userprincipalname=userprincipalname,
            isRelativePath=isRelativePath,
        )

        if data:
            payloads: List[_OneDriveResourcePayload] = []
            for item in data["value"]:
                if (
                    "folder" in item and include_subfolders
                ):  # It's a folder; traverse if flag is set
                    subfolder_metadata = self._connect_download_and_return_metadata(
                        access_token,
                        local_dir=local_dir,
                        item_id=item["id"],
                        include_subfolders=include_subfolders,
                        mime_types=mime_types,
                        userprincipalname=userprincipalname,
                    )
                    payloads.extend(subfolder_metadata)  # Merge metadata

                elif "file" in item:
                    payload = self._check_approved_mimetype_and_download_file(
                        item,
                        local_dir=local_dir,
                        mime_types=mime_types,
                        access_token=access_token,
                    )
                    payloads.append(payload)

            return payloads

        # No data received; raise exception
        current_item = item_id if item_id else "RootFolder"
        raise Exception(f"Unable to retrieve items for: {current_item}")

    def _init_download_and_get_metadata(
        self,
        temp_dir: str,
        folder_id: Optional[str] = None,
        file_ids: Optional[List[str]] = None,
        folder_path: Optional[str] = None,
        file_paths: Optional[List[str]] = None,
        recursive: bool = False,
        mime_types: Optional[List[str]] = None,
        userprincipalname: Optional[str] = None,
    ) -> List[_OneDriveResourcePayload]:
        """
        Download files from OneDrive based on specified folder or file IDs/Paths.

        Parameters:
        - temp_dir (str): The temporary directory where files will be downloaded.
        - folder_id (str, optional): The ID of the OneDrive folder to download. If provided, files within the folder are downloaded.
        - file_ids (List[str], optional): List of specific file IDs to download.
        - folder_path (str, optional): The relative path of the OneDrive folder to download. If provided, files within the folder are downloaded.
        - file_paths (List[str], optional): List of specific file paths to download.
        - recursive (bool): Flag indicating whether to download files from subfolders if a folder_id is provided.
        - mime_types(List[str], optional): the mimeTypes you want to allow e.g.: "application/pdf", default is None which loads all files
        - userprincipalname (str): The userprincipalname (normally organization-provided email) whose OneDrive will be accessed. Required for App authentication.

        """
        access_token = self._authenticate_with_msal()
        is_download_from_root = True
        payloads: List[_OneDriveResourcePayload] = []
        # If a folder_id is provided, download files from the folder
        if folder_id:
            is_download_from_root = False
            _payloads = self._connect_download_and_return_metadata(
                access_token,
                local_dir=temp_dir,
                item_id=folder_id,
                include_subfolders=recursive,
                mime_types=mime_types,
                userprincipalname=userprincipalname,
            )
            payloads.extend(_payloads)

        # Download files using the provided file IDs
        if file_ids:
            is_download_from_root = False
            for file_id in file_ids or []:
                item = self._get_items_in_drive_with_maxretries(
                    access_token,
                    file_id,
                    userprincipalname=userprincipalname,
                    isFile=True,
                )
                payload = self._check_approved_mimetype_and_download_file(
                    item,
                    local_dir=temp_dir,
                    mime_types=mime_types,
                    access_token=access_token,
                )
                payloads.append(payload)

        # If a folder_path is provided, download files from the folder
        if folder_path:
            is_download_from_root = False
            payload = self._connect_download_and_return_metadata(
                access_token,
                local_dir=temp_dir,
                item_id=folder_path,
                include_subfolders=recursive,
                mime_types=mime_types,
                userprincipalname=userprincipalname,
                isRelativePath=True,
            )
            payloads.extend(payload)

        # Download files using the provided file paths
        if file_paths:
            is_download_from_root = False
            for file_path in file_paths or []:
                item = self._get_items_in_drive_with_maxretries(
                    access_token,
                    file_path,
                    userprincipalname=userprincipalname,
                    isFile=True,
                    isRelativePath=True,
                )
                payload = self._check_approved_mimetype_and_download_file(
                    item,
                    local_dir=temp_dir,
                    mime_types=mime_types,
                    access_token=access_token,
                )
                payloads.append(payload)

        if is_download_from_root:
            # download files from root folder
            payload = self._connect_download_and_return_metadata(
                access_token,
                local_dir=temp_dir,
                item_id="root",
                include_subfolders=recursive,
                mime_types=mime_types,
                userprincipalname=userprincipalname,
            )
            payloads.extend(payload)

        return payloads

    def _load_documents_with_metadata(
        self,
        payloads: List[_OneDriveResourcePayload],
        directory: str,
        recursive: bool = True,
    ) -> List[Document]:
        """
        Load documents from a specified directory using the SimpleDirectoryReader
        and associate them with their respective metadata.

        Parameters:
        - payloads (List[_OneDriveResourcePayload]): List of payloads containing metadata and downloaded file paths.
        - directory (str): The directory from which to load the documents.
        - recursive (bool, optional): Whether to perform a recursive search through the directory. Defaults to True.

        Returns:
        - List[Document]: Loaded documents from the specified directory with associated metadata.
        """
        file_name_to_metadata = {
            payload.downloaded_file_path: payload.resource_info for payload in payloads
        }

        def get_metadata(filename: str) -> Any:
            return file_name_to_metadata[filename]

        simple_loader = SimpleDirectoryReader(
            directory,
            file_extractor=self.file_extractor,
            required_exts=self.required_exts,
            file_metadata=get_metadata,
            recursive=recursive,
        )
        return simple_loader.load_data()

    def _get_downloaded_files_metadata(
        self,
        folder_id: Optional[str] = None,
        file_ids: Optional[List[str]] = None,
        folder_path: Optional[str] = None,
        file_paths: Optional[List[str]] = None,
        mime_types: Optional[List[str]] = None,
        recursive: bool = True,
        userprincipalname: Optional[str] = None,
        temp_dir: Optional[str] = None,
    ) -> List[_OneDriveResourcePayload]:
        # If arguments are not provided to load_data(), initialize them from the object's attributes
        if not userprincipalname:
            userprincipalname = self.userprincipalname

        if not folder_id:
            folder_id = self.folder_id

        if not file_ids:
            file_ids = self.file_ids

        if not folder_path:
            folder_path = self.folder_path

        if not file_paths:
            file_paths = self.file_paths

        return self._init_download_and_get_metadata(
            temp_dir=temp_dir,
            folder_id=folder_id,
            file_ids=file_ids,
            folder_path=folder_path,
            file_paths=file_paths,
            recursive=recursive,
            mime_types=mime_types,
            userprincipalname=userprincipalname,
        )

    def load_data(
        self,
        folder_id: Optional[str] = None,
        file_ids: Optional[List[str]] = None,
        folder_path: Optional[str] = None,
        file_paths: Optional[List[str]] = None,
        mime_types: Optional[List[str]] = None,
        recursive: bool = True,
        userprincipalname: Optional[str] = None,
    ) -> List[Document]:
        """Load data from the folder id / file ids, f both are not provided download from the root.

        Args:
            folder_id (str, optional): folder id of the folder in OneDrive.
            file_ids (List[str], optional): file ids of the files in OneDrive.
            folder_path (str, optional): The relative path of the OneDrive folder to download. If provided, files within the folder are downloaded.
            file_paths (List[str], optional): List of specific file paths to download.
            mime_types: the mimeTypes you want to allow e.g.: "application/pdf", default is none, which loads all files found
            recursive: boolean value to traverse and read subfolder, default is True
            userprincipalname: str value indicating the userprincipalname (normally organization-provided email) whose OneDrive will be accessed. Required for App authentication scenarios.


        Returns:
            List[Document]: A list of documents.
        """
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                payloads = self._get_downloaded_files_metadata(
                    folder_id=folder_id,
                    file_ids=file_ids,
                    folder_path=folder_path,
                    file_paths=file_paths,
                    mime_types=mime_types,
                    recursive=recursive,
                    userprincipalname=userprincipalname,
                    temp_dir=temp_dir,
                )
                logger.debug("Downloaded %d files from OneDriveReader", len(payloads))
                return self._load_documents_with_metadata(
                    payloads, temp_dir, recursive=recursive
                )
        except Exception as e:
            logger.error(
                f"An error occurred while loading the data: {e}", exc_info=True
            )

    def get_permission_info(self, resource_id: str, *args: Any, **kwargs: Any) -> Dict:
        payloads = self._get_downloaded_files_metadata(
            file_paths=[resource_id], *args, **kwargs
        )

        item = next(
            payload.resource_info
            for payload in payloads
            if payload.resource_info["file_path"] == resource_id
        )

        access_token = self._authenticate_with_msal()

        return self._get_permissions_info(item, self.userprincipalname, access_token)

    def _get_permissions_info(
        self, item: Dict[str, Any], userprincipalname: str, access_token: str
    ) -> Dict[str, Any]:
        """
        Extracts the permissions information for the file in OneDrive.

        Args:
            item (Dict[str, Any]): Dictionary containing file metadata.

        Returns:
            Dict[str, Any]: A dictionary containing the extracted permissions information.
        """
        item_id = item.get("id")

        if self._is_interactive_auth:
            permissions_info_endpoint = (
                f"https://graph.microsoft.com/v1.0/me/drive/items/{item_id}/permissions"
            )
        else:
            permissions_info_endpoint = f"https://graph.microsoft.com/v1.0/users/{userprincipalname}/drive/items/{item_id}/permissions"

        headers = {"Authorization": f"Bearer {access_token}"}

        response = requests.get(url=permissions_info_endpoint, headers=headers)
        permissions = response.json()
        identity_sets = []

        for permission in permissions["value"]:
            # user type permissions
            granted_to = permission.get("grantedToV2", None)
            if granted_to:
                identity_sets.append(granted_to)

            # link type permissions
            granted_to_identities = permission.get("grantedToIdentitiesV2", [])
            for identity in granted_to_identities:
                identity_sets.append(identity)

        # Extract the identity information from each identity set
        # they can be 'application', 'device', 'user', 'group', 'siteUser' or 'siteGroup'
        # 'siteUser' and 'siteGroup' are site-specific, 'group' is for Microsoft 365 groups
        permissions_dict = {}

        for identity_set in identity_sets:
            for identity, identity_info in identity_set.items():
                # For OneDrive, we don't need to consider siteUser and siteGroup
                if identity in ["siteUser", "siteGroup"]:
                    continue

                id = identity_info.get("id")
                display_name = identity_info.get("displayName")
                ids_key = f"allowed_{identity}_ids"
                display_names_key = f"allowed_{identity}_display_names"

                if ids_key not in permissions_dict:
                    permissions_dict[ids_key] = []
                if display_names_key not in permissions_dict:
                    permissions_dict[display_names_key] = []

                permissions_dict[ids_key].append(id)
                permissions_dict[display_names_key].append(display_name)

        return permissions_dict

    def list_resources(
        self,
        folder_id: Optional[str] = None,
        file_ids: Optional[List[str]] = None,
        folder_path: Optional[str] = None,
        file_paths: Optional[List[str]] = None,
        mime_types: Optional[List[str]] = None,
        recursive: bool = True,
        userprincipalname: Optional[str] = None,
    ) -> List[str]:
        """
        List resources from the folder id / file ids, if both are not provided list from the root.

        Args:
            folder_id (str, optional): folder id of the folder in OneDrive.
            file_ids (List[str], optional): file ids of the files in OneDrive.
            folder_path (str, optional): The relative path of the OneDrive folder to download. If provided, files within the folder are downloaded.
            file_paths (List[str], optional): List of specific file paths to download.
            mime_types: the mimeTypes you want to allow e.g.: "application/pdf", default is none, which loads all files found
            recursive: boolean value to traverse and read subfolder, default is True
            userprincipalname: str value indicating the userprincipalname (normally organization-provided email) whose OneDrive will be accessed. Required for App authentication scenarios.

        Returns:
            List[str]: A list of resources.
        """
        try:
            payloads = self._get_downloaded_files_metadata(
                folder_id=folder_id,
                file_ids=file_ids,
                folder_path=folder_path,
                file_paths=file_paths,
                mime_types=mime_types,
                recursive=recursive,
                userprincipalname=userprincipalname,
            )
            return [str(payload.resource_info["file_path"]) for payload in payloads]
        except Exception as e:
            logger.error(
                f"An error occurred while listing resources: {e}", exc_info=True
            )
            raise

    async def alist_resources(
        self,
        folder_id: Optional[str] = None,
        file_ids: Optional[List[str]] = None,
        folder_path: Optional[str] = None,
        file_paths: Optional[List[str]] = None,
        mime_types: Optional[List[str]] = None,
        recursive: bool = True,
        userprincipalname: Optional[str] = None,
    ) -> List[str]:
        return self.list_resources(
            folder_id=folder_id,
            file_ids=file_ids,
            folder_path=folder_path,
            file_paths=file_paths,
            mime_types=mime_types,
            recursive=recursive,
            userprincipalname=userprincipalname,
        )

    def get_resource_info(self, resource_id: str, *args: Any, **kwargs: Any) -> Dict:
        payloads = self._get_downloaded_files_metadata(
            file_paths=[resource_id], *args, **kwargs
        )
        return next(
            payload.resource_info
            for payload in payloads
            if payload.resource_info["file_path"] == resource_id
        )

    async def aget_resource_info(
        self, resource_id: str, *args: Any, **kwargs: Any
    ) -> Dict:
        return self.get_resource_info(resource_id, *args, **kwargs)

    def load_resource(
        self, resource_id: str, *args: Any, **kwargs: Any
    ) -> List[Document]:
        return self.load_data(file_paths=[resource_id], *args, **kwargs)

    async def aload_resource(
        self, resource_id: str, *args: Any, **kwargs: Any
    ) -> List[Document]:
        return self.load_resource(resource_id, *args, **kwargs)

    def read_file_content(self, resource_id: str, **kwargs) -> bytes:
        with tempfile.TemporaryDirectory() as temp_dir:
            payloads = self._get_downloaded_files_metadata(
                file_paths=[resource_id], temp_dir=temp_dir, **kwargs
            )
            local_file_path = next(
                payloads.downloaded_file_path
                for payloads in payloads
                if payloads.resource_info["file_path"] == resource_id
            )
            if not local_file_path:
                raise ValueError("File was not downloaded successfully.")
            with open(local_file_path, "rb") as f:
                return f.read()

    async def aread_file_content(self, resource_id: str, **kwargs) -> bytes:
        return self.read_file_content(resource_id, **kwargs)
