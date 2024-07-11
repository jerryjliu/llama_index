import logging
import tempfile
from typing import List, Optional, Dict, Any, Union

from llama_index.core.readers import SimpleDirectoryReader
from llama_index.core.readers.base import (
    BaseReader,
)
from llama_index.core.schema import Document
from llama_index.core.bridge.pydantic import Field

from llama_index.readers.box.BoxAPI.box_api import (
    _BoxResourcePayload,
    get_box_files_payload,
    get_box_folder_payload,
    download_file_by_id,
)

from box_sdk_gen import (
    BoxAPIError,
    BoxClient,
)

logger = logging.getLogger(__name__)


# TODO: Implement , ResourcesReaderMixin, FileSystemReaderMixin
class BoxReader(BaseReader):
    """
    A reader class for loading data from Box files.

    This class inherits from the BaseReader class and provides functionality
    to retrieve, download, and process data from Box files. It utilizes the
    provided BoxClient object to interact with the Box API and can optionally
    leverage a user-defined file extractor for more complex file formats.

    Attributes:
        _box_client (BoxClient): An authenticated Box client object used
            for interacting with the Box API.
        file_extractor (Optional[Dict[str, Union[str, BaseReader]]], optional):
            A dictionary mapping file extensions or mimetypes to either a string
            specifying a custom extractor function or another BaseReader subclass
            for handling specific file formats. Defaults to None.
    """

    _box_client: BoxClient
    file_extractor: Optional[Dict[str, Union[str, BaseReader]]] = Field(
        default=None, exclude=True
    )

    @classmethod
    def class_name(cls) -> str:
        return "BoxReader"

    def __init__(
        self,
        box_client: BoxClient,
        file_extractor: Optional[Dict[str, Union[str, BaseReader]]] = None,
    ):
        self._box_client = box_client
        self.file_extractor = file_extractor

    def load_data(
        self,
        folder_id: Optional[str] = None,
        file_ids: Optional[List[str]] = None,
        is_recursive: bool = False,
    ) -> List[Document]:
        """
        Loads data from Box files into a list of Document objects.

        This method retrieves Box files based on the provided parameters and
        processes them into a structured format using a SimpleDirectoryReader.

        Args:
            self (BoxDataHandler): An instance of the BoxDataHandler class.
            folder_id (Optional[str], optional): The ID of the Box folder to load
                data from. If provided, along with is_recursive set to True, retrieves
                data from sub-folders as well. Defaults to None.
            file_ids (Optional[List[str]], optional): A list of Box file IDs to
                load data from. If provided, folder_id is ignored. Defaults to None.
            is_recursive (bool, optional): If True and folder_id is provided, retrieves
                data from sub-folders within the specified folder. Defaults to False.

        Returns:
            List[Document]: A list of Document objects containing the processed data
                extracted from the Box files.

        Raises:
            BoxAPIError: If an error occurs while interacting with the Box API.
        """
        # Connect to Box
        try:
            me = self._box_client.users.get_user_me()
            logger.info(f"Connected to Box as user: {me.id} {me.name}({me.login})")
        except BoxAPIError as e:
            logger.error(
                f"An error occurred while connecting to Box: {e}", exc_info=True
            )
            raise

        # Get the file resources
        payloads: List[_BoxResourcePayload] = []
        if file_ids is not None:
            payloads.extend(
                get_box_files_payload(box_client=self._box_client, file_ids=file_ids)
            )
        elif folder_id is not None:
            payloads.extend(
                get_box_folder_payload(
                    box_client=self._box_client,
                    folder_id=folder_id,
                    is_recursive=is_recursive,
                )
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            payloads = self._download_files(payloads, temp_dir)

            file_name_to_metadata = {
                payload.downloaded_file_path: payload.resource_info.to_dict()
                for payload in payloads
            }

            def get_metadata(filename: str) -> Any:
                return file_name_to_metadata[filename]

            simple_loader = SimpleDirectoryReader(
                input_dir=temp_dir,
                file_metadata=get_metadata,
                file_extractor=self.file_extractor,
            )
            return simple_loader.load_data()

    def _download_files(
        self, payloads: List[_BoxResourcePayload], temp_dir: str
    ) -> List[_BoxResourcePayload]:
        """
        Downloads Box files and updates the corresponding payloads with local paths.

        This internal helper function iterates through the provided payloads,
        downloads each file referenced by the payload's resource_info attribute
        to the specified temporary directory, and updates the downloaded_file_path
        attribute of the payload with the local file path.

        Args:
            self (BoxReader): An instance of the BoxReader class.
            payloads (List[_BoxResourcePayload]): A list of _BoxResourcePayload objects
                containing information about Box files.
            temp_dir (str): The path to the temporary directory where the files will
                be downloaded.

        Returns:
            List[_BoxResourcePayload]: The updated list of _BoxResourcePayload objects
                with the downloaded_file_path attribute set for each payload.
        """
        for payload in payloads:
            file = payload.resource_info
            local_path = download_file_by_id(
                box_client=self._box_client, box_file=file, temp_dir=temp_dir
            )
            payload.downloaded_file_path = local_path
        return payloads