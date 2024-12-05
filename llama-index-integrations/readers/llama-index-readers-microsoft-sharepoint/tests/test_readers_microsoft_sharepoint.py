import os
import pytest
import tempfile

from llama_index.core.readers.base import BaseReader
from llama_index.readers.microsoft_sharepoint import SharePointReader

from unittest.mock import patch, MagicMock

test_client_id = "test_client_id"
test_client_secret = "test_client_secret"
test_tenant_id = "test_tenant_id"


def test_class():
    names_of_base_classes = [b.__name__ for b in SharePointReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes


def test_serialize():
    reader = SharePointReader(
        client_id=test_client_id,
        client_secret=test_client_secret,
        tenant_id=test_tenant_id,
    )

    schema = reader.schema()
    assert schema is not None
    assert len(schema) > 0
    assert "client_id" in schema["properties"]
    assert "client_secret" in schema["properties"]
    assert "tenant_id" in schema["properties"]

    json = reader.json(exclude_unset=True)

    new_reader = SharePointReader.parse_raw(json)
    assert new_reader.client_id == reader.client_id
    assert new_reader.client_secret == reader.client_secret
    assert new_reader.tenant_id == reader.tenant_id


@pytest.fixture()
def sharepoint_reader():
    sharepoint_reader = SharePointReader(
        client_id="dummy_client_id",
        client_secret="dummy_client_secret",
        tenant_id="dummy_tenant_id",
        sharepoint_site_name="dummy_site_name",
        sharepoint_folder_path="dummy_folder_path",
        drive_name="dummy_drive_name",
    )

    sharepoint_reader._drive_id_endpoint = (
        "https://graph.microsoft.com/v1.0/sites/dummy_site_id/drives"
    )
    sharepoint_reader._authorization_headers = {"Authorization": "Bearer dummy_token"}

    return sharepoint_reader


def mock_send_get_with_retry(url):
    mock_response = MagicMock()
    mock_response.status_code = 200

    if url == "https://graph.microsoft.com/v1.0/sites":
        # Mock response for site information endpoint
        mock_response.json.return_value = {
            "value": [{"id": "dummy_site_id", "name": "dummy_site_name"}]
        }
    elif url == "https://graph.microsoft.com/v1.0/sites/dummy_site_id/drives":
        # Mock response for drive information endpoint
        mock_response.json.return_value = {
            "value": [{"id": "dummy_drive_id", "name": "dummy_drive_name"}]
        }
    elif (
        url
        == "https://graph.microsoft.com/v1.0/sites/dummy_site_id/drives/dummy_drive_id/root:/dummy_folder_path"
    ):
        # Mock response for folder information endpoint
        mock_response.json.return_value = {"id": "dummy_folder_id"}
    elif (
        url
        == "https://graph.microsoft.com/v1.0/sites/dummy_site_id/drives/dummy_drive_id/items/dummy_folder_id/children"
    ):
        # Mock response for listing folder contents
        mock_response.json.return_value = {
            "value": [
                {"id": "file1_id", "name": "file1.txt", "file": {}},
                {"id": "file2_id", "name": "file2.txt", "file": {}},
            ]
        }
    elif (
        url
        == "https://graph.microsoft.com/v1.0/sites/dummy_site_id/drives/dummy_drive_id/items/file1_id/permissions"
    ):
        # Mock response for file1 permissions
        mock_response.json.return_value = {
            "value": [
                {"grantedToV2": {"user": {"id": "user1", "displayName": "User One"}}}
            ]
        }
    elif (
        url
        == "https://graph.microsoft.com/v1.0/sites/dummy_site_id/drives/dummy_drive_id/items/file2_id/permissions"
    ):
        # Mock response for file2 permissions
        mock_response.json.return_value = {
            "value": [
                {"grantedToV2": {"user": {"id": "user2", "displayName": "User Two"}}}
            ]
        }
    elif (
        url
        == "https://graph.microsoft.com/v1.0/sites/dummy_site_id/drives/dummy_drive_id/items"
    ):
        # Mock response for getting item details by path
        if "file1.txt" in url:
            mock_response.json.return_value = {
                "id": "file1_id",
                "name": "file1.txt",
                "@microsoft.graph.downloadUrl": "http://dummyurl/file1.txt",
            }
        elif "file2.txt" in url:
            mock_response.json.return_value = {
                "id": "file2_id",
                "name": "file2.txt",
                "@microsoft.graph.downloadUrl": "http://dummyurl/file2.txt",
            }
    else:
        mock_response.status_code = 404
        mock_response.json.return_value = {"error": {"message": "Not Found"}}

    return mock_response


@pytest.fixture(autouse=True)
def mock_sharepoint_api_calls():
    with patch.object(
        SharePointReader, "_get_access_token", return_value="dummy_token"
    ), patch.object(
        SharePointReader, "_get_site_id_with_host_name", return_value="dummy_site_id"
    ), patch.object(
        SharePointReader, "_get_sharepoint_folder_id", return_value="dummy_folder_id"
    ), patch.object(
        SharePointReader, "_get_drive_id", return_value="dummy_drive_id"
    ), patch.object(
        SharePointReader, "_send_get_with_retry", side_effect=mock_send_get_with_retry
    ):
        yield


def test_list_resources(sharepoint_reader):
    # Setting the _drive_id_endpoint manually to avoid the AttributeError
    file_paths = sharepoint_reader.list_resources(
        sharepoint_site_name="dummy_site_name",
        sharepoint_folder_path="dummy_folder_path",
        recursive=False,
    )
    assert len(file_paths) == 2
    assert file_paths[0] == "dummy_site_name/dummy_folder_path/file1.txt"
    assert file_paths[1] == "dummy_site_name/dummy_folder_path/file2.txt"


def test_load_documents_with_metadata(sharepoint_reader):
    # Setting the _drive_id_endpoint manually to avoid the AttributeError
    sharepoint_reader._drive_id_endpoint = (
        "https://graph.microsoft.com/v1.0/sites/dummy_site_id/drives/dummy_drive_id"
    )

    with tempfile.TemporaryDirectory() as tmpdirname:
        # Create mock files in the temporary directory
        file1_path = os.path.join(tmpdirname, "file1.txt")
        file2_path = os.path.join(tmpdirname, "file2.txt")
        with open(file1_path, "w") as f:
            f.write("File 1 content")
        with open(file2_path, "w") as f:
            f.write("File 2 content")

        # Prepare metadata for the mock files
        files_metadata = {
            file1_path: {
                "file_id": "file1_id",
                "file_name": "file1.txt",
                "url": "http://dummyurl/file1.txt",
                "file_path": file1_path,
            },
            file2_path: {
                "file_id": "file2_id",
                "file_name": "file2.txt",
                "url": "http://dummyurl/file2.txt",
                "file_path": file2_path,
            },
        }

        documents = sharepoint_reader._load_documents_with_metadata(
            files_metadata, tmpdirname, recursive=False
        )

        assert documents is not None
        assert len(documents) == 2
        assert documents[0].metadata["file_name"] == "file1.txt"
        assert documents[1].metadata["file_name"] == "file2.txt"
        assert documents[0].text == "File 1 content"
        assert documents[1].text == "File 2 content"


def test_required_exts():
    sharepoint_reader = SharePointReader(
        client_id="dummy_client_id",
        client_secret="dummy_client_secret",
        tenant_id="dummy_tenant_id",
        sharepoint_site_name="dummy_site_name",
        sharepoint_folder_path="dummy_folder_path",
        drive_name="dummy_drive_name",
        required_exts=[".md"],
    )

    with tempfile.TemporaryDirectory() as tmpdirname:
        readme_file_path = os.path.join(tmpdirname, "readme.md")
        audio_file_path = os.path.join(tmpdirname, "audio.aac")
        with open(readme_file_path, "w") as f:
            f.write("Readme content")
        with open(audio_file_path, "wb") as f:
            f.write(bytearray([0xFF, 0xF1, 0x50, 0x80, 0x00, 0x7F, 0xFC, 0x00]))

        file_metadata = {
            readme_file_path: {
                "file_id": "readme_file_id",
                "file_name": "readme.md",
                "url": "http://dummyurl/readme.md",
                "file_path": readme_file_path,
            },
            audio_file_path: {
                "file_id": "audio_file_id",
                "file_name": "audio.aac",
                "url": "http://dummyurl/audio.aac",
                "file_path": audio_file_path,
            },
        }

        documents = sharepoint_reader._load_documents_with_metadata(
            file_metadata, tmpdirname, recursive=False
        )

        assert documents is not None
        assert len(documents) == 1
        assert documents[0].metadata["file_name"] == "readme.md"
        assert documents[0].text == "Readme content"
