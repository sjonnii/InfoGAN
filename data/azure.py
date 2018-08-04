"""Module containing methods for downloading files from Azure blob storage."""

import os
from azure.common import AzureException
from azure.storage.blob import BlockBlobService


ACCOUNT_NAME = "cellexplorer"
#pylint: disable=line-too-long
ACCOUNT_KEY = "BwSNrGLsVlQlib/ztbSUGuDq+yW1hrwEV8k9TXWSCiWXd/BQskHSx9lGjRYwxGPT++J0Waj7hPRs8jCAfIYcaw=="
CONTAINER_NAME = "data"
DOWNLOADS_DIR = "downloads"


def may_download_file(name, downloads_dir=DOWNLOADS_DIR):
    """Downloads a file from Azure blob storage if needed.

    Args:
        name - the file to download

    Keyword Args:
        download_dir - path to the downloads directory [./downloads]

    Returns:
        path if the file exists (or can be obtained), otherwise None
    """
    path = os.path.join(downloads_dir, name)
    if os.path.exists(path):
        return path

    if not os.path.exists(downloads_dir):
        os.makedirs(downloads_dir)

    try:
        block_blob_service = BlockBlobService(account_name=ACCOUNT_NAME, account_key=ACCOUNT_KEY)
        print("Downloading blob to", path)
        block_blob_service.get_blob_to_path(CONTAINER_NAME, name, path)
        return path

    except AzureException as azure_error:
        print("Unable to download blob:", azure_error)
        if os.path.exists(path):
            os.remove(path)

        return None

def upload_file(path, name):
    """Uploads a file to Azure blob storage.

    Args:
        path - the path to the local file
        name - the name of the blob in Azure
    """
    try:
        block_blob_service = BlockBlobService(account_name=ACCOUNT_NAME, account_key=ACCOUNT_KEY)
        print("Uploading", path, "to blob")
        block_blob_service.create_blob_from_path(CONTAINER_NAME, name, path)

    except AzureException as azure_error:
        print("Unable to upload blob:", azure_error)
