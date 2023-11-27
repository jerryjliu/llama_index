from typing import List, Tuple, Type

import llama_index.download.download_utils as download_utils
from llama_index import Document
from llama_index.llama_dataset.base import BaseLlamaDataset
from llama_index.llama_dataset.rag import LabelledRagDataset
from llama_index.readers import SimpleDirectoryReader


def download_llama_dataset(
    llama_dataset_class: str,
    download_dir: str,
    llama_hub_url: str = download_utils.LLAMA_HUB_URL,
    llama_datasets_url: str = download_utils.LLAMA_DATASETS_URL,
) -> Tuple[Type[BaseLlamaDataset], List[Document]]:
    """Download a single LlamaDataset from Llama Hub.

    Args:
        llama_dataset_class: The name of the LlamaPack class you want to download,
            such as `PaulGrahamEssayDataset`.
        refresh_cache: If true, the local cache will be skipped and the
            loader will be fetched directly from the remote repo.
        download_dir: Custom dirpath to download the pack into.

    Returns:
        A Loader.
    """
    rag_dataset_filename, source_filenames = download_utils.download_llama_dataset(
        llama_dataset_class,
        llama_hub_url=llama_hub_url,
        llama_datasets_url=llama_datasets_url,
        refresh_cache=True,
        custom_path=download_dir,
        library_path="llama_datasets/library.json",
        disable_library_cache=True,
        override_path=True,
    )

    return (
        LabelledRagDataset.from_json(rag_dataset_filename),
        SimpleDirectoryReader(input_files=source_filenames).load_data(),
    )
