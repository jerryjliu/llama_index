"""Simple reader that ."""
from gpt_index.schema import Document
from gpt_index.readers.base import BaseReader
from typing import List
from pathlib import Path


class SimpleDirectoryReader(BaseReader):
    """Simple directory reader.

    Concatenates all files into one document text.

    """

    def __init__(self, input_dir: Path) -> None:
        """Initialize with parameters."""
        self.input_dir = input_dir
        input_files = list(input_dir.iterdir())
        for input_file in input_files:
            if not input_file.is_file():
                raise ValueError(f"Expected {input_file} to be a file.")
        self.input_files = input_files

    def load_data(self) -> List[Document]:
        """Load data from the input directory."""
        data = ""
        for input_file in self.input_files:
            with open(input_file, "r") as f:
                data += f.read()
            data += "\n"
        return [Document(data)]
