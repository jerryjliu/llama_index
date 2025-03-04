"""JSON Reader."""

import json
import re
from typing import Any, Dict, Generator, List, Optional

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document


def _depth_first_yield(
    json_data: Any,
    levels_back: int,
    collapse_length: Optional[int],
    path: List[str],
    ensure_ascii: bool = False,
) -> Generator[str, None, None]:
    """Do depth first yield of all of the leaf nodes of a JSON.

    Combines keys in the JSON tree using spaces.

    If levels_back is set to 0, prints all levels.
    If collapse_length is not None and the json_data is <= that number
      of characters, then we collapse it into one line.

    """
    if isinstance(json_data, (dict, list)):
        # only try to collapse if we're not at a leaf node
        json_str = json.dumps(json_data, ensure_ascii=ensure_ascii)
        if collapse_length is not None and len(json_str) <= collapse_length:
            new_path = path[-levels_back:]
            new_path.append(json_str)
            yield " ".join(new_path)
            return
        elif isinstance(json_data, dict):
            for key, value in json_data.items():
                new_path = path[:]
                new_path.append(key)
                yield from _depth_first_yield(
                    value, levels_back, collapse_length, new_path
                )
        elif isinstance(json_data, list):
            for _, value in enumerate(json_data):
                yield from _depth_first_yield(value, levels_back, collapse_length, path)
    else:
        new_path = path[-levels_back:]
        new_path.append(str(json_data))
        yield " ".join(new_path)


class JSONReader(BaseReader):
    """JSON reader.

    Reads JSON documents with options to help us out relationships between nodes.

    Args:
        levels_back (int): the number of levels to go back in the JSON tree, 0
          if you want all levels. If levels_back is None, then we just format the
          JSON and make each line an embedding

        collapse_length (int): the maximum number of characters a JSON fragment
          would be collapsed in the output (levels_back needs to be not None)
          ex: if collapse_length = 10, and
          input is {a: [1, 2, 3], b: {"hello": "world", "foo": "bar"}}
          then a would be collapsed into one line, while b would not.
          Recommend starting around 100 and then adjusting from there.

        is_jsonl (Optional[bool]): If True, indicates that the file is in JSONL format.
        Defaults to False.

        clean_json (Optional[bool]): If True, lines containing only JSON structure are removed.
        This removes lines that are not as useful. If False, no lines are removed and the document maintains a valid JSON object structure.
        If levels_back is set the json is not cleaned and this option is ignored.
        Defaults to True.
    """

    def __init__(
        self,
        levels_back: Optional[int] = None,
        collapse_length: Optional[int] = None,
        ensure_ascii: bool = False,
        is_jsonl: Optional[bool] = False,
        clean_json: Optional[bool] = True,
    ) -> None:
        """Initialize with arguments."""
        super().__init__()
        self.levels_back = levels_back
        self.collapse_length = collapse_length
        self.ensure_ascii = ensure_ascii
        self.is_jsonl = is_jsonl
        self.clean_json = clean_json

    def load_data(
        self, input_file: str, metadata: Optional[Dict] = {}
    ) -> List[Document]:
        """Load data from the input file."""
        with open(input_file, encoding="utf-8") as f:
            load_data = []
            if self.is_jsonl:
                for line in f:
                    load_data.append(json.loads(line.strip()))
            else:
                load_data = [json.load(f)]

            documents = []
            for data in load_data:
                if self.levels_back is None and self.clean_json is True:
                    # If levels_back isn't set and clean json is set,
                    # remove lines containing only formatting, we just format and make each
                    # line an embedding
                    json_output = json.dumps(
                        data, indent=0, ensure_ascii=self.ensure_ascii
                    )
                    lines = json_output.split("\n")
                    useful_lines = [
                        line for line in lines if not re.match(r"^[{}\[\],]*$", line)
                    ]
                    documents.append(
                        Document(text="\n".join(useful_lines), metadata=metadata)
                    )

                elif self.levels_back is None and self.clean_json is False:
                    # If levels_back isn't set  and clean json is False, create documents without cleaning
                    json_output = json.dumps(data, ensure_ascii=self.ensure_ascii)
                    documents.append(Document(text=json_output, metadata=metadata))

                elif self.levels_back is not None:
                    # If levels_back is set, we make the embeddings contain the labels
                    # from further up the JSON tree
                    lines = [
                        *_depth_first_yield(
                            data,
                            self.levels_back,
                            self.collapse_length,
                            [],
                            self.ensure_ascii,
                        )
                    ]
                    documents.append(
                        Document(text="\n".join(lines), metadata=metadata)
                    )
            return documents
