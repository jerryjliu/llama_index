"""JSON Reader."""

import json
import re
from typing import Any, Generator, List, Optional

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document, NodeRelationship, RelatedNodeInfo

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
        # Only try to collapse if we're not at a leaf node
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


class HierarchicalJSONReader(BaseReader):
    """Class JSON Reader extended to support hierarchical metadata.

    Reads JSON documents with options to help suss out relationships between nodes.

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

    """

    def __init__(
        self,
        levels_back: Optional[int] = None,
        collapse_length: Optional[int] = None,
        ensure_ascii: bool = False,
        is_jsonl: Optional[bool] = False,
    ) -> None:
        """Initialize with arguments."""
        super().__init__()
        self.levels_back = levels_back
        self.collapse_length = collapse_length
        self.ensure_ascii = ensure_ascii
        self.is_jsonl = is_jsonl

    def transform_data(self, input_data):
        transformed_data_list = []
        metadata_list = []
        relationships_list = []

        for data in input_data:
            transformed_metadata = {}
            categories = []
            subcategories = []
            relationships = {}

            subcategory_node_ids = {}

            for category, contents in data.items():
                if category == "metadata":
                    continue
                categories.append(category)

                for subcategory, content_list in contents.items():
                    if isinstance(content_list, list):
                        # Processes as a list
                        new_subcategory = f"{category} - {subcategory}"
                        subcategories.append(new_subcategory)

                        subcategory_id = len(subcategory_node_ids) + 1
                        subcategory_node_ids[new_subcategory] = subcategory_id

                        for content in content_list:
                            content_id = f"content-{subcategory_id}-{content_list.index(content)}"
                            relationships[content_id] = subcategory_id

                        transformed_metadata[new_subcategory] = content_list
                    elif isinstance(content_list, dict):
                        # Processes as a dictionary
                        for nested_subcategory, nested_content in content_list.items():
                            combined_subcategory = f"{subcategory} - {nested_subcategory}"
                            subcategories.append(combined_subcategory)

                            subcategory_id = len(subcategory_node_ids) + 1
                            subcategory_node_ids[combined_subcategory] = subcategory_id

                            if isinstance(nested_content, list):
                                for content in nested_content:
                                    content_id = f"content-{subcategory_id}-{nested_content.index(content)}"
                                    relationships[content_id] = subcategory_id

                            transformed_metadata[combined_subcategory] = nested_content

            extracted_metadata = {
                "categories": categories,
                "subcategories": subcategories
            }

            transformed_data_list.append(transformed_metadata)
            metadata_list.append(extracted_metadata)
            relationships_list.append(relationships)

        return transformed_data_list, metadata_list, relationships_list


    def load_data(self, input_file: str) -> List[Document]:
        with open(input_file, encoding="utf-8") as f:
            if self.is_jsonl:
                load_data = [json.loads(line.strip()) for line in f]
            else:
                load_data = json.load(f)

        transformed_data, metadata_list, relationships_list = self.transform_data(load_data)

        documents = []

        for idx, (data, metadata, relationships) in enumerate(zip(transformed_data, metadata_list, relationships_list)):
            if self.levels_back is None:
                # Keeps the formatting of the original JSONReader in the document text
                json_output = json.dumps(data, indent=1, ensure_ascii=self.ensure_ascii)
                json_output = json_output.replace("\n  ", "\n").replace("\n ", "\n")
                
                # Iterates over each category
                for category in metadata['categories']:
                    # Filters subcategories for the current category
                    related_subcategories = [sc for sc in metadata['subcategories'] if sc.startswith(f"{category} -")]

                    # Creation of the category document (root node)
                    category_doc = Document(
                        text=category,
                        metadata={"category": category, "subcategories": related_subcategories},
                        relationships={}
                    )
                    documents.append(category_doc)

                    # Iterates through the filtered subcategories and contents associated with the category
                    for subcategory in related_subcategories:
                        contents = data.get(subcategory, [])
                        contents_text = json.dumps(contents, ensure_ascii=self.ensure_ascii, indent=1)
                        subcategory_text = f"{subcategory}: {contents_text}"

                        # Creation of the subcategory document with contents
                        subcategory_doc = Document(
                            text=subcategory_text,
                            metadata={"category": category, "subcategory": subcategory},
                            relationships={NodeRelationship.PARENT: RelatedNodeInfo(node_id=category_doc.id_)}
                        )
                        documents.append(subcategory_doc)                            
                            
            else:
                # If self.levels_back is set, process differently
                lines = [
                    *_depth_first_yield(data, self.levels_back, self.collapse_length, [], self.ensure_ascii)
                ]
                documents.append(Document(text="\n".join(lines)))

        return documents
