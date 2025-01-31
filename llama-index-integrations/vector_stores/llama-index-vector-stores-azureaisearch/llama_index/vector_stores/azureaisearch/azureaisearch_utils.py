from typing import Any, Dict, List, Optional, Tuple
from llama_index.core.schema import BaseNode, TextNode
from llama_index.core.vector_stores.utils import (
    metadata_dict_to_node,
    legacy_metadata_dict_to_node,
)
import json
import logging

logger = logging.getLogger(__name__)


def create_node_from_result(
    result: Dict[str, Any],
    field_mapping: Dict[str, str],
) -> BaseNode:
    """Create a node from a search result.

    Args:
        result (Dict[str, Any]): Search result dictionary
        field_mapping (Dict[str, str]): Field mapping dictionary

    Returns:
        BaseNode: Created node
    """
    node_id = result[field_mapping["id"]]
    chunk = result[field_mapping["chunk"]]

    # Try LlamaIndex metadata first
    metadata = {}
    if field_mapping["metadata"] in result:
        metadata_str = result[field_mapping["metadata"]]
        if metadata_str:
            try:
                metadata = json.loads(metadata_str)
            except json.JSONDecodeError:
                logger.debug(
                    "Could not parse metadata JSON, if chunk is not empty, we'll use it anyways"
                )
                if len(chunk) == 0:
                    raise json.JSONDecodeError(
                        "Could not parse metadata JSON, and chunk is empty"
                    )

    try:
        # Try creating node using current metadata format
        node = metadata_dict_to_node(metadata)
        node.set_content(chunk)
    except Exception:
        # NOTE: deprecated legacy logic for backward compatibility
        try:
            metadata, node_info, relationships = legacy_metadata_dict_to_node(metadata)
        except Exception:
            # If both metadata conversions fail, assume flat metadata structure
            node_info = {}
            relationships = {}

        node = TextNode(
            text=chunk,
            id_=node_id,
            metadata=metadata,
            start_char_idx=node_info.get("start", None),
            end_char_idx=node_info.get("end", None),
            relationships=relationships,
        )

    # Add embedding if available
    if "embedding" in field_mapping:
        node.embedding = result.get(field_mapping["embedding"])

    logger.debug(f"Retrieved node id {node_id} with node data of {node}")
    return node


def process_batch_results(
    batch_nodes: List[BaseNode],
    nodes: List[BaseNode],
    batch_size: int,
    limit: Optional[int] = None,
) -> Tuple[List[BaseNode], bool]:
    """Process batch results and determine if we should continue fetching.

    Args:
        batch_nodes (List[BaseNode]): Current batch of nodes
        nodes (List[BaseNode]): Accumulated nodes
        batch_size (int): Size of each batch
        limit (Optional[int]): Maximum number of nodes to retrieve

    Returns:
        Tuple[List[BaseNode], bool]: Updated nodes list and whether to continue fetching
    """
    if not batch_nodes:
        return nodes, False

    nodes.extend(batch_nodes)

    # If we've hit the requested limit, stop
    if limit and len(nodes) >= limit:
        return nodes[:limit], False

    # If we got less than batch_size results, we've hit the end
    if len(batch_nodes) < batch_size:
        return nodes, False

    return nodes, True


def create_search_request(
    field_mapping: Dict[str, str],
    filter_str: Optional[str],
    batch_size: int,
    offset: int,
) -> Dict[str, Any]:
    """Create a search request dictionary.

    Args:
        field_mapping (Dict[str, str]): Field mapping dictionary
        filter_str (Optional[str]): OData filter string
        batch_size (int): Size of batch to retrieve
        offset (int): Number of results to skip
        semantic_config_name (Optional[str]): Name of semantic configuration to use
        vector_search_profile (Optional[str]): Name of vector search profile to use

    Returns:
        Dict[str, Any]: Search request parameters
    """
    return {
        "search_text": "*",
        "filter": filter_str,
        "top": batch_size,
        "skip": offset,
        "select": list(field_mapping.values()),
    }


def handle_search_error(e: Exception) -> None:
    """Handle search errors by logging them appropriately.

    Args:
        e (Exception): The exception that occurred
    """
    if isinstance(e, ValueError):
        logger.error(f"Invalid search parameters: {e}")
    else:
        logger.error(f"Error during search operation: {e}")
