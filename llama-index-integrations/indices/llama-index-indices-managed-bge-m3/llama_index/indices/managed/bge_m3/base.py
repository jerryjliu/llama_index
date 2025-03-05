import os
import shutil
import pickle
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.data_structs.data_structs import IndexDict
from llama_index.core.indices.base import BaseIndex, IndexNode  
from llama_index.core.schema import BaseNode, NodeWithScore
from llama_index.core.storage.docstore.types import RefDocInfo
from llama_index.core.storage.storage_context import StorageContext

class SafeUnpickler(pickle.Unpickler):
    ALLOWED_CLASSES = {
        "builtins": {"list", "dict", "str", "int", "float", "tuple", "set"},
        "numpy.core.multiarray": {"_reconstruct", "scalar"},
        "numpy": {"ndarray", "dtype"},
        "collections": {"defaultdict"}
    }

    def find_class(self, module, name):
        if module in self.ALLOWED_CLASSES and name in self.ALLOWED_CLASSES[module]:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"Unauthorized class: {module}.{name}")

def safe_pickle_load(file):
    return SafeUnpickler(file).load()


class BGEM3Index(BaseIndex[IndexDict]):
    """
    Store for BGE-M3 with PLAID indexing.

    BGE-M3 is a multilingual embedding model with multi-functionality:
    Dense retrieval, Sparse retrieval and Multi-vector retrieval.

    Parameters:

    index_path: directory containing PLAID index files.
    model_name: BGE-M3 hugging face model name.
        Default: "BAAI/bge-m3".
    show_progress: whether to show progress bar when building index.
        Default: False. noop for BGE-M3 for now.
    doc_maxlen: max document length. Default: 120.
    query_maxlen: max query length. Default: 60.

    """

    def __init__(
        self,
        nodes: Optional[Sequence[BaseNode]] = None,
        objects: Optional[Sequence[IndexNode]] = None,
        index_struct: Optional[IndexDict] = None,
        storage_context: Optional[StorageContext] = None,
        model_name: str = "BAAI/bge-m3",
        index_name: str = "",
        show_progress: bool = False,
        pooling_method: str = "cls",
        normalize_embeddings: bool = True,
        use_fp16: bool = False,
        batch_size: int = 32,
        doc_maxlen: int = 8192,
        query_maxlen: int = 8192,
        weights_for_different_modes: List[float] = None,
        **kwargs: Any,
    ) -> None:
        self.index_path = "storage/bge_m3_index"
        self.index_name = index_name
        self.batch_size = batch_size
        self.doc_maxlen = doc_maxlen
        self.query_maxlen = query_maxlen
        self.weights_for_different_modes = weights_for_different_modes
        self._multi_embed_store = None
        self._docs_pos_to_node_id: Dict[int, str] = {}
        try:
            from FlagEmbedding import BGEM3FlagModel
        except ImportError as exc:
            raise ImportError(
                "Please install FlagEmbedding to use this feature from the repo:",
                "https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/BGE_M3",
            ) from exc
        self.model = BGEM3FlagModel(
            model_name,
            pooling_method=pooling_method,
            normalize_embeddings=normalize_embeddings,
            use_fp16=use_fp16,
        )
        super().__init__(
            nodes=nodes,
            index_struct=index_struct,
            index_name=index_name,
            storage_context=storage_context,
            show_progress=show_progress,
            objects=objects,
            **kwargs,
        )

    def _insert(self, nodes: Sequence[BaseNode], **insert_kwargs: Any) -> None:
        raise NotImplementedError("BGEM3Index does not support insertion yet.")

    def _delete_node(self, node_id: str, **delete_kwargs: Any) -> None:
        raise NotImplementedError("BGEM3Index does not support deletion yet.")

    def as_retriever(self, **kwargs: Any) -> BaseRetriever:
        from .retriever import BGEM3Retriever

        return BGEM3Retriever(index=self, object_map=self._object_map, **kwargs)

    @property
    def ref_doc_info(self) -> Dict[str, RefDocInfo]:
        raise NotImplementedError("BGEM3Index does not support ref_doc_info.")

    def _build_index_from_nodes(
        self, nodes: Sequence[BaseNode], **kwargs: Any
    ) -> IndexDict:
        """Generate a PLAID index from the BGE-M3 checkpoint via its hugging face
        model_name.
        """
        index_struct = IndexDict()

        docs_list = []
        for i, node in enumerate(nodes):
            docs_list.append(node.get_content())
            self._docs_pos_to_node_id[i] = node.node_id
            index_struct.add_node(node, text_id=str(i))

        self._multi_embed_store = self.model.encode(
            docs_list,
            batch_size=self.batch_size,
            max_length=self.doc_maxlen,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=True,
        )
        return index_struct

    def persist(self, persist_dir: str) -> None:
        # Check if the destination directory exists
        if os.path.exists(persist_dir):
            # Remove the existing destination directory
            shutil.rmtree(persist_dir)

        self._storage_context.persist(persist_dir=persist_dir)
        # Save _multi_embed_store
        pickle.dump(
            self._multi_embed_store,
            open(Path(persist_dir) / "multi_embed_store.pkl", "wb"),
        )

    @classmethod
    def load_from_disk(
        cls,
        persist_dir: str,
        model_name: str = "BAAI/bge-m3",
        index_name: str = "",
        weights_for_different_modes: List[float] = None,
    ) -> "BGEM3Index":
        sc = StorageContext.from_defaults(persist_dir=persist_dir)
        index = BGEM3Index(
            model_name=model_name,
            index_name=index_name,
            index_struct=sc.index_store.index_structs()[0],
            storage_context=sc,
            weights_for_different_modes=weights_for_different_modes,
        )
        docs_pos_to_node_id = {
            int(k): v for k, v in index.index_struct.nodes_dict.items()
        }
        index._docs_pos_to_node_id = docs_pos_to_node_id
        index._multi_embed_store = safe_pickle_load(
            open(Path(persist_dir) / "multi_embed_store.pkl", "rb")
        )
        return index

    def query(self, query_str: str, top_k: int = 10) -> List[NodeWithScore]:
        """
        Query the BGE-M3 + Plaid store.

        Returns: list of NodeWithScore.
        """
        query_embed = self.model.encode(
            query_str,
            batch_size=self.batch_size,
            max_length=self.query_maxlen,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=True,
        )

        dense_scores = np.matmul(
            query_embed["dense_vecs"], self._multi_embed_store["dense_vecs"].T
        )

        sparse_scores = np.array(
            [
                self.model.compute_lexical_matching_score(
                    query_embed["lexical_weights"], doc_lexical_weights
                )
                for doc_lexical_weights in self._multi_embed_store["lexical_weights"]
            ]
        )

        colbert_scores = np.array(
            [
                self.model.colbert_score(
                    query_embed["colbert_vecs"], doc_colbert_vecs
                ).item()
                for doc_colbert_vecs in self._multi_embed_store["colbert_vecs"]
            ]
        )

        if self.weights_for_different_modes is None:
            weights_for_different_modes = [1.0, 1.0, 1.0]
            weight_sum = 3.0
        else:
            weights_for_different_modes = self.weights_for_different_modes
            weight_sum = sum(weights_for_different_modes)

        combined_scores = (
            dense_scores * weights_for_different_modes[0]
            + sparse_scores * weights_for_different_modes[1]
            + colbert_scores * weights_for_different_modes[2]
        ) / weight_sum

        topk_indices = np.argsort(combined_scores)[::-1][:top_k]
        topk_scores = [combined_scores[idx] for idx in topk_indices]

        node_doc_ids = [self._docs_pos_to_node_id[idx] for idx in topk_indices]
        nodes = self.docstore.get_nodes(node_doc_ids)

        nodes_with_score = []
        for node, score in zip(nodes, topk_scores):
            nodes_with_score.append(NodeWithScore(node=node, score=score))
        return nodes_with_score
