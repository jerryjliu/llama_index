"""Base reader class."""

from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    List,
)

if TYPE_CHECKING:
    from llama_index.core.bridge.langchain import Document as LCDocument
from llama_index.core.bridge.pydantic import Field, ConfigDict
from llama_index.core.schema import BaseComponent, Document


class BaseReader(ABC):
    """Utilities for loading data from a directory."""

    def lazy_load_data(self, *args: Any, **load_kwargs: Any) -> Iterable[Document]:
        """Load data from the input directory lazily."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not provide lazy_load_data method currently"
        )

    async def alazy_load_data(
        self, *args: Any, **load_kwargs: Any
    ) -> Iterable[Document]:
        """Load data from the input directory lazily."""
        # Fake async - just calls the sync method. Override in subclasses for real async implementations.
        return self.lazy_load_data(*args, **load_kwargs)

    def load_data(self, *args: Any, **load_kwargs: Any) -> List[Document]:
        """Load data from the input directory."""
        return list(self.lazy_load_data(*args, **load_kwargs))

    async def aload_data(self, *args: Any, **load_kwargs: Any) -> List[Document]:
        """Load data from the input directory."""
        return self.load_data(*args, **load_kwargs)

    def load_langchain_documents(self, **load_kwargs: Any) -> List["LCDocument"]:
        """Load data in LangChain document format."""
        docs = self.load_data(**load_kwargs)
        return [d.to_langchain_format() for d in docs]


class BasePydanticReader(BaseReader, BaseComponent):
    """Serialiable Data Loader with Pydantic."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    is_remote: bool = Field(
        default=False,
        description="Whether the data is loaded from a remote API or a local file.",
    )


class ResourcesReaderMixin(ABC):
    """
    Mixin for readers that provide access to different types of resources.

    Resources refer to specific data entities that can be accessed by the reader.
    Examples of resources include files for a filesystem reader, channel IDs for a Slack reader, or pages for a Notion reader.
    """

    @abstractmethod
    def list_resources(self, *args: Any, **kwargs: Any) -> List[str]:
        """
        List of identifiers for the specific type of resources available in the reader.

        Returns:
            List[str]: List of identifiers for the specific type of resources available in the reader.
        """

    async def alist_resources(self, *args: Any, **kwargs: Any) -> List[str]:
        """
        List of identifiers for the specific type of resources available in the reader asynchronously.

        Returns:
            List[str]: A list of resources based on the reader type, such as files for a filesystem reader,
            channel IDs for a Slack reader, or pages for a Notion reader.
        """
        return self.list_resources(*args, **kwargs)

    def get_permission_info(self, resource_id: str, *args: Any, **kwargs: Any) -> Dict:
        """
        Get a dictionary of information about the permissions of a specific resource.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not provide get_permission_info method currently"
        )

    async def aget_permission_info(
        self, resource_id: str, *args: Any, **kwargs: Any
    ) -> Dict:
        """
        Get a dictionary of information about the permissions of a specific resource asynchronously.
        """
        return self.get_permission_info(resource_id, *args, **kwargs)

    @abstractmethod
    def get_resource_info(self, resource_id: str, *args: Any, **kwargs: Any) -> Dict:
        """
        Get a dictionary of information about a specific resource.

        Args:
            resource (str): The resource identifier.

        Returns:
            Dict: A dictionary of information about the resource.
        """

    async def aget_resource_info(
        self, resource_id: str, *args: Any, **kwargs: Any
    ) -> Dict:
        """
        Get a dictionary of information about a specific resource asynchronously.

        Args:
            resource (str): The resource identifier.

        Returns:
            Dict: A dictionary of information about the resource.
        """
        return self.get_resource_info(resource_id, *args, **kwargs)

    def list_resources_with_info(self, *args: Any, **kwargs: Any) -> Dict[str, Dict]:
        """
        Get a dictionary of information about all resources.

        Returns:
            Dict[str, Dict]: A dictionary of information about all resources.
        """
        return {
            resource: self.get_resource_info(resource, *args, **kwargs)
            for resource in self.list_resources(*args, **kwargs)
        }

    async def alist_resources_with_info(
        self, *args: Any, **kwargs: Any
    ) -> Dict[str, Dict]:
        """
        Get a dictionary of information about all resources asynchronously.

        Returns:
            Dict[str, Dict]: A dictionary of information about all resources.
        """
        return {
            resource: await self.aget_resource_info(resource, *args, **kwargs)
            for resource in await self.alist_resources(*args, **kwargs)
        }

    @abstractmethod
    def load_resource(
        self, resource_id: str, *args: Any, **kwargs: Any
    ) -> List[Document]:
        """
        Load data from a specific resource.

        Args:
            resource (str): The resource identifier.

        Returns:
            List[Document]: A list of documents loaded from the resource.
        """

    async def aload_resource(
        self, resource_id: str, *args: Any, **kwargs: Any
    ) -> List[Document]:
        """Read file from filesystem and return documents asynchronously."""
        return self.load_resource(resource_id, *args, **kwargs)

    def load_resources(
        self, resource_ids: List[str], *args: Any, **kwargs: Any
    ) -> List[Document]:
        """
        Similar to load_data, but only for specific resources.

        Args:
            resource_ids (List[str]): List of resource identifiers.

        Returns:
            List[Document]: A list of documents loaded from the resources.
        """
        return [
            doc
            for resource in resource_ids
            for doc in self.load_resource(resource, *args, **kwargs)
        ]

    async def aload_resources(
        self, resource_ids: List[str], *args: Any, **kwargs: Any
    ) -> Dict[str, List[Document]]:
        """
        Similar ato load_data, but only for specific resources.

        Args:
            resource_ids (List[str]): List of resource identifiers.

        Returns:
            Dict[str, List[Document]]: A dictionary of documents loaded from the resources.
        """
        return {
            resource: await self.aload_resource(resource, *args, **kwargs)
            for resource in resource_ids
        }


class ReaderConfig(BaseComponent):
    """Represents a reader and it's input arguments."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    reader: BasePydanticReader = Field(..., description="Reader to use.")
    reader_args: List[Any] = Field(default_factory=list, description="Reader args.")
    reader_kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Reader kwargs."
    )

    @classmethod
    def class_name(cls) -> str:
        """Get the name identifier of the class."""
        return "ReaderConfig"

    def to_dict(self, **kwargs: Any) -> Dict[str, Any]:
        """Convert the class to a dictionary."""
        return {
            "loader": self.reader.to_dict(**kwargs),
            "reader_args": self.reader_args,
            "reader_kwargs": self.reader_kwargs,
            "class_name": self.class_name(),
        }

    def read(self) -> List[Document]:
        """Call the loader with the given arguments."""
        return self.reader.load_data(*self.reader_args, **self.reader_kwargs)
