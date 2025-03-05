import hashlib
import json
import logging
import os
import tempfile
import threading
import time
import warnings
from llama_index.core.readers.base import BaseReader
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BufferedIOBase
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

try:
    from google import genai
except ImportError:
    import_err_msg = """ `google-genai` package not found.
    Install google-genai package by running:
    ```pip install google-genai```
    """
    raise ImportError(import_err_msg)

import PIL.Image
from pdf2image import convert_from_path
from pydantic import BaseModel, Field, field_validator

from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.schema import Document

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

FileInput = Union[str, bytes, BufferedIOBase, Path]


class TextChunk(BaseModel):
    """A text chunk with metadata extracted from document."""

    text: str = Field(description="The extracted text content")


class Chunks(BaseModel):
    """Model for structured output from Gemini."""

    chunks: List[TextChunk] = Field(description="List of text chunks")


class ProcessingStats(BaseModel):
    """Statistics about the PDF processing operation."""

    start_time: float
    end_time: Optional[float] = None
    total_pages: int = 0
    processed_pages: int = 0
    total_chunks: int = 0
    errors: List[str] = Field(default_factory=list)

    @property
    def duration(self) -> float:
        """Get the processing duration in seconds."""
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time

    @property
    def pages_per_second(self) -> float:
        """Get the processing rate in pages per second."""
        if self.processed_pages == 0:
            return 0
        return self.processed_pages / self.duration

    @property
    def chunks_per_page(self) -> float:
        """Get the average number of chunks per page."""
        if self.processed_pages == 0:
            return 0
        return self.total_chunks / self.processed_pages


class CacheItem(BaseModel):
    """Cache item for storing processed PDF results."""

    timestamp: float
    document_hash: str
    documents: List[Dict[str, Any]]
    stats: ProcessingStats


class GeminiReader(BaseReader):
    """
    A PDF reader that uses Google's Gemini API for OCR
        and intelligent document chunking.
    This reader processes PDFs by converting them to images and using
    Gemini's multimodal capabilities to extract text, handle complex layouts,
    recognize tables, forms, and other document structures.
    It produces semantically meaningful chunks optimized for retrieval.

    Features:
    - High-quality OCR with layout preservation
    - Intelligent semantic chunking
    - Table and form recognition
    - Mathematical formula extraction
    - Multi-language support
    - Caching for repeated processing
    - Parallel processing for improved performance

    Args:
        api_key (str): The API key for Google's Gemini API.
        model_name (str): The model name to use. Default: "gemini-2.0-flash".
        split_by_page (bool): Whether to split the document by page. Default: True.
        verbose (bool): Whether to print progress information. Default: False.
        ignore_errors (bool): Whether to ignore errors and continue processing. Default: True.
        dpi (int): DPI for PDF to image conversion. Default: 600.
        language (str): Primary language of the documents. Default: "en".
        max_workers (int): Maximum number of workers for parallel processing. Default: 4.
        enable_caching (bool): Whether to cache processed results. Default: True.
        cache_dir (str): Directory for caching processed results. Default: temp directory.
        cache_ttl (int): Time-to-live for cache entries in seconds. Default: 24 hours.
        extract_forms (bool): Whether to apply special handling for forms. Default: True.
        extract_tables (bool): Whether to extract and format tables. Default: True.
        extract_math_formulas (bool): Whether to extract mathematical formulas. Default: True.
        continuous_mode (bool): Process PDF continuously for cross-page content. Default: False.
        chunk_size (str): Target size range for chunks in words. Default: "256-512".
        max_retries (int): Maximum number of retries for API calls. Default: 3.
        retry_delay (float): Delay between retries in seconds. Default: 1.0.
        progress_callback (Optional[Callable]): Callback for reporting progress. Default: None.
    """

    # API access configuration
    api_key: str = Field(
        description="The API key for the Google Gemini API",
        validate_default=True
    )
    model_name: str = Field(
        default="gemini-2.0-flash",
        description="The model name to use for OCR and chunking",
    )

    # Processing configuration
    split_by_page: bool = Field(
        default=True, description="Whether to split the document by page"
    )
    verbose: bool = Field(
        default=False, description="Whether to print progress information"
    )
    ignore_errors: bool = Field(
        default=True,
        description="Whether to ignore errors and continue processing"
    )
    dpi: int = Field(
        default=300, description="DPI for PDF to image conversion"
    )
    language: str = Field(
        default="en", description="Primary language of the documents"
    )
    max_workers: int = Field(
        default=4,
        gt=0,
        lt=11,
        description="Maximum number of workers for parallel processing",
    )
    continuous_mode: bool = Field(
        default=False,
        description="Parse documents continuously, maintaining content across page boundaries",
    )
    chunk_size: Union[str, int] = Field(
        default="256-512",
        description=(
            "Target size range for chunks in words (e.g., '256-512')",
            "Can also be a single number (e.g., 256).",
        ),
    )

    # Caching configuration
    enable_caching: bool = Field(
        default=True, description="Whether to cache processed results"
    )
    cache_dir: str = Field(
        default=os.path.join(tempfile.gettempdir(), "gemini_pdf_cache"),
        description="Directory for caching processed results",
    )
    cache_ttl: int = Field(
        default=24 * 60 * 60,  # 24 hours
        description="Time-to-live for cache entries in seconds",
    )

    # Document content extraction options
    extract_forms: bool = Field(
        default=True, description="Whether to apply special handling for forms"
    )
    extract_tables: bool = Field(
        default=True, description="Whether to extract and format tables"
    )
    extract_math_formulas: bool = Field(
        default=True, 
        description="Whether to extract and format mathematical formulas"
    )

    # Retry configuration
    max_retries: int = Field(
        default=3, description="Maximum number of retries for API calls"
    )
    retry_delay: float = Field(
        default=1.0, description="Delay between retries in seconds"
    )

    # Private attributes
    _client: Optional[genai.Client] = PrivateAttr(default=None)
    _progress_callback: Optional[Callable[[int, int], None]] = PrivateAttr(
        default=None)
    _cache: Dict[str, CacheItem] = PrivateAttr(default_factory=dict)
    _stats: ProcessingStats = PrivateAttr(
        default_factory=lambda: ProcessingStats(start_time=time.time())
    )

    # Constants
    # System prompt template for document extraction and chunking
    SYSTEM_PROMPT_TEMPLATE: str = """
ANALYZE THIS DOCUMENT IMAGE AND EXTRACT CHUNK TEXTS:

STEP 1: OCR - Optical Character Recognition
- Extract all visible text from the document image with high accuracy
- Preserve the original document structure in clean markdown format
- Maintain proper reading order and paragraph flow
- Correctly identify and format headings, subheadings, and sections
- Preserve footnotes, citations, and references when present
- Distinguish between main content, captions, and sidebars
{form_instructions}
- Include text from tables with proper structural formatting
{math_formula_instructions}

STEP 2: Document Chunking 
- Identify natural content boundaries in the document
- Create semantic chunks of approximately {chunk_instruction} words each
- Respect the document's logical structure (sections, subsections)
- Keep coherent content together in the same chunk to avoid fragmentation of meaning
- Preserve the relationship between:
  * Questions and answers
  * Headers and their content
  * Lists and their context
  * Citations and their references
  * Images and their captions
- Never split tables, figures, or equations across chunks
- Optimize chunks for semantic relevance in a retrieval system

General Processing Guidelines:
1. Preserve the document's logical flow and reading order
2. Handle various layout types (single-column, multi-column, complex layouts)
3. Distinguish between different textual elements (body text, headers, captions, footnotes)
4. Ensure complete text extraction with no content omissions
5. Maintain semantic coherence within each chunk
6. Preserve the context and meaning of the original document
7. Handle specialized formatting appropriately for the document type
{table_instructions}
{checkbox_instructions}
10. For hierarchical or nested content:
   - Maintain the hierarchical structure
   - Preserve parent-child relationships between elements
   - Keep related components together
{language_instructions}

For each document image, provide:
1. A list of "chunks" with:
   - "text": The extracted text content formatted in clean markdown

Format your response matching the specified schema.
"""

    # Form-specific instructions
    FORM_INSTRUCTIONS: str = """
- For forms and structured documents:
  * Preserve the relationship between labels and their fields
  * Maintain the logical structure of forms and questionnaires
  * Keep related form elements together
  * Preserve the state of selection elements (checkboxes, radio buttons)
  * Maintain the distinction between questions and their possible answers
  * If form has multiple questions, keep them in the same chunk
"""

    # Table-specific instructions
    TABLE_INSTRUCTIONS: str = """
8. For tables and tabular data:
   - Preserve the complete table structure using markdown table format
   - Maintain column headers with their respective data
   - Properly align and format all table cells
   - Handle merged cells, spanning cells, and nested tables appropriately
   - Preserve numerical precision and formatting in table cells
   - Include table captions or titles when present
   - For complex tables, maintain the relationship between data cells
   - Ensure all table content is included in the same chunk
"""

    # Checkbox-specific instructions
    CHECKBOX_INSTRUCTIONS: str = """
9. For documents with selection elements:
   - Represent checkboxes using standard Unicode symbols:
     * ☐ (U+2610) for unchecked boxes
     * ☑ (U+2611) or ☒ (U+2612) for checked boxes
   - Maintain consistent formatting for selection indicators
   - Preserve the relationship between selection elements and their labels
   - Correctly identify selected vs. unselected options
   - Keep related checkbox options together in the same chunk
"""

    # Mathematical formula instructions
    MATH_FORMULA_INSTRUCTIONS: str = """
- For documents with mathematical content:
  * Extract equations and formulas accurately
  * Represent mathematical notation using appropriate markdown syntax
  * Maintain the relationship between formulas and surrounding text
  * Preserve subscripts, superscripts, and special mathematical symbols
"""

    # Language-specific instructions template
    LANGUAGE_INSTRUCTIONS_TEMPLATE: str = """
11. This document appears to be primarily in {language}.
Optimize text extraction for this language, paying attention to:
    - Character sets and special symbols specific to {language}
    - Reading order conventions for {language}
    - Proper handling of language-specific punctuation and formatting
"""

    @field_validator("api_key", mode="before")
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        """
        Validate the API key from parameter or environment variable.

        Args:
            v (str): API key value

        Returns:
            str: Valid API key

        Raises:
            ValueError: If API key is not provided and not in environment
        """
        if not v:
            api_key = os.getenv("GOOGLE_API_KEY")
            if api_key is None:
                raise ValueError(
                    "API key not provided and GOOGLE_API_KEY environment variable not set"
                )
            return api_key
        return v

    def __init__(
        self,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        **kwargs: Any,
    ):
        """
        Initialize with parameters.

        Args:
            progress_callback (Optional[Callable]): Optional callback function for reporting progress
            **kwargs: Additional parameters passed to the base class
        """
        super().__init__(**kwargs)

        if self.continuous_mode:
            warnings.warn(
                "Warning: continuous_mode is enabled. This may result in larger chunk sizes."
            )
        self._progress_callback = progress_callback
        self._initialize_client()
        self._initialize_cache()

    def _initialize_client(self) -> None:
        """Initialize the Gemini API client."""
        self._client = genai.Client(api_key=self.api_key)
        if self.verbose:
            logger.info(f"Initialized Gemini PDF reader with model: {self.model_name}")

    def _initialize_cache(self) -> None:
        """Initialize the cache directory and load existing cache entries."""
        if not self.enable_caching:
            return

        os.makedirs(self.cache_dir, exist_ok=True)

        # Load existing cache entries
        if os.path.exists(self.cache_dir):
            for filename in os.listdir(self.cache_dir):
                if filename.endswith(".json"):
                    try:
                        with open(os.path.join(self.cache_dir, filename), "r") as f:
                            cache_data = json.load(f)
                            cache_item = CacheItem(**cache_data)

                            # Check if the cache entry is still valid
                            if time.time() - cache_item.timestamp < self.cache_ttl:
                                self._cache[cache_item.document_hash] = cache_item
                    except Exception as e:
                        if self.verbose:
                            logger.warning(
                                f"Failed to load cache entry {filename}: {str(e)}"
                            )

    @property
    def client(self) -> genai.Client:
        """
        Get the Gemini API client, initializing if needed.

        Returns:
            genai.Client: Initialized API client
        """
        if self._client is None:
            self._initialize_client()
        return self._client

    @property
    def system_prompt(self) -> str:
        """
        Generate the system prompt based on configuration.

        Returns:
            str: Customized system prompt for the model
        """
        # Fill in template sections based on configuration
        form_instructions = self.FORM_INSTRUCTIONS if self.extract_forms else ""
        table_instructions = self.TABLE_INSTRUCTIONS if self.extract_tables else ""
        checkbox_instructions = self.CHECKBOX_INSTRUCTIONS if self.extract_forms else ""
        language_instructions = self.LANGUAGE_INSTRUCTIONS_TEMPLATE.format(
            language=self.language
        )
        math_formula_instructions = (
            self.MATH_FORMULA_INSTRUCTIONS if self.extract_math_formulas else ""
        )

        # Format the template
        return self.SYSTEM_PROMPT_TEMPLATE.format(
            form_instructions=form_instructions,
            table_instructions=table_instructions,
            checkbox_instructions=checkbox_instructions,
            language_instructions=language_instructions,
            math_formula_instructions=math_formula_instructions,
            chunk_instruction=self.chunk_size,
        )

    def _update_progress(self, current: int, total: int) -> None:
        """
        Update progress and call the progress callback if provided.

        Args:
            current (int): Current progress value
            total (int): Total expected operations
        """
        if self._progress_callback:
            self._progress_callback(current, total)

        if self.verbose:
            progress_pct = int(100 * current / total) if total > 0 else 0
            logger.info(f"Progress: {progress_pct}% ({current}/{total})")

    def _compute_file_hash(self, file_path: str) -> str:
        """
        Compute a hash of the file for caching purposes.

        Args:
            file_path (str): Path to the file

        Returns:
            str: MD5 hash string
        """
        file_stat = os.stat(file_path)
        # Use file path, size and modification time for the hash
        hash_string = f"{file_path}:{file_stat.st_size}:{file_stat.st_mtime}"
        return hashlib.md5(hash_string.encode()).hexdigest()

    def _save_to_cache(
        self, file_hash: str, documents: List[Document], stats: ProcessingStats
    ) -> None:
        """
        Save processed results to cache.

        Args:
            file_hash (str): Hash identifier for the document
            documents (List[Document]): Processed document chunks
            stats (ProcessingStats): Processing statistics
        """
        if not self.enable_caching:
            return

        try:
            # Convert Documents to dictionaries
            doc_dicts = []
            for doc in documents:
                doc_dict = {"text": doc.text, "metadata": doc.metadata}
                doc_dicts.append(doc_dict)

            # Create cache item
            cache_item = CacheItem(
                timestamp=time.time(),
                document_hash=file_hash,
                documents=doc_dicts,
                stats=stats,
            )

            # Save to memory cache
            self._cache[file_hash] = cache_item

            # Save to disk cache
            cache_path = os.path.join(self.cache_dir, f"{file_hash}.json")
            with open(cache_path, "w") as f:
                json.dump(cache_item.model_dump(), f)

            if self.verbose:
                logger.info(
                    f"Cached results for {file_hash} (documents: {len(documents)})"
                )

        except Exception as e:
            logger.warning(f"Failed to cache results: {str(e)}")

    def _load_from_cache(self, file_hash: str) -> Optional[List[Document]]:
        """
        Load processed results from cache if available.

        Args:
            file_hash (str): Hash identifier for the document

        Returns:
            Optional[List[Document]]: Cached document chunks or None if not in cache
        """
        if not self.enable_caching:
            return None

        cache_item = self._cache.get(file_hash)
        if not cache_item:
            return None

        # Check if the cache entry is still valid
        if time.time() - cache_item.timestamp > self.cache_ttl:
            if self.verbose:
                logger.info(f"Cache entry expired for {file_hash}")
            return None

        # Convert cached dictionaries back to Documents
        documents = []
        for doc_dict in cache_item.documents:
            doc = Document(text=doc_dict["text"], metadata=doc_dict["metadata"])
            documents.append(doc)

        if self.verbose:
            logger.info(f"Loaded {len(documents)} documents from cache for {file_hash}")

        return documents

    def convert_pdf_to_images(self, pdf_path: str) -> List[PIL.Image.Image]:
        """
        Convert a PDF file to a list of PIL images.

        Args:
            pdf_path (str): Path to the PDF file

        Returns:
            List[PIL.Image.Image]: List of PIL images

        Raises:
            FileNotFoundError: If the PDF file cannot be found
            RuntimeError: If PDF conversion fails
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        if self.verbose:
            logger.info(f"Converting PDF to images: {pdf_path} (DPI: {self.dpi})")

        try:
            # Convert PDF to images using pdf2image
            images = convert_from_path(
                pdf_path=pdf_path,
                dpi=self.dpi,
                fmt="png",
                thread_count=os.cpu_count() or 4,
                use_cropbox=True,
                use_pdftocairo=True,
                grayscale=False,
                strict=False,
            )
            self._stats.total_pages = len(images)
            if self.verbose:
                logger.info(f"Converted PDF to {len(images)} images")
            return images
        except Exception as e:
            logger.error(f"PDF conversion error: {str(e)}")
            self._stats.errors.append(f"PDF conversion error: {str(e)}")
            raise RuntimeError(f"Failed to convert PDF: {str(e)}") from e

    def process_image(
        self, image: PIL.Image.Image, retry_count: int = 0
    ) -> List[TextChunk]:
        """
        Process a single image for OCR and document chunking.

        Args:
            image (PIL.Image.Image): Image to process
            retry_count (int): Current retry attempt (for internal use)

        Returns:
            List[TextChunk]: List of text chunks extracted from the image

        Raises:
            RuntimeError: If processing fails after retries
        """
        try:
            if self.verbose and retry_count == 0:
                logger.info(f"Processing image ({image.width}x{image.height})")
            elif self.verbose and retry_count > 0:
                logger.info(
                    f"Retrying image processing (attempt {retry_count}/{self.max_retries})"
                )

            # Generate OCR and chunking
            response = self.client.models.generate_content(
                contents=[image, self.system_prompt],
                model=self.model_name,
                config={
                    "temperature": 0,
                    "response_mime_type": "application/json",
                    "response_schema": Chunks,
                },
            )

            chunks = response.parsed.chunks
            if self.verbose:
                logger.info(f"Processed document with {len(chunks)} chunks")

            return chunks

        except Exception as e:
            # Handle retries
            if retry_count < self.max_retries:
                if self.verbose:
                    logger.warning(f"Processing error: {str(e)}. Retrying...")

                # Wait before retrying
                time.sleep(self.retry_delay * (retry_count + 1))

                # Retry with incremented count
                return self.process_image(image, retry_count + 1)
            else:
                logger.error(
                    f"Processing error after {self.max_retries} retries: {str(e)}"
                )
                self._stats.errors.append(f"Processing error: {str(e)}")
                raise RuntimeError(f"Failed to process image: {str(e)}") from e

    def _update_stats(self, chunks_count: int) -> None:
        """
        Update processing statistics.

        Args:
            chunks_count (int): Number of chunks processed
        """
        self._stats.processed_pages += 1
        self._stats.total_chunks += chunks_count

        # Update progress
        self._update_progress(self._stats.processed_pages, self._stats.total_pages)

    def _process_page(
        self, args: Tuple[PIL.Image.Image, str, int, int]
    ) -> List[Document]:
        """
        Process a single page of the PDF.

        Args:
            args: Tuple containing (image, pdf_path, page_number, total_pages)

        Returns:
            List[Document]: Documents extracted from the page

        Raises:
            RuntimeError: If page processing fails and ignore_errors is False
        """
        image, pdf_path, page_number, total_pages = args
        documents = []

        try:
            # Process the image
            text_chunks = self.process_image(image)

            # Create Document objects
            if self.split_by_page:
                # Create a separate Document for each chunk
                for chunk_number, chunk in enumerate(text_chunks, start=1):
                    # Create metadata
                    metadata = {
                        "file_path": pdf_path,
                        "file_name": os.path.basename(pdf_path),
                        "page_number": page_number,
                        "chunk_number_in_page": chunk_number,
                        "total_pages": total_pages,
                        "total_chunks_in_page": len(text_chunks),
                        "source_type": "pdf",
                    }

                    doc = Document(
                        text=chunk.text,
                        metadata=metadata,
                        excluded_llm_metadata_keys=[
                            "file_path",
                            "file_name",
                            "page_number",
                            "chunk_number_in_page",
                            "total_pages",
                            "total_chunks_in_page",
                            "source_type",
                        ],
                    )
                    documents.append(doc)
            else:
                # Create a single Document per page with all chunks
                combined_text = "\n\n".join([chunk.text for chunk in text_chunks])

                # Create metadata
                metadata = {
                    "file_path": pdf_path,
                    "file_name": os.path.basename(pdf_path),
                    "page_number": page_number,
                    "total_pages": total_pages,
                    "source_type": "pdf",
                }

                doc = Document(
                    text=combined_text,
                    metadata=metadata,
                    excluded_llm_metadata_keys=[
                        "file_path",
                        "file_name",
                        "page_number",
                        "total_pages",
                        "source_type",
                    ],
                )
                documents.append(doc)

            # Update statistics
            with ThreadPoolExecutor(max_workers=1) as executor:
                executor.submit(self._update_stats, len(text_chunks))

            return documents

        except Exception as e:
            error_msg = f"Error processing page {page_number}: {str(e)}"
            logger.error(error_msg)
            self._stats.errors.append(error_msg)

            if self.ignore_errors:
                return []
            else:
                raise RuntimeError(error_msg) from e

    def _process_pages_parallel(
        self, images: List[PIL.Image.Image], pdf_path: str
    ) -> List[Document]:
        """
        Process multiple pages in parallel.

        Args:
            images: List of page images
            pdf_path: Path to the PDF file

        Returns:
            List[Document]: Documents extracted from all pages
        """
        total_pages = len(images)

        # Create arguments for each page
        page_args = [
            (image, pdf_path, i, total_pages) for i, image in enumerate(images, start=1)
        ]

        all_documents = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_page = {
                executor.submit(self._process_page, args): args for args in page_args
            }

            for future in as_completed(future_to_page):
                try:
                    page_documents = future.result()
                    all_documents.extend(page_documents)
                except Exception as e:
                    if not self.ignore_errors:
                        raise e

        # Sort documents by page number and chunk number if needed
        if self.split_by_page:
            all_documents.sort(
                key=lambda doc: (
                    doc.metadata.get("page_number", 0),
                    doc.metadata.get("chunk_number_in_page", 0),
                )
            )
        else:
            all_documents.sort(key=lambda doc: doc.metadata.get("page_number", 0))

        return all_documents

    def _process_continuous_mode(
        self, images: List[PIL.Image.Image], pdf_path: str
    ) -> List[Document]:
        """
        Process the PDF in continuous mode with parallel extraction.

        In continuous mode, content that spans across pages is treated as continuous.

        Args:
            images: List of page images
            pdf_path: Path to the PDF file

        Returns:
            List[Document]: Documents extracted from the PDF
        """
        if self.verbose:
            logger.info(f"Processing PDF in continuous mode with {len(images)} pages")

        total_pages = len(images)

        # Create arguments for each page
        page_args = [
            (image, pdf_path, i, total_pages) for i, image in enumerate(images, start=1)
        ]

        # Process all pages in parallel to extract chunks
        all_page_chunks = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Define a helper function to process a single page and return its chunks
            def extract_page_chunks(args):
                image, pdf_path, page_num, total_pages = args

                if self.verbose:
                    logger.info(f"Processing page {page_num}/{total_pages}")

                # Extract text from the page
                text_chunks = self.process_image(image)

                # Update statistics (needs to be thread-safe)
                with threading.Lock():
                    self._stats.processed_pages += 1
                    self._stats.total_chunks += len(text_chunks)
                    self._update_progress(
                        self._stats.processed_pages, self._stats.total_pages
                    )

                # Return the chunks with their page number
                return [(page_num, chunk) for chunk in text_chunks]

            # Submit all tasks
            future_to_page = {
                executor.submit(extract_page_chunks, args): args for args in page_args
            }

            # Collect results as they complete
            for future in as_completed(future_to_page):
                try:
                    page_chunks = future.result()
                    all_page_chunks.extend(page_chunks)
                except Exception as e:
                    error_msg = f"Error in continuous mode processing: {str(e)}"
                    logger.error(error_msg)
                    self._stats.errors.append(error_msg)
                    if not self.ignore_errors:
                        raise RuntimeError(error_msg) from e

        # Sort all chunks by page number to ensure proper ordering
        all_page_chunks.sort(key=lambda x: x[0])

        # Merge chunks that might have continuous content
        documents = []
        current_text = ""
        current_pages = []

        for page_num, chunk in all_page_chunks:
            # Check if the current chunk should be merged with previous content
            is_continuation = current_text and (
                current_text.rstrip().endswith("-")  # Hyphenated word continuing
                or current_text.rstrip().endswith("\u2014")  # Em dash
                or not current_text.rstrip().endswith(".")
            )  # No period ending

            if is_continuation or not current_text:
                # Continue the current document or start a new one
                if current_text:
                    current_text += f"\n\n{chunk.text}"
                else:
                    current_text = chunk.text

                if page_num not in current_pages:
                    current_pages.append(page_num)
            else:
                # Create a document for the previous content
                if current_text:
                    metadata = {
                        "file_path": pdf_path,
                        "file_name": os.path.basename(pdf_path),
                        "page_range": f"{min(current_pages)}-{max(current_pages)}",
                        "pages": current_pages,
                        "total_pages": total_pages,
                        "source_type": "pdf_continuous",
                    }

                    doc = Document(
                        text=current_text,
                        metadata=metadata,
                        excluded_llm_metadata_keys=[
                            "file_path",
                            "file_name",
                            "page_range",
                            "pages",
                            "total_pages",
                            "source_type",
                        ],
                    )
                    documents.append(doc)

                # Start a new document with the current chunk
                current_text = chunk.text
                current_pages = [page_num]

        # Add the last document if any
        if current_text:
            metadata = {
                "file_path": pdf_path,
                "file_name": os.path.basename(pdf_path),
                "page_range": f"{min(current_pages)}-{max(current_pages)}",
                "pages": current_pages,
                "total_pages": total_pages,
                "source_type": "pdf_continuous",
            }

            doc = Document(
                text=current_text,
                metadata=metadata,
                excluded_llm_metadata_keys=[
                    "file_path",
                    "file_name",
                    "page_range",
                    "pages",
                    "total_pages",
                    "source_type",
                ],
            )
            documents.append(doc)

        return documents

    def process_pdf(self, pdf_path: str) -> List[Document]:
        """
        Process a PDF file and return a list of Document objects.

        Args:
            pdf_path (str): Path to the PDF file

        Returns:
            List[Document]: List of Document objects with extracted text

        Raises:
            FileNotFoundError: If the PDF file cannot be found
            RuntimeError: If processing fails and ignore_errors is False
        """
        # Check cache first
        if self.enable_caching:
            file_hash = self._compute_file_hash(pdf_path)
            cached_docs = self._load_from_cache(file_hash)
            if cached_docs:
                return cached_docs

        # Reset statistics
        self._stats = ProcessingStats(start_time=time.time())

        try:
            # Convert PDF to images
            images = self.convert_pdf_to_images(pdf_path)

            # Process based on mode
            if self.continuous_mode:
                documents = self._process_continuous_mode(images, pdf_path)
            else:
                documents = self._process_pages_parallel(images, pdf_path)

            # Update and finalize statistics
            self._stats.end_time = time.time()

            if self.verbose:
                logger.info(
                    f"PDF processing complete. Extracted {len(documents)} documents."
                )
                logger.info(f"Processing time: {self._stats.duration:.2f} seconds")
                logger.info(f"Pages per second: {self._stats.pages_per_second:.2f}")
                logger.info(
                    f"Average chunks per page: {self._stats.chunks_per_page:.2f}"
                )
                if self._stats.errors:
                    logger.warning(
                        f"Encountered {len(self._stats.errors)} errors during processing"
                    )

            # Cache the results if enabled
            if self.enable_caching:
                file_hash = self._compute_file_hash(pdf_path)
                self._save_to_cache(file_hash, documents, self._stats)

            return documents

        except Exception as e:
            error_msg = f"Error processing PDF {pdf_path}: {str(e)}"
            logger.error(error_msg)
            if self.ignore_errors:
                return []
            else:
                raise RuntimeError(error_msg) from e

    def _is_web_url(self, file_path: str) -> bool:
        """
        Check if the input is a web URL.

        Args:
            file_path (str): Path to check

        Returns:
            bool: True if the input is a web URL, False otherwise
        """
        try:
            from urllib.parse import urlparse

            parsed = urlparse(file_path)
            return bool(parsed.scheme in ["http", "https"] and parsed.netloc)
        except Exception:
            return False

    def _download_from_url(self, url: str) -> str:
        """
        Download a PDF from a URL.

        Args:
            url (str): URL to download from

        Returns:
            str: Path to the downloaded file
        """
        import requests
        import tempfile
        import os

        if self.verbose:
            logging.info(f"Downloading PDF from URL: {url}")

        try:
            # Create a temporary file
            temp_dir = tempfile.gettempdir()
            temp_file_path = os.path.join(
                temp_dir, f"gemini_pdf_download_{hash(url)}.pdf"
            )

            # Download the file
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()  # Raise exception for HTTP errors

            # Check content type
            content_type = response.headers.get("Content-Type", "")
            if "application/pdf" not in content_type and not url.lower().endswith(
                ".pdf"
            ):
                if self.verbose:
                    logging.warning(
                        f"URL may not be a PDF (Content-Type: {content_type})"
                    )

            # Save the file
            with open(temp_file_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            if self.verbose:
                logging.info(f"Downloaded PDF to temporary file: {temp_file_path}")

            return temp_file_path

        except Exception as e:
            if self.verbose:
                logging.error(f"Error downloading from URL {url}: {str(e)}")
            raise

    def load_data(
        self,
        file_path: Union[FileInput, List[FileInput]],
        extra_info: Optional[Dict] = None,
    ) -> List[Document]:
        """
        Load data from the input file(s).

        Args:
            file_path (Union[FileInput, List[FileInput]]): Path to PDF file(s), list of paths, or web URL(s)
            extra_info (Optional[Dict]): Extra information to add to document metadata

        Returns:
            List[Document]: List of documents with extracted text
        """
        metadata = extra_info or {}

        # Handle list of files/URLs
        if isinstance(file_path, list):
            all_documents = []
            for i, single_path in enumerate(file_path):
                if self.verbose:
                    logging.info(f"Processing file {i+1}/{len(file_path)}")

                try:
                    file_docs = self.load_data(single_path, extra_info)
                    all_documents.extend(file_docs)
                except Exception as e:
                    if self.verbose:
                        logging.error(
                            f"Error processing file {i+1}/{len(file_path)}: {str(e)}"
                        )
                    if not self.ignore_errors:
                        raise

            return all_documents

        # Handle different input types for a single file
        if isinstance(file_path, (str, Path)):
            file_str = str(file_path)

            # Check if the input is a web URL
            if isinstance(file_path, str) and self._is_web_url(file_path):
                try:
                    # Download the PDF from the URL
                    temp_path = self._download_from_url(file_path)

                    # Process the downloaded PDF
                    documents = self.process_pdf(temp_path)

                    # Add source URL to metadata
                    for doc in documents:
                        doc.metadata["source_url"] = file_path
                        doc.metadata["source_type"] = "web"
                        if metadata:
                            doc.metadata.update(metadata)

                    # Clean up
                    try:
                        os.unlink(temp_path)
                    except Exception:
                        pass

                    return documents
                except Exception as e:
                    if self.verbose:
                        logging.error(
                            f"Error loading data from URL {file_path}: {str(e)}"
                        )
                    if self.ignore_errors:
                        return []
                    else:
                        raise

            # Regular file path
            try:
                documents = self.process_pdf(file_str)

                # Add extra metadata if provided
                if metadata:
                    for doc in documents:
                        doc.metadata.update(metadata)

                return documents
            except Exception as e:
                if self.verbose:
                    logging.error(f"Error loading data from {file_str}: {str(e)}")
                if self.ignore_errors:
                    return []
                else:
                    raise

        elif isinstance(file_path, (bytes, BufferedIOBase)):
            # For bytes or buffer, save to a temporary file first
            try:
                with tempfile.NamedTemporaryFile(
                    suffix=".pdf", delete=False
                ) as temp_file:
                    temp_path = temp_file.name

                    if isinstance(file_path, bytes):
                        temp_file.write(file_path)
                    else:
                        # It's a buffer
                        temp_file.write(file_path.read())

                documents = self.process_pdf(temp_path)

                # Add file source to metadata
                for doc in documents:
                    doc.metadata["source_type"] = "buffer"
                    if metadata:
                        doc.metadata.update(metadata)

                # Clean up
                os.unlink(temp_path)

                return documents

            except Exception as e:
                if self.verbose:
                    logging.error(f"Error loading data from buffer: {str(e)}")
                if self.ignore_errors:
                    return []
                else:
                    raise
        else:
            raise ValueError(
                "file_path must be a string, Path, bytes, BufferedIOBase, web URL, or a list of these types"
            )

    def get_processing_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the last PDF processing operation.

        Returns:
            Dict[str, Any]: Dictionary of processing statistics.
        """
        stats = {
            "duration_seconds": self._stats.duration,
            "total_pages": self._stats.total_pages,
            "processed_pages": self._stats.processed_pages,
            "total_chunks": self._stats.total_chunks,
            "pages_per_second": self._stats.pages_per_second,
            "chunks_per_page": self._stats.chunks_per_page,
            "error_count": len(self._stats.errors),
        }

        if self._stats.errors:
            stats["errors"] = self._stats.errors[:10]  # Limit to first 10 errors

        return stats
