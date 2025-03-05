# GeminiReader

> A high-performance PDF extractor and chunker powered by Google's Gemini AI

GeminiReader is a library that leverages Google's Gemini AI for accurate PDF text extraction, OCR, and intelligent document chunking. Built as an extension for LlamaIndex, it provides superior extraction quality compared to traditional PDF parsers, especially for complex layouts, forms, and tables.

## Features

- **AI-powered OCR**: Uses Gemini models to extract text with high accuracy, even from complex layouts, scanned documents, and images
- **Intelligent Chunking**: Creates semantically meaningful chunks that preserve document structure and context
- **Table Extraction**: Accurately extracts and formats tables using markdown formatting
- **Form Recognition**: Special handling for forms, checkboxes, and structured documents
- **Math Formula Support**: Extracts and formats mathematical formulas properly
- **Multilingual Support**: Optimized for multiple languages beyond English
- **Parallel Processing**: Efficiently processes multi-page documents with configurable parallelism
- **Continuous Mode**: Option to parse documents continuously for better handling of content that spans multiple pages
- **Caching System**: Built-in caching to avoid redundant processing of the same documents
- **Progress Tracking**: Provides detailed statistics and progress tracking during processing
- **Configurable**: Extensive configuration options to customize behavior

## Installation

```bash
pip install llama-index-readers-gemini
```

### Requirements

- Python 3.10+
- Google API Key with access to Gemini models
- Required dependencies: LlamaIndex, pdf2image, Pillow, poppler, and Google AI SDK

For pdf2image to work properly, you need to install the poppler-utils:

**Ubuntu/Debian**:
```bash
apt-get install -y poppler-utils
```

**macOS**:
```bash
brew install poppler
```

**Windows**:
Poppler for Windows binaries can be downloaded [here](https://github.com/oschwartz10612/poppler-windows/releases/).

## Usage

### Basic Usage

```python
from llama_index.readers.gemini import GeminiReader
from llama_index.core import VectorStoreIndex, ServiceContext

# Initialize with your Google API key
reader = GeminiReader(
    api_key="your_google_api_key",  # Or set GOOGLE_API_KEY environment variable
    model_name="gemini-2.0-flash",
    verbose=True
)

# Load a single PDF
documents = reader.load_data("path/to/your/document.pdf")

# Process multiple PDFs
documents = reader.load_data([
    "path/to/document1.pdf",
    "path/to/document2.pdf",
    "https://example.com/document3.pdf"  # Also supports URLs
])

# Use with LlamaIndex
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()
response = query_engine.query("What is this document about?")
print(response)
```

### Processing Statistics

```python
# Get detailed processing statistics
stats = reader.get_processing_stats()
print(f"Processed {stats['processed_pages']} pages in {stats['duration_seconds']:.2f} seconds")
print(f"Average processing speed: {stats['pages_per_second']:.2f} pages/second")
print(f"Extracted {stats['total_chunks']} chunks, {stats['chunks_per_page']:.2f} chunks/page")
```

### Progress Tracking

```python
def progress_callback(current, total):
    progress = (current / total) * 100
    print(f"Processing: {progress:.2f}% ({current}/{total} pages)")

reader = GeminiReader(
    api_key="your_google_api_key",
    progress_callback=progress_callback
)
```

## Configuration Options

GeminiReader provides numerous configuration options:

### API Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_key` | str | None | The API key for Google's Gemini API. If not provided, will try to use GOOGLE_API_KEY environment variable. |
| `model_name` | str | "gemini-2.0-flash" | The model name to use for OCR and chunking. |

### Processing Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `split_by_page` | bool | True | Whether to split the document by page. |
| `verbose` | bool | False | Whether to print progress information. |
| `ignore_errors` | bool | True | Whether to ignore errors and continue processing. |
| `dpi` | int | 300 | DPI for PDF to image conversion. |
| `language` | str | "en" | Primary language of the documents. |
| `max_workers` | int | 4 | Maximum number of workers for parallel processing (1-10). |
| `continuous_mode` | bool | False | Parse documents continuously for better results with content spanning multiple pages. |
| `chunk_size` | str, int | "256-512" | Target size range for chunks in words. |

### Extraction Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `extract_forms` | bool | True | Whether to apply special handling for forms. |
| `extract_tables` | bool | True | Whether to extract and format tables. |
| `extract_math_formulas` | bool | True | Whether to extract and format mathematical formulas. |

### Caching Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_caching` | bool | True | Whether to cache processed results. |
| `cache_dir` | str | temp directory | Directory for caching processed results. |
| `cache_ttl` | int | 86400 (24h) | Time-to-live for cache entries in seconds. |

### API Retry Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_retries` | int | 3 | Maximum number of retries for API calls. |
| `retry_delay` | float | 1.0 | Delay between retries in seconds. |

## Advanced Usage

### Continuous Mode

For documents where content flows between pages (like tables spanning multiple pages):

```python
reader = GeminiReader(
    api_key="your_google_api_key",
    continuous_mode=True
)
```

### Custom Chunk Sizes

```python
# For smaller chunks (better for retrieval)
reader = GeminiReader(
    api_key="your_google_api_key",
    chunk_size="128-256"
)

# For larger chunks (better for context)
reader = GeminiReader(
    api_key="your_google_api_key",
    chunk_size="512-1024"
)
```

## Performance Considerations

- **DPI Setting**: Higher DPI values provide better extraction quality but increase processing time and memory usage.
- **Parallelism**: Adjust `max_workers` based on your machine's capabilities.
- **Model Selection**: "gemini-2.0-flash" provides a good balance of speed and quality.
- **Caching**: Enable caching to avoid redundant processing of the same documents.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This project uses Google's Gemini API for AI-powered text extraction and document understanding.
- Built as an extension for the LlamaIndex framework.