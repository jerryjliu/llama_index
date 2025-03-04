"""Docs parser.

Contains parsers for docx, pdf files.

"""

import io
import logging
import struct
import zlib
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional

from tenacity import retry, stop_after_attempt

from fsspec import AbstractFileSystem

from llama_index.core.readers.base import BaseReader
from llama_index.core.readers.file.base import get_default_fs, is_default_fs
from llama_index.core.schema import Document

logger = logging.getLogger(__name__)

RETRY_TIMES = 3


class PDFReader(BaseReader):
    """PDF parser."""

    def __init__(self, return_full_document: Optional[bool] = False) -> None:
        """
        Initialize PDFReader.
        """
        self.return_full_document = return_full_document

    @retry(
        stop=stop_after_attempt(RETRY_TIMES),
    )
    def load_data(
        self,
        file: Path,
        extra_info: Optional[Dict] = None,
        fs: Optional[AbstractFileSystem] = None,
    ) -> List[Document]:
        """Parse file."""
        if not isinstance(file, Path) and not isinstance(file, PurePosixPath):
            file = PurePosixPath(file)

        try:
            import pypdf
        except ImportError:
            raise ImportError(
                "pypdf is required to read PDF files: `pip install pypdf`"
            )
        fs = fs or get_default_fs()
        with fs.open(str(file), "rb") as fp:
            # Load the file in memory if the filesystem is not the default one to avoid
            # issues with pypdf
            stream = fp if is_default_fs(fs) else io.BytesIO(fp.read())

            # Create a PDF object
            pdf = pypdf.PdfReader(stream)

            # Get the number of pages in the PDF document
            num_pages = len(pdf.pages)

            docs = []

            # This block returns a whole PDF as a single Document
            if self.return_full_document:
                metadata = {"file_name": file.name}
                if extra_info is not None:
                    metadata.update(extra_info)

                # Join text extracted from each page
                text = "\n".join(
                    pdf.pages[page].extract_text() for page in range(num_pages)
                )

                docs.append(Document(text=text, metadata=metadata))

            # This block returns each page of a PDF as its own Document
            else:
                # Iterate over every page

                for page in range(num_pages):
                    # Extract the text from the page
                    page_text = pdf.pages[page].extract_text()
                    page_label = pdf.page_labels[page]

                    metadata = {"page_label": page_label, "file_name": file.name}
                    if extra_info is not None:
                        metadata.update(extra_info)

                    docs.append(Document(text=page_text, metadata=metadata))

            return docs


class DocxReader(BaseReader):
    """Docx parser."""

    def load_data(
        self,
        file: Path,
        extra_info: Optional[Dict] = None,
        fs: Optional[AbstractFileSystem] = None,
    ) -> List[Document]:
        """Parse file."""
        if not isinstance(file, Path):
            file = Path(file)

        try:
            import docx2txt
        except ImportError:
            raise ImportError(
                "docx2txt is required to read Microsoft Word files: "
                "`pip install docx2txt`"
            )

        if fs:
            with fs.open(str(file)) as f:
                text = docx2txt.process(f)
        else:
            text = docx2txt.process(file)
        metadata = {"file_name": file.name}
        if extra_info is not None:
            metadata.update(extra_info)

        return [Document(text=text, metadata=metadata or {})]


class HWPReader(BaseReader):
    """Hwp Parser."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.FILE_HEADER_SECTION = "FileHeader"
        self.HWP_SUMMARY_SECTION = "\x05HwpSummaryInformation"
        self.SECTION_NAME_LENGTH = len("Section")
        self.BODYTEXT_SECTION = "BodyText"
        self.HWP_TEXT_TAGS = [67]
        self.text = ""

    def load_data(
        self,
        file: Path,
        extra_info: Optional[Dict] = None,
        fs: Optional[AbstractFileSystem] = None,
    ) -> List[Document]:
        """Load data and extract table from Hwp file.

        Args:
            file (Path): Path for the Hwp file.

        Returns:
            List[Document]
        """
        import olefile

        if fs:
            logger.warning(
                "fs was specified but HWPReader doesn't support loading "
                "from fsspec filesystems. Will load from local filesystem instead."
            )

        if not isinstance(file, Path):
            file = Path(file)
        load_file = olefile.OleFileIO(file)
        file_dir = load_file.listdir()
        if self.is_valid(file_dir) is False:
            raise Exception("Not Valid HwpFile")

        result_text = self._get_text(load_file, file_dir)
        result = self._text_to_document(text=result_text, extra_info=extra_info)
        return [result]

    def is_valid(self, dirs: List[str]) -> bool:
        if [self.FILE_HEADER_SECTION] not in dirs:
            return False

        return [self.HWP_SUMMARY_SECTION] in dirs

    def get_body_sections(self, dirs: List[str]) -> List[str]:
        m = []
        for d in dirs:
            if d[0] == self.BODYTEXT_SECTION:
                m.append(int(d[1][self.SECTION_NAME_LENGTH :]))

        return ["BodyText/Section" + str(x) for x in sorted(m)]

    def _text_to_document(
        self, text: str, extra_info: Optional[Dict] = None
    ) -> Document:
        return Document(text=text, extra_info=extra_info or {})

    def get_text(self) -> str:
        return self.text

        # 전체 text 추출

    def _get_text(self, load_file: Any, file_dirs: List[str]) -> str:
        sections = self.get_body_sections(file_dirs)
        text = ""
        for section in sections:
            text += self.get_text_from_section(load_file, section)
            text += "\n"

        self.text = text
        return self.text

    def is_compressed(self, load_file: Any) -> bool:
        header = load_file.openstream("FileHeader")
        header_data = header.read()
        return (header_data[36] & 1) == 1

    def get_text_from_section(self, load_file: Any, section: str) -> str:
        bodytext = load_file.openstream(section)
        data = bodytext.read()

        unpacked_data = (
            zlib.decompress(data, -15) if self.is_compressed(load_file) else data
        )
        size = len(unpacked_data)

        i = 0

        text = ""
        while i < size:
            header = struct.unpack_from("<I", unpacked_data, i)[0]
            rec_type = header & 0x3FF
            (header >> 10) & 0x3FF
            rec_len = (header >> 20) & 0xFFF

            if rec_type in self.HWP_TEXT_TAGS:
                rec_data = unpacked_data[i + 4 : i + 4 + rec_len]
                text += rec_data.decode("utf-16")
                text += "\n"

            i += 4 + rec_len

        return text
