"""Slides parser.

Contains parsers for .pptx files.

"""

import os
import sys
import shutil
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
from fsspec import AbstractFileSystem

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document
from llama_index.core.utils import infer_torch_device


class PptxReader(BaseReader):
    """Powerpoint parser.

    Extract text, caption images, and specify slides.

    """

    def __init__(self) -> None:
        """Init parser."""
        try:
            import torch  # noqa
            from PIL import Image  # noqa
            from pptx import Presentation  # noqa
            from transformers import (
                AutoTokenizer,
                VisionEncoderDecoderModel,
                ViTFeatureExtractor,
            )
        except ImportError:
            raise ImportError(
                "Please install extra dependencies that are required for "
                "the PptxReader: "
                "`pip install torch transformers python-pptx Pillow`"
            )

        model = VisionEncoderDecoderModel.from_pretrained(
            "nlpconnect/vit-gpt2-image-captioning"
        )
        feature_extractor = ViTFeatureExtractor.from_pretrained(
            "nlpconnect/vit-gpt2-image-captioning"
        )
        tokenizer = AutoTokenizer.from_pretrained(
            "nlpconnect/vit-gpt2-image-captioning"
        )

        self.parser_config = {
            "feature_extractor": feature_extractor,
            "model": model,
            "tokenizer": tokenizer,
        }

    def find_libreoffice(self) -> str:
        """Finds the LibreOffice executable path."""
        libreoffice_path = shutil.which("soffice")

        if not libreoffice_path and sys.platform == "win32":
            # Check common installation paths on Windows
            possible_paths = [
                r"C:\Program Files\LibreOffice\program\soffice.exe",
                r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
            ]
            libreoffice_path = next(
                (path for path in possible_paths if os.path.exists(path)), None
            )

        if not libreoffice_path:
            raise OSError(
                "LibreOffice (soffice) not found. Please install LibreOffice or add it to your system PATH."
            )

        return libreoffice_path

    def convert_wmf_to_png(self, input_path: str) -> str:
        """Convert WMF/EMF to PNG using LibreOffice."""
        file_path = Path(input_path)
        output_path = file_path.with_suffix(".png")

        libreoffice_path = self.find_libreoffice()

        subprocess.run(
            [
                libreoffice_path,
                "--headless",
                "--convert-to",
                "png",
                "--outdir",
                str(file_path.parent),
                str(file_path),
            ],
            check=True,
        )

        return str(output_path)

    def caption_image(self, tmp_image_file: str) -> str:
        """Generate text caption of image."""
        from PIL import Image, UnidentifiedImageError

        model = self.parser_config["model"]
        feature_extractor = self.parser_config["feature_extractor"]
        tokenizer = self.parser_config["tokenizer"]

        device = infer_torch_device()
        model.to(device)

        max_length = 16
        num_beams = 4
        gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

        try:
            i_image = Image.open(tmp_image_file)
            image_format = i_image.format
        except UnidentifiedImageError:
            return "Error opening image file."

        if image_format in ["WMF", "EMF"]:
            try:
                converted_path = self.convert_wmf_to_png(tmp_image_file)
                i_image = Image.open(converted_path)
            except Exception as e:
                print(f"Error converting WMF/EMF image: {e}")
                return f"Error converting WMF/EMF image"

        if i_image.mode != "RGB":
            i_image = i_image.convert(mode="RGB")

        pixel_values = feature_extractor(
            images=[i_image], return_tensors="pt"
        ).pixel_values
        pixel_values = pixel_values.to(device)

        output_ids = model.generate(pixel_values, **gen_kwargs)

        preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        return preds[0].strip()

    def load_data(
        self,
        file: Path,
        extra_info: Optional[Dict] = None,
        fs: Optional[AbstractFileSystem] = None,
    ) -> List[Document]:
        """Parse file."""
        from pptx import Presentation

        if fs:
            with fs.open(file) as f:
                presentation = Presentation(f)
        else:
            presentation = Presentation(file)
        result = ""
        for i, slide in enumerate(presentation.slides):
            result += f"\n\nSlide #{i}: \n"
            for shape in slide.shapes:
                if hasattr(shape, "image"):
                    image = shape.image
                    # get image "file" contents
                    image_bytes = image.blob
                    # temporarily save the image to feed into model
                    f = tempfile.NamedTemporaryFile("wb", delete=False)
                    try:
                        f.write(image_bytes)
                        f.close()
                        result += f"\n Image: {self.caption_image(f.name)}\n\n"
                    finally:
                        os.unlink(f.name)

                if hasattr(shape, "text"):
                    result += f"{shape.text}\n"

        return [Document(text=result, metadata=extra_info or {})]
