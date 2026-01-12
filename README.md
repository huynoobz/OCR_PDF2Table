# OCR PDF to Table - Image Processing System

A two-module system for processing PDF files into images with advanced image processing capabilities and a user interface for inspection and manipulation.

## Modules

### Module 1: Input and Image Processing
Pure logic module for PDF to image conversion and image processing pipeline.

### Module 2: User Interface
GUI module for visual inspection, manipulation, and export of processed images.

## Installation

```bash
pip install -r requirements.txt
```

**Note:** For `pdf2image`, you may need to install `poppler`:
- Windows: Download from [poppler-windows](https://github.com/oschwartz10612/poppler-windows/releases/)
- Linux: `sudo apt-get install poppler-utils`
- macOS: `brew install poppler`

## Usage

```python
from module1_image_processing import PDFImageProcessor

processor = PDFImageProcessor()
images = processor.process_pdf("path/to/file.pdf", dpi=300)
```

Then use Module 2 to view and manipulate the images.
