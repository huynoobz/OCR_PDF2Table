# OCR PDF to Table - Image Processing System

A two-module system for processing PDF files into images with advanced image processing capabilities and a user interface for inspection and manipulation.

## Modules

### Module 1: Input and Image Processing
Pure logic module for PDF to image conversion and image processing pipeline.

### Module 2: User Interface
GUI module for visual inspection, manipulation, and export of processed images.

## Installation

### 1) Python dependencies

```bash
pip install -r requirements.txt
```

### 2) Poppler (required for PDF → images)

`pdf2image` relies on Poppler on Windows (and commonly on other platforms) to rasterize PDFs.

- Windows: Download from [poppler-windows](https://github.com/oschwartz10612/poppler-windows/releases/)
  - After download, either add Poppler `bin` to your PATH, or use the portable EXE layout described below (`vendor\poppler\...`)
- Linux: `sudo apt-get install poppler-utils`
- macOS: `brew install poppler`

### 3) Tesseract OCR (required for OCR)

`pytesseract` is a Python wrapper, but you still must install **Tesseract OCR** itself.

- Windows:
  - Install Tesseract (a common choice is the UB-Mannheim Windows installer).
  - Ensure `tesseract.exe` is on PATH (or use the portable EXE layout described below: `vendor\tesseract\...`).
- Linux:
  - `sudo apt-get install tesseract-ocr`
- macOS:
  - `brew install tesseract`

### 4) Install OCR language traineddata

Tesseract uses `*.traineddata` language packs (e.g. `eng.traineddata`, `vie.traineddata`).

- Windows:
  - Find your `tessdata` folder (typical locations):
    - `C:\Program Files\Tesseract-OCR\tessdata`
    - or if using portable layout: `vendor\tesseract\tessdata`
  - Download the languages you need and put the `*.traineddata` files into that `tessdata` folder.
  - If Tesseract can’t find languages, set an environment variable:
    - `TESSDATA_PREFIX` = path to the `tessdata` folder
- Linux:
  - Install languages via packages (example):
    - `sudo apt-get install tesseract-ocr-eng tesseract-ocr-vie`
- macOS:
  - Homebrew usually ships English; for more languages, install additional language files or point `TESSDATA_PREFIX` to your `tessdata`.

Language data downloads (official repo):
- Best-trained models: `https://github.com/tesseract-ocr/tessdata_best`
- Fast models: `https://github.com/tesseract-ocr/tessdata_fast`

In the app, set the OCR language in **Settings → OCR language** (example values: `eng`, `vie`, or multi-lang like `eng+vie`).

## Set up a dev environment (recommended)

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
python main.py ui
```

### macOS / Linux (bash/zsh)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt
python main.py ui
```

If you get PDF loading errors, verify Poppler is installed and available. If OCR fails, verify `tesseract` is installed and that your selected `ocr_lang` exists in `tessdata`.

## Build a single-file EXE (Windows)

This project can be packaged into **one** `OCR_PDF2Table.exe` using PyInstaller.

1. In PowerShell, from the project folder, run:

```powershell
.\build_exe.ps1
```

2. Output:
- `dist\OCR_PDF2Table.exe`

### Optional: bundle Poppler/Tesseract next to the EXE (portable)

If you want the EXE to work on another PC without installing Poppler/Tesseract globally,
place them next to the EXE in a `vendor\` folder:

- **Poppler**:
  - `vendor\poppler\bin\...`  (or `vendor\poppler\Library\bin\...`)
- **Tesseract**:
  - `vendor\tesseract\tesseract.exe`
  - `vendor\tesseract\tessdata\...`

The app auto-detects these paths at runtime.

## Usage

```python
from module1_image_processing import PDFImageProcessor

processor = PDFImageProcessor()
images = processor.process_pdf("path/to/file.pdf", dpi=300)
```

Then use Module 2 to view and manipulate the images.
