# OCR_PDF2Table

Desktop tool to **extract tables from PDFs/images** using:
- **Table detection** (OpenCV) → cell grid
- **OCR** (Tesseract via `pytesseract`) → per-cell text
- **CSV export** (current page / all pages / selected pages via wizard)

## Quick start (run the UI)

### Run from Python (dev)

```bash
python main.py ui
```

### Run the EXE (Windows)

After building, run:
- `dist\OCR_PDF2Table.exe`

## Features (high level)

- **Detect table**: finds table lines + cells and lets you show masks
- **OCR selected cells**: right-click a cell to OCR/export text (after Detect table)
- **Export table to CSV**:
  - current page
  - all pages merged
  - **wizard**: choose pages like `1,2` or `1-5`

## Installation (prerequisites)

### 1) Python dependencies

```bash
pip install -r requirements.txt
```

### 2) Poppler (required for PDF → images)

`pdf2image` needs Poppler to rasterize PDF pages.

- **Windows**: download Poppler from [poppler-windows releases](https://github.com/oschwartz10612/poppler-windows/releases/)
  - Add the Poppler `bin` folder to your PATH **or** use the portable EXE layout below (`vendor\poppler\...`).
- **Linux**: `sudo apt-get install poppler-utils`
- **macOS**: `brew install poppler`

### 3) Tesseract OCR (required for OCR)

`pytesseract` is only a wrapper. You must install **Tesseract**.

- **Windows**: install Tesseract (commonly the “UB Mannheim” installer) and ensure `tesseract.exe` is on PATH (or use portable layout: `vendor\tesseract\...`).
- **Linux**: `sudo apt-get install tesseract-ocr`
- **macOS**: `brew install tesseract`

### 4) Install OCR language traineddata

Tesseract uses `*.traineddata` files (example: `eng.traineddata`, `vie.traineddata`).

- **Download**:
  - Best quality: `https://github.com/tesseract-ocr/tessdata_best`
  - Faster: `https://github.com/tesseract-ocr/tessdata_fast`

- **Where to put them**:
  - **Windows installed Tesseract**: usually `C:\Program Files\Tesseract-OCR\tessdata`
  - **Portable layout**: `vendor\tesseract\tessdata`

- **If languages are not found**: set `TESSDATA_PREFIX` to the **folder that contains** `tessdata`.

In the app, set OCR language in **Settings → OCR language**:
- Single language: `eng`, `vie`, `jpn`
- Multi-language: `eng+vie`

## Set up a dev environment

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

## Export table wizard (page selection syntax)

Open **File → Export table (CSV) – wizard…** and enter:
- `1,2` = pages 1 and 2
- `1-5` = pages 1 to 5
- `1-3,7,10-12` = ranges + single pages
- `all` = all pages

Pages are **1-based** in the wizard.

## Build a single-file EXE (Windows)

```powershell
.\build_exe.ps1
```

Output:
- `dist\OCR_PDF2Table.exe`

### Portable mode (bundle Poppler/Tesseract next to the EXE)

If you want the EXE to run on another PC without installing Poppler/Tesseract globally,
create a `vendor\` folder next to the EXE:

- **Poppler**:
  - `vendor\poppler\bin\...` *(or `vendor\poppler\Library\bin\...`)*
- **Tesseract**:
  - `vendor\tesseract\tesseract.exe`
  - `vendor\tesseract\tessdata\...`

The app auto-detects these paths.

## CLI usage (optional)

```bash
python main.py ui
python main.py test --pdf test.pdf --out auto_test_output --dpi 300
```

## Troubleshooting

- **PDF load fails (Windows)**: Poppler not installed or not found.
  - Fix: add Poppler `bin` to PATH or place it in `vendor\poppler\bin` next to the EXE.
- **OCR fails / language not found**: Tesseract or traineddata missing.
  - Fix: install Tesseract, ensure `tesseract.exe` is on PATH, and put `*.traineddata` into `tessdata`.
  - If needed set: `TESSDATA_PREFIX`.
