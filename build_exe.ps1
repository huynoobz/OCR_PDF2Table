$ErrorActionPreference = "Stop"

Write-Host "Building one-file EXE with PyInstaller..."
Write-Host "Project: OCR_PDF2Table"

# Ensure dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install pyinstaller

# Clean previous builds
if (Test-Path ".\\build") { Remove-Item ".\\build" -Recurse -Force }
if (Test-Path ".\\dist") { Remove-Item ".\\dist" -Recurse -Force }

# Build
python -m PyInstaller `
  --noconfirm `
  --clean `
  --onefile `
  --windowed `
  --name "OCR_PDF2Table" `
  main.py

Write-Host ""
Write-Host "Done. EXE is at: .\\dist\\OCR_PDF2Table.exe"
Write-Host ""
Write-Host "Notes:"
Write-Host "- For PDF loading on Windows, Poppler is required (system-installed or vendor\\poppler\\...)."
Write-Host "- For OCR, Tesseract is required (system-installed or vendor\\tesseract\\...)."

