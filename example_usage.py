"""
Auto test script for Module 1.

This script is meant to be run non-interactively using a local `test.pdf`.
It converts the PDF into per-page images, runs the processing pipeline, and
exports images + a short report.

Usage:
  python example_usage.py

Expected:
  - A file named `test.pdf` in the project root
  - An `auto_test_output/` folder will be created
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import asdict

from module1_image_processing import PDFImageProcessor


def run_auto_test(
    pdf_path: str = "test.pdf",
    output_dir: str = "auto_test_output",
    dpi: int = 300,
) -> int:
    """
    Run a simple end-to-end test of Module 1 using `test.pdf`.

    Returns:
        0 on success, non-zero on failure.
    """
    abs_pdf = os.path.abspath(pdf_path)
    abs_out = os.path.abspath(output_dir)

    print("=" * 60)
    print("AUTO TEST: Module 1 (PDF -> processed images)")
    print(f"PDF: {abs_pdf}")
    print(f"OUT: {abs_out}")
    print("=" * 60)

    if not os.path.exists(abs_pdf):
        print(f"[FAIL] Missing file: {abs_pdf}")
        print("Place `test.pdf` in the project root and re-run.")
        return 2

    os.makedirs(abs_out, exist_ok=True)

    processor = PDFImageProcessor()

    # Example per-page config: rotate page 1 by 0 degrees (no-op) by default.
    page_rotations = {}
    crop_boxes = {}

    start = time.time()
    try:
        images = processor.process_pdf(
            pdf_path=abs_pdf,
            dpi=dpi,
            page_rotations=page_rotations,
            crop_boxes=crop_boxes,
            apply_grayscale=True,
            apply_denoise=True,
            apply_deskew=True,
            apply_contrast=True,
            contrast_factor=1.2,
        )
    except Exception as e:
        print(f"[FAIL] Exception while processing PDF: {e}")
        return 3

    elapsed = time.time() - start

    if not images:
        print("[FAIL] No pages/images were produced.")
        return 4

    # Export images + report
    report_lines: list[str] = []
    report_lines.append(f"pdf={abs_pdf}")
    report_lines.append(f"dpi={dpi}")
    report_lines.append(f"pages={len(images)}")
    report_lines.append(f"elapsed_seconds={elapsed:.3f}")
    report_lines.append("")

    for item in images:
        meta = item.metadata
        filename = f"page_{meta.page_number:03d}.png"
        out_path = os.path.join(abs_out, filename)
        item.pil_image.save(out_path, "PNG")

        report_lines.append(f"[page {meta.page_number}] saved={filename}")
        report_lines.append(f"  original_size={meta.original_size}")
        report_lines.append(f"  processed_size={meta.processed_size}")
        report_lines.append(f"  applied_operations={meta.applied_operations}")
        report_lines.append("")

        # Basic sanity checks
        if meta.page_number <= 0:
            print("[FAIL] Invalid page_number in metadata.")
            return 5
        if meta.processed_size[0] <= 0 or meta.processed_size[1] <= 0:
            print("[FAIL] Invalid processed_size in metadata.")
            return 6

    # Write metadata json-ish dump for debugging
    meta_dump_path = os.path.join(abs_out, "metadata.txt")
    with open(meta_dump_path, "w", encoding="utf-8") as f:
        for item in images:
            f.write(str(asdict(item.metadata)) + "\n")

    report_path = os.path.join(abs_out, "report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines).strip() + "\n")

    print(f"[OK] Processed pages: {len(images)} in {elapsed:.2f}s")
    print(f"[OK] Exported PNGs + report to: {abs_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_auto_test())
