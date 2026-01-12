"""
Project entrypoint.

Usage:
  python main.py            # Launch UI
  python main.py ui         # Launch UI
  python main.py test       # Run Module 1 auto-test (uses ./test.pdf)
  python main.py test --pdf path/to/file.pdf --out auto_test_output --dpi 300
"""

from __future__ import annotations

import argparse
import pathlib
import sys


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="main.py")
    sub = p.add_subparsers(dest="command")

    ui = sub.add_parser("ui", help="Launch the image review UI (Module 2)")
    ui.set_defaults(command="ui")

    test = sub.add_parser("test", help="Run Module 1 auto-test using a PDF")
    test.add_argument("--pdf", default="test.pdf", help="Path to PDF (default: test.pdf)")
    test.add_argument("--out", default="auto_test_output", help="Output directory (default: auto_test_output)")
    test.add_argument("--dpi", type=int, default=300, help="DPI for PDF rasterization (default: 300)")
    test.set_defaults(command="test")

    return p


def main(argv: list[str] | None = None) -> int:
    # Ensure running from any working directory still imports local modules.
    project_root = str(pathlib.Path(__file__).resolve().parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    args = _build_parser().parse_args(argv)

    # Default: launch UI
    cmd = args.command or "ui"

    if cmd == "ui":
        # Import lazily so headless runs (like test) don't require tkinter.
        from module2_user_interface import main as ui_main

        ui_main()
        return 0

    if cmd == "test":
        from auto_test import run_auto_test

        return int(run_auto_test(pdf_path=args.pdf, output_dir=args.out, dpi=args.dpi))

    return 2


if __name__ == "__main__":
    raise SystemExit(main())

