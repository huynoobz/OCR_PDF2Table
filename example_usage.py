"""
Example usage of Module 1 and Module 2.
Demonstrates how to use the image processing module and UI module.
"""

from module1_image_processing import PDFImageProcessor
from module2_user_interface import ImageManagementUI
import tkinter as tk


def example_module1_standalone():
    """Example of using Module 1 standalone (without UI)."""
    print("Example: Using Module 1 standalone")
    print("-" * 50)
    
    # Initialize processor
    processor = PDFImageProcessor()
    
    # Process PDF with custom settings
    pdf_path = "example.pdf"  # Replace with your PDF path
    
    try:
        # Process with default settings
        images = processor.process_pdf(
            pdf_path=pdf_path,
            dpi=300,
            apply_grayscale=True,
            apply_denoise=True,
            apply_deskew=True,
            apply_contrast=True,
            contrast_factor=1.2
        )
        
        print(f"Processed {len(images)} pages")
        
        # Access processed images and metadata
        for processed in images:
            meta = processed.metadata
            print(f"\nPage {meta.page_number}:")
            print(f"  Original size: {meta.original_size}")
            print(f"  Processed size: {meta.processed_size}")
            print(f"  Operations: {', '.join(meta.applied_operations)}")
            
            # Access image as numpy array
            img_array = processed.image
            
            # Access image as PIL Image
            pil_img = processed.pil_image
            
            # Save if needed
            # pil_img.save(f"output_page_{meta.page_number}.png")
        
    except FileNotFoundError:
        print(f"PDF file not found: {pdf_path}")
        print("Please update the pdf_path variable with a valid PDF file.")
    except Exception as e:
        print(f"Error: {str(e)}")


def example_module1_with_custom_settings():
    """Example of using Module 1 with custom per-page settings."""
    print("\nExample: Using Module 1 with custom settings")
    print("-" * 50)
    
    processor = PDFImageProcessor()
    pdf_path = "example.pdf"  # Replace with your PDF path
    
    try:
        # Custom rotation for specific pages
        page_rotations = {
            1: 90,   # Rotate page 1 by 90 degrees
            3: 180,  # Rotate page 3 by 180 degrees
        }
        
        # Custom crop boxes for specific pages
        crop_boxes = {
            2: (100, 100, 800, 1000),  # Crop page 2
        }
        
        images = processor.process_pdf(
            pdf_path=pdf_path,
            dpi=300,
            page_rotations=page_rotations,
            crop_boxes=crop_boxes,
            apply_grayscale=True,
            apply_denoise=True,
            apply_deskew=True,
            apply_contrast=True
        )
        
        print(f"Processed {len(images)} pages with custom settings")
        
    except FileNotFoundError:
        print(f"PDF file not found: {pdf_path}")
    except Exception as e:
        print(f"Error: {str(e)}")


def example_module2_ui():
    """Example of launching the UI module."""
    print("\nExample: Launching Module 2 UI")
    print("-" * 50)
    print("The UI window will open. You can:")
    print("  - Load PDF files directly from the UI")
    print("  - Add external image files")
    print("  - View, zoom, and pan images")
    print("  - Apply non-destructive edits (rotate, crop, brightness, contrast)")
    print("  - Reorder and delete images")
    print("  - Export selected images")
    
    root = tk.Tk()
    app = ImageManagementUI(root)
    root.mainloop()


def example_integration():
    """Example of integrating Module 1 and Module 2."""
    print("\nExample: Integrating Module 1 and Module 2")
    print("-" * 50)
    
    # Process PDF with Module 1
    processor = PDFImageProcessor()
    pdf_path = "example.pdf"  # Replace with your PDF path
    
    try:
        images = processor.process_pdf(pdf_path, dpi=300)
        print(f"Processed {len(images)} pages with Module 1")
        
        # Load into Module 2 UI
        root = tk.Tk()
        app = ImageManagementUI(root)
        app.load_images(images)
        print("Loaded images into Module 2 UI")
        root.mainloop()
        
    except FileNotFoundError:
        print(f"PDF file not found: {pdf_path}")
        print("Launching UI without pre-loaded images...")
        # Launch UI anyway
        root = tk.Tk()
        app = ImageManagementUI(root)
        root.mainloop()
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    print("=" * 50)
    print("OCR PDF to Table - Example Usage")
    print("=" * 50)
    
    # Uncomment the example you want to run:
    
    # Example 1: Use Module 1 standalone
    # example_module1_standalone()
    
    # Example 2: Use Module 1 with custom settings
    # example_module1_with_custom_settings()
    
    # Example 3: Launch Module 2 UI directly
    example_module2_ui()
    
    # Example 4: Integrate both modules
    # example_integration()
