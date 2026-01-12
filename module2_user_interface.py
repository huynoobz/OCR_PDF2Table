"""
Module 2: User Interface
GUI module for visual inspection, manipulation, and export of processed images.
All image data comes from Module 1 - no direct PDF parsing.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageEnhance
from typing import List, Optional, Dict
import os
from module1_image_processing import ProcessedImage, PDFImageProcessor


class ImageViewer:
    """Canvas-based image viewer with zoom and pan capabilities."""
    
    def __init__(self, canvas: tk.Canvas):
        self.canvas = canvas
        self.image = None
        self.image_tk = None
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.canvas_width = 0
        self.canvas_height = 0
        
        # Bind mouse events (only if not in crop mode)
        # Note: Crop mode will rebind these events
        self._bind_viewer_events()
        
        self.canvas.update_idletasks()
        self.canvas_width = self.canvas.winfo_width()
        self.canvas_height = self.canvas.winfo_height()
    
    def _bind_viewer_events(self):
        """Bind viewer mouse events."""
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<MouseWheel>", self.on_wheel)
        self.canvas.bind("<Button-4>", self.on_wheel)  # Linux
        self.canvas.bind("<Button-5>", self.on_wheel)  # Linux
    
    def set_image(self, pil_image: Image.Image):
        """Set the image to display."""
        self.image = pil_image.copy()
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self._update_display()
    
    def _update_display(self):
        """Update the canvas display."""
        if self.image is None:
            return
        
        # Update canvas dimensions
        self.canvas.update_idletasks()
        self.canvas_width = self.canvas.winfo_width()
        self.canvas_height = self.canvas.winfo_height()
        
        if self.canvas_width <= 1 or self.canvas_height <= 1:
            return
        
        # Calculate display size
        img_width, img_height = self.image.size
        display_width = int(img_width * self.scale)
        display_height = int(img_height * self.scale)
        
        # Resize image
        resized = self.image.resize((display_width, display_height), Image.Resampling.LANCZOS)
        self.image_tk = ImageTk.PhotoImage(resized)
        
        # Clear and redraw
        self.canvas.delete("all")
        
        # Calculate position
        x = self.canvas_width // 2 + self.offset_x
        y = self.canvas_height // 2 + self.offset_y
        
        self.canvas.create_image(x, y, image=self.image_tk, anchor=tk.CENTER)
    
    def on_click(self, event):
        """Handle mouse click for panning."""
        self.last_pan_x = event.x
        self.last_pan_y = event.y
    
    def on_drag(self, event):
        """Handle mouse drag for panning."""
        if hasattr(self, 'last_pan_x'):
            dx = event.x - self.last_pan_x
            dy = event.y - self.last_pan_y
            self.offset_x += dx
            self.offset_y += dy
            self.last_pan_x = event.x
            self.last_pan_y = event.y
            self._update_display()
    
    def on_wheel(self, event):
        """Handle mouse wheel for zooming."""
        if event.num == 4 or event.delta > 0:
            self.zoom_in()
        elif event.num == 5 or event.delta < 0:
            self.zoom_out()
    
    def zoom_in(self, factor=1.2):
        """Zoom in."""
        self.scale *= factor
        if self.scale > 5.0:
            self.scale = 5.0
        self._update_display()
    
    def zoom_out(self, factor=1.2):
        """Zoom out."""
        self.scale /= factor
        if self.scale < 0.1:
            self.scale = 0.1
        self._update_display()
    
    def reset_view(self):
        """Reset zoom and pan."""
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self._update_display()
    
    def fit_to_window(self):
        """Fit image to window."""
        if self.image is None:
            return
        
        self.canvas.update_idletasks()
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        
        if canvas_w <= 1 or canvas_h <= 1:
            return
        
        img_w, img_h = self.image.size
        scale_w = canvas_w / img_w
        scale_h = canvas_h / img_h
        self.scale = min(scale_w, scale_h) * 0.9  # 90% to leave some margin
        
        self.offset_x = 0
        self.offset_y = 0
        self._update_display()


class ImageEditor:
    """Non-destructive image editor for operations like rotate, crop, brightness, contrast."""
    
    def __init__(self):
        self.original_image: Optional[Image.Image] = None
        self.current_image: Optional[Image.Image] = None
        self.rotation = 0.0
        self.brightness = 1.0
        self.contrast = 1.0
        self.crop_box: Optional[tuple] = None
    
    def set_image(self, pil_image: Image.Image):
        """Set the base image."""
        self.original_image = pil_image.copy()
        self.current_image = pil_image.copy()
        self.rotation = 0.0
        self.brightness = 1.0
        self.contrast = 1.0
        self.crop_box = None
    
    def rotate(self, angle: float):
        """Rotate image (non-destructive)."""
        if self.original_image is None:
            return
        self.rotation += angle
        self._apply_all_operations()
    
    def set_brightness(self, factor: float):
        """Set brightness factor (1.0 = no change)."""
        self.brightness = factor
        self._apply_all_operations()
    
    def set_contrast(self, factor: float):
        """Set contrast factor (1.0 = no change)."""
        self.contrast = factor
        self._apply_all_operations()
    
    def set_crop(self, crop_box: Optional[tuple]):
        """Set crop box (left, top, right, bottom)."""
        self.crop_box = crop_box
        self._apply_all_operations()
    
    def _apply_all_operations(self):
        """Apply all operations to create current image."""
        if self.original_image is None:
            return
        
        img = self.original_image.copy()
        
        # Apply rotation
        if self.rotation != 0:
            img = img.rotate(-self.rotation, expand=True)
        
        # Apply crop
        if self.crop_box:
            left, top, right, bottom = self.crop_box
            img = img.crop((left, top, right, bottom))
        
        # Apply brightness
        if self.brightness != 1.0:
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(self.brightness)
        
        # Apply contrast
        if self.contrast != 1.0:
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(self.contrast)
        
        self.current_image = img
    
    def get_current_image(self) -> Optional[Image.Image]:
        """Get the current edited image."""
        return self.current_image
    
    def reset(self):
        """Reset all operations."""
        if self.original_image:
            self.set_image(self.original_image)


class ImageManagementUI:
    """Main UI class for image management and editing."""
    
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Image Processing and Management")
        self.root.geometry("1200x800")
        
        # Data
        self.processed_images: List[ProcessedImage] = []
        self.current_index = -1
        self.selected_indices: set = set()
        self.image_editors: Dict[int, ImageEditor] = {}
        
        # Create UI
        self._create_ui()
    
    def _create_ui(self):
        """Create the user interface."""
        # Main container
        main_frame = ttk.Frame(self.root, padding="5")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        # Left panel - Image list
        left_panel = ttk.Frame(main_frame, width=200)
        left_panel.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)
        
        # Image list
        list_label = ttk.Label(left_panel, text="Images")
        list_label.pack(fill=tk.X, pady=(0, 5))
        
        # Listbox with scrollbar
        list_frame = ttk.Frame(left_panel)
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.image_listbox = tk.Listbox(list_frame, selectmode=tk.EXTENDED, yscrollcommand=scrollbar.set)
        self.image_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.image_listbox.yview)
        
        self.image_listbox.bind('<<ListboxSelect>>', self._on_listbox_select)
        self.image_listbox.bind('<Double-Button-1>', self._on_listbox_double_click)
        
        # Image management buttons
        btn_frame = ttk.Frame(left_panel)
        btn_frame.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Button(btn_frame, text="Load from Module 1", command=self._load_from_module1).pack(fill=tk.X, pady=2)
        ttk.Button(btn_frame, text="Add External", command=self._add_external_image).pack(fill=tk.X, pady=2)
        ttk.Button(btn_frame, text="Delete Selected", command=self._delete_selected).pack(fill=tk.X, pady=2)
        ttk.Button(btn_frame, text="Move Up", command=self._move_up).pack(fill=tk.X, pady=2)
        ttk.Button(btn_frame, text="Move Down", command=self._move_down).pack(fill=tk.X, pady=2)
        ttk.Button(btn_frame, text="Export Selected", command=self._export_selected).pack(fill=tk.X, pady=2)
        
        # Right panel - Image viewer and controls
        right_panel = ttk.Frame(main_frame)
        right_panel.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Viewer frame
        viewer_frame = ttk.Frame(right_panel)
        viewer_frame.pack(fill=tk.BOTH, expand=True)
        
        # Canvas for image display
        canvas_frame = ttk.Frame(viewer_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(canvas_frame, bg="gray", cursor="crosshair")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        self.viewer = ImageViewer(self.canvas)
        
        # Navigation controls
        nav_frame = ttk.Frame(viewer_frame)
        nav_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(nav_frame, text="◀ Prev", command=self._prev_image).pack(side=tk.LEFT, padx=2)
        ttk.Button(nav_frame, text="Next ▶", command=self._next_image).pack(side=tk.LEFT, padx=2)
        ttk.Label(nav_frame, text="Page:").pack(side=tk.LEFT, padx=5)
        self.page_label = ttk.Label(nav_frame, text="0 / 0")
        self.page_label.pack(side=tk.LEFT, padx=2)
        
        ttk.Button(nav_frame, text="Reset View", command=self.viewer.reset_view).pack(side=tk.LEFT, padx=5)
        ttk.Button(nav_frame, text="Fit to Window", command=self.viewer.fit_to_window).pack(side=tk.LEFT, padx=2)
        ttk.Button(nav_frame, text="Zoom In", command=self.viewer.zoom_in).pack(side=tk.LEFT, padx=2)
        ttk.Button(nav_frame, text="Zoom Out", command=self.viewer.zoom_out).pack(side=tk.LEFT, padx=2)
        
        # Image operations frame
        ops_frame = ttk.LabelFrame(right_panel, text="Image Operations", padding="10")
        ops_frame.pack(fill=tk.X, pady=5)
        
        # Rotation
        rot_frame = ttk.Frame(ops_frame)
        rot_frame.pack(fill=tk.X, pady=2)
        ttk.Label(rot_frame, text="Rotation:").pack(side=tk.LEFT, padx=5)
        ttk.Button(rot_frame, text="↺ -90°", command=lambda: self._rotate(-90)).pack(side=tk.LEFT, padx=2)
        ttk.Button(rot_frame, text="↻ +90°", command=lambda: self._rotate(90)).pack(side=tk.LEFT, padx=2)
        ttk.Button(rot_frame, text="180°", command=lambda: self._rotate(180)).pack(side=tk.LEFT, padx=2)
        ttk.Button(rot_frame, text="Reset", command=self._reset_operations).pack(side=tk.LEFT, padx=5)
        
        # Brightness
        bright_frame = ttk.Frame(ops_frame)
        bright_frame.pack(fill=tk.X, pady=2)
        ttk.Label(bright_frame, text="Brightness:").pack(side=tk.LEFT, padx=5)
        self.brightness_var = tk.DoubleVar(value=1.0)
        brightness_scale = ttk.Scale(bright_frame, from_=0.5, to=2.0, variable=self.brightness_var, 
                                     orient=tk.HORIZONTAL, length=200, command=self._on_brightness_change)
        brightness_scale.pack(side=tk.LEFT, padx=5)
        self.brightness_label = ttk.Label(bright_frame, text="1.00")
        self.brightness_label.pack(side=tk.LEFT, padx=5)
        
        # Contrast
        contrast_frame = ttk.Frame(ops_frame)
        contrast_frame.pack(fill=tk.X, pady=2)
        ttk.Label(contrast_frame, text="Contrast:").pack(side=tk.LEFT, padx=5)
        self.contrast_var = tk.DoubleVar(value=1.0)
        contrast_scale = ttk.Scale(contrast_frame, from_=0.5, to=2.0, variable=self.contrast_var,
                                   orient=tk.HORIZONTAL, length=200, command=self._on_contrast_change)
        contrast_scale.pack(side=tk.LEFT, padx=5)
        self.contrast_label = ttk.Label(contrast_frame, text="1.00")
        self.contrast_label.pack(side=tk.LEFT, padx=5)
        
        # Crop
        crop_frame = ttk.Frame(ops_frame)
        crop_frame.pack(fill=tk.X, pady=2)
        ttk.Label(crop_frame, text="Crop:").pack(side=tk.LEFT, padx=5)
        ttk.Button(crop_frame, text="Select Crop Area", command=self._start_crop).pack(side=tk.LEFT, padx=2)
        ttk.Button(crop_frame, text="Clear Crop", command=self._clear_crop).pack(side=tk.LEFT, padx=2)
        
        # Metadata display
        meta_frame = ttk.LabelFrame(right_panel, text="Metadata", padding="10")
        meta_frame.pack(fill=tk.X, pady=5)
        
        self.metadata_text = tk.Text(meta_frame, height=6, wrap=tk.WORD)
        self.metadata_text.pack(fill=tk.BOTH, expand=True)
        
        # Crop selection state
        self.crop_start = None
        self.crop_rect = None
        self.crop_mode = False
    
    def load_images(self, processed_images: List[ProcessedImage]):
        """Load processed images from Module 1."""
        self.processed_images = processed_images
        self.image_editors = {}
        self.current_index = 0 if processed_images else -1
        self._update_listbox()
        if processed_images:
            self._display_current_image()
    
    def _load_from_module1(self):
        """Load images by processing a PDF file."""
        pdf_path = filedialog.askopenfilename(
            title="Select PDF File",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")]
        )
        if not pdf_path:
            return
        
        try:
            processor = PDFImageProcessor()
            images = processor.process_pdf(pdf_path, dpi=300)
            self.load_images(images)
            messagebox.showinfo("Success", f"Loaded {len(images)} pages from PDF")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load PDF: {str(e)}")
    
    def _add_external_image(self):
        """Add an external image file."""
        file_path = filedialog.askopenfilename(
            title="Select Image File",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff *.gif"),
                ("All files", "*.*")
            ]
        )
        if not file_path:
            return
        
        try:
            pil_image = Image.open(file_path)
            processor = PDFImageProcessor()
            page_num = len(self.processed_images) + 1
            processed = processor.process_single_image(pil_image, page_number=page_num)
            self.processed_images.append(processed)
            self._update_listbox()
            self.current_index = len(self.processed_images) - 1
            self._display_current_image()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def _delete_selected(self):
        """Delete selected images."""
        selected = self.image_listbox.curselection()
        if not selected:
            messagebox.showwarning("Warning", "No images selected")
            return
        
        # Delete in reverse order to maintain indices
        for idx in reversed(selected):
            del self.processed_images[idx]
            if idx in self.image_editors:
                del self.image_editors[idx]
        
        # Update current index
        if self.current_index >= len(self.processed_images):
            self.current_index = len(self.processed_images) - 1
        
        self._update_listbox()
        if self.processed_images:
            self._display_current_image()
        else:
            self.viewer.set_image(Image.new('RGB', (800, 600), color='white'))
            self.metadata_text.delete(1.0, tk.END)
    
    def _move_up(self):
        """Move selected images up."""
        selected = list(self.image_listbox.curselection())
        if not selected or selected[0] == 0:
            return
        
        for idx in selected:
            if idx > 0:
                self.processed_images[idx], self.processed_images[idx-1] = \
                    self.processed_images[idx-1], self.processed_images[idx]
                # Swap editors if they exist
                if idx in self.image_editors and idx-1 in self.image_editors:
                    self.image_editors[idx], self.image_editors[idx-1] = \
                        self.image_editors[idx-1], self.image_editors[idx]
        
        self._update_listbox()
        # Reselect items
        for idx in selected:
            if idx > 0:
                self.image_listbox.selection_set(idx - 1)
    
    def _move_down(self):
        """Move selected images down."""
        selected = list(self.image_listbox.curselection())
        if not selected or selected[-1] >= len(self.processed_images) - 1:
            return
        
        for idx in reversed(selected):
            if idx < len(self.processed_images) - 1:
                self.processed_images[idx], self.processed_images[idx+1] = \
                    self.processed_images[idx+1], self.processed_images[idx]
                # Swap editors if they exist
                if idx in self.image_editors and idx+1 in self.image_editors:
                    self.image_editors[idx], self.image_editors[idx+1] = \
                        self.image_editors[idx+1], self.image_editors[idx]
        
        self._update_listbox()
        # Reselect items
        for idx in selected:
            if idx < len(self.processed_images) - 1:
                self.image_listbox.selection_set(idx + 1)
    
    def _export_selected(self):
        """Export selected images to a folder."""
        selected = self.image_listbox.curselection()
        if not selected:
            messagebox.showwarning("Warning", "No images selected")
            return
        
        folder = filedialog.askdirectory(title="Select Export Folder")
        if not folder:
            return
        
        try:
            for idx in selected:
                if idx < len(self.processed_images):
                    processed = self.processed_images[idx]
                    # Get edited image if editor exists
                    if idx in self.image_editors:
                        editor = self.image_editors[idx]
                        img = editor.get_current_image()
                    else:
                        img = processed.pil_image
                    
                    # Save image
                    filename = f"page_{processed.metadata.page_number:03d}.png"
                    filepath = os.path.join(folder, filename)
                    img.save(filepath, "PNG")
            
            messagebox.showinfo("Success", f"Exported {len(selected)} image(s) to {folder}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export images: {str(e)}")
    
    def _update_listbox(self):
        """Update the image listbox."""
        self.image_listbox.delete(0, tk.END)
        for i, processed in enumerate(self.processed_images):
            meta = processed.metadata
            label = f"Page {meta.page_number} ({meta.processed_size[0]}x{meta.processed_size[1]})"
            self.image_listbox.insert(tk.END, label)
        
        if self.current_index >= 0:
            self.image_listbox.selection_set(self.current_index)
            self.image_listbox.see(self.current_index)
    
    def _on_listbox_select(self, event):
        """Handle listbox selection."""
        selected = self.image_listbox.curselection()
        if selected:
            self.current_index = selected[0]
            self._display_current_image()
    
    def _on_listbox_double_click(self, event):
        """Handle listbox double-click."""
        selected = self.image_listbox.curselection()
        if selected:
            self.current_index = selected[0]
            self._display_current_image()
    
    def _prev_image(self):
        """Go to previous image."""
        if self.processed_images and self.current_index > 0:
            self.current_index -= 1
            self._update_listbox()
            self._display_current_image()
    
    def _next_image(self):
        """Go to next image."""
        if self.processed_images and self.current_index < len(self.processed_images) - 1:
            self.current_index += 1
            self._update_listbox()
            self._display_current_image()
    
    def _display_current_image(self):
        """Display the current image."""
        if self.current_index < 0 or self.current_index >= len(self.processed_images):
            return
        
        processed = self.processed_images[self.current_index]
        
        # Get or create editor
        if self.current_index not in self.image_editors:
            editor = ImageEditor()
            editor.set_image(processed.pil_image)
            self.image_editors[self.current_index] = editor
        else:
            editor = self.image_editors[self.current_index]
        
        # Display edited image
        current_img = editor.get_current_image()
        self.viewer.set_image(current_img)
        
        # Update page label
        self.page_label.config(text=f"{self.current_index + 1} / {len(self.processed_images)}")
        
        # Update metadata
        meta = processed.metadata
        meta_text = f"Page Number: {meta.page_number}\n"
        meta_text += f"Original Size: {meta.original_size[0]} x {meta.original_size[1]}\n"
        meta_text += f"Processed Size: {meta.processed_size[0]} x {meta.processed_size[1]}\n"
        meta_text += f"DPI: {meta.dpi}\n"
        meta_text += f"Applied Operations: {', '.join(meta.applied_operations)}\n"
        if meta.rotation != 0:
            meta_text += f"Rotation: {meta.rotation}°\n"
        
        self.metadata_text.delete(1.0, tk.END)
        self.metadata_text.insert(1.0, meta_text)
        
        # Update sliders
        self.brightness_var.set(editor.brightness)
        self.contrast_var.set(editor.contrast)
        self._update_brightness_label()
        self._update_contrast_label()
    
    def _rotate(self, angle: float):
        """Rotate current image."""
        if self.current_index < 0 or self.current_index >= len(self.processed_images):
            return
        
        if self.current_index not in self.image_editors:
            processed = self.processed_images[self.current_index]
            editor = ImageEditor()
            editor.set_image(processed.pil_image)
            self.image_editors[self.current_index] = editor
        
        editor = self.image_editors[self.current_index]
        editor.rotate(angle)
        self.viewer.set_image(editor.get_current_image())
    
    def _on_brightness_change(self, value):
        """Handle brightness slider change."""
        if self.current_index < 0 or self.current_index >= len(self.processed_images):
            return
        
        if self.current_index not in self.image_editors:
            processed = self.processed_images[self.current_index]
            editor = ImageEditor()
            editor.set_image(processed.pil_image)
            self.image_editors[self.current_index] = editor
        
        editor = self.image_editors[self.current_index]
        editor.set_brightness(float(value))
        self.viewer.set_image(editor.get_current_image())
        self._update_brightness_label()
    
    def _update_brightness_label(self):
        """Update brightness label."""
        self.brightness_label.config(text=f"{self.brightness_var.get():.2f}")
    
    def _on_contrast_change(self, value):
        """Handle contrast slider change."""
        if self.current_index < 0 or self.current_index >= len(self.processed_images):
            return
        
        if self.current_index not in self.image_editors:
            processed = self.processed_images[self.current_index]
            editor = ImageEditor()
            editor.set_image(processed.pil_image)
            self.image_editors[self.current_index] = editor
        
        editor = self.image_editors[self.current_index]
        editor.set_contrast(float(value))
        self.viewer.set_image(editor.get_current_image())
        self._update_contrast_label()
    
    def _update_contrast_label(self):
        """Update contrast label."""
        self.contrast_label.config(text=f"{self.contrast_var.get():.2f}")
    
    def _reset_operations(self):
        """Reset all operations on current image."""
        if self.current_index < 0 or self.current_index >= len(self.processed_images):
            return
        
        if self.current_index in self.image_editors:
            editor = self.image_editors[self.current_index]
            editor.reset()
            self.viewer.set_image(editor.get_current_image())
            self.brightness_var.set(1.0)
            self.contrast_var.set(1.0)
            self._update_brightness_label()
            self._update_contrast_label()
    
    def _start_crop(self):
        """Start crop selection mode."""
        self.crop_mode = True
        self.canvas.config(cursor="crosshair")
        # Unbind viewer events
        self.canvas.unbind("<Button-1>")
        self.canvas.unbind("<B1-Motion>")
        self.canvas.unbind("<MouseWheel>")
        self.canvas.unbind("<Button-4>")
        self.canvas.unbind("<Button-5>")
        # Bind crop events
        self.canvas.bind("<Button-1>", self._crop_start)
        self.canvas.bind("<B1-Motion>", self._crop_drag)
        self.canvas.bind("<ButtonRelease-1>", self._crop_end)
    
    def _crop_start(self, event):
        """Start crop selection."""
        self.crop_start = (event.x, event.y)
        if self.crop_rect:
            self.canvas.delete(self.crop_rect)
    
    def _crop_drag(self, event):
        """Update crop rectangle during drag."""
        if self.crop_start:
            if self.crop_rect:
                self.canvas.delete(self.crop_rect)
            x1, y1 = self.crop_start
            x2, y2 = event.x, event.y
            self.crop_rect = self.canvas.create_rectangle(x1, y1, x2, y2, outline="red", width=2)
    
    def _crop_end(self, event):
        """End crop selection and apply crop."""
        if self.crop_start and self.current_index >= 0:
            x1, y1 = self.crop_start
            x2, y2 = event.x, event.y
            
            # Convert canvas coordinates to image coordinates
            # Account for viewer's zoom and pan
            img = self.image_editors[self.current_index].get_current_image()
            if img:
                img_w, img_h = img.size
                canvas_w = self.canvas.winfo_width()
                canvas_h = self.canvas.winfo_height()
                
                # Get viewer's current scale and offset
                viewer_scale = self.viewer.scale
                viewer_offset_x = self.viewer.offset_x
                viewer_offset_y = self.viewer.offset_y
                
                # Calculate image center position on canvas
                img_center_x = canvas_w // 2 + viewer_offset_x
                img_center_y = canvas_h // 2 + viewer_offset_y
                
                # Convert canvas coordinates to image coordinates
                # Image coordinates relative to center
                rel_x1 = (x1 - img_center_x) / viewer_scale
                rel_y1 = (y1 - img_center_y) / viewer_scale
                rel_x2 = (x2 - img_center_x) / viewer_scale
                rel_y2 = (y2 - img_center_y) / viewer_scale
                
                # Convert to absolute image coordinates
                left = max(0, int(img_w // 2 + rel_x1))
                top = max(0, int(img_h // 2 + rel_y1))
                right = min(img_w, int(img_w // 2 + rel_x2))
                bottom = min(img_h, int(img_h // 2 + rel_y2))
                
                # Ensure valid crop box
                if right > left and bottom > top:
                    crop_box = (left, top, right, bottom)
                    editor = self.image_editors[self.current_index]
                    editor.set_crop(crop_box)
                    self.viewer.set_image(editor.get_current_image())
        
        # Clean up
        self.crop_mode = False
        self.canvas.config(cursor="")
        # Unbind crop events
        self.canvas.unbind("<Button-1>")
        self.canvas.unbind("<B1-Motion>")
        self.canvas.unbind("<ButtonRelease-1>")
        # Rebind viewer events
        self.viewer._bind_viewer_events()
        if self.crop_rect:
            self.canvas.delete(self.crop_rect)
            self.crop_rect = None
    
    def _clear_crop(self):
        """Clear crop on current image."""
        if self.current_index >= 0 and self.current_index in self.image_editors:
            editor = self.image_editors[self.current_index]
            editor.set_crop(None)
            self.viewer.set_image(editor.get_current_image())


def main():
    """Main entry point for the UI module."""
    root = tk.Tk()
    app = ImageManagementUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
