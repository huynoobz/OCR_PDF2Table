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
import re
import json
from dataclasses import dataclass, asdict
from PIL import ImageDraw
from module1_image_processing import ProcessedImage, PDFImageProcessor


@dataclass
class UISettings:
    default_dpi: int = 300
    auto_fit_on_load: bool = True
    show_toolbar_hint: bool = True
    confirm_delete: bool = True
    # Friendly key strings (e.g. "Ctrl+O", "Tab") -> converted to Tk sequences at bind time
    keymap: Optional[Dict[str, str]] = None


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
        self._base_image: Optional[Image.Image] = None  # after transforms, before paint overlay
        self.current_image: Optional[Image.Image] = None
        self.rotation = 0.0
        self.brightness = 1.0
        self.contrast = 1.0
        self.crop_box: Optional[tuple] = None

        # Paint overlay (non-destructive)
        self.paint_layer: Optional[Image.Image] = None  # RGBA, same size as _base_image
        self.brush_color = (255, 0, 0)  # RGB
        self.brush_size = 10
    
    def set_image(self, pil_image: Image.Image):
        """Set the base image."""
        self.original_image = pil_image.copy()
        self.current_image = pil_image.copy()
        self._base_image = pil_image.copy()
        self.rotation = 0.0
        self.brightness = 1.0
        self.contrast = 1.0
        self.crop_box = None
        self.paint_layer = None
    
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

        # Store base image (after transforms/color, before paint)
        self._base_image = img

        # Composite paint layer (if any)
        if self.paint_layer is not None:
            if self.paint_layer.size != img.size:
                # Transform size changed; discard paint layer (safe default)
                self.paint_layer = None
            else:
                base_rgba = img.convert("RGBA")
                img = Image.alpha_composite(base_rgba, self.paint_layer).convert("RGB")

        self.current_image = img

    def get_base_image(self) -> Optional[Image.Image]:
        """Get current base image (after transforms, before paint overlay)."""
        return self._base_image

    def _ensure_paint_layer(self):
        if self._base_image is None:
            return
        if self.paint_layer is None or self.paint_layer.size != self._base_image.size:
            self.paint_layer = Image.new("RGBA", self._base_image.size, (0, 0, 0, 0))

    def paint_line(self, p1: tuple[float, float], p2: tuple[float, float]):
        """Draw a brush line segment on the paint layer."""
        self._ensure_paint_layer()
        if self.paint_layer is None:
            return
        draw = ImageDraw.Draw(self.paint_layer)
        rgba = (int(self.brush_color[0]), int(self.brush_color[1]), int(self.brush_color[2]), 255)
        draw.line([p1, p2], fill=rgba, width=int(self.brush_size))
        # Refresh current image composition
        self._apply_all_operations()

    def clear_paint(self):
        """Clear painted strokes."""
        self.paint_layer = None
        self._apply_all_operations()
    
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

        # Settings / shortcuts
        self.settings_path = os.path.join(os.path.dirname(__file__), "ui_settings.json")
        self.settings = self._load_settings()
        self._settings_window: Optional[tk.Toplevel] = None
        self._shortcut_bind_ids: Dict[str, str] = {}
        # Used by the Select Wizard dialog (menu); kept even if the left-panel wizard is hidden.
        self.select_pattern_var = tk.StringVar(value="")
        
        # Create UI
        self._create_ui()
        self._apply_settings_to_ui()
        self._bind_shortcuts()
    
    def _create_ui(self):
        """Create the user interface."""
        # Root grid: toolbar (row 0) + main content (row 1)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=0)
        self.root.rowconfigure(1, weight=1)

        # Top toolbar with grouped dropdowns
        toolbar = ttk.Frame(self.root, padding="4")
        toolbar.grid(row=0, column=0, sticky=(tk.W, tk.E))
        toolbar.columnconfigure(0, weight=1)
        self._create_toolbar(toolbar)

        # Main container
        main_frame = ttk.Frame(self.root, padding="5")
        main_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Layout: [Tools | Viewer+controls | Image list]
        main_frame.columnconfigure(0, weight=0)  # tools
        main_frame.columnconfigure(1, weight=1)  # viewer
        main_frame.columnconfigure(2, weight=0)  # list
        main_frame.rowconfigure(0, weight=1)

        # Left panel - Tools palette
        tools_panel = ttk.Frame(main_frame, width=140)
        tools_panel.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.W), padx=(0, 6))
        self._create_tools_panel(tools_panel)

        # Center panel - Image viewer and controls
        right_panel = ttk.Frame(main_frame)
        right_panel.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Right panel - Image list
        list_panel = ttk.Frame(main_frame, width=240)
        list_panel.grid(row=0, column=2, sticky=(tk.N, tk.S, tk.E), padx=(6, 0))

        list_label = ttk.Label(list_panel, text="Images")
        list_label.pack(fill=tk.X, pady=(0, 5))

        list_frame = ttk.Frame(list_panel)
        list_frame.pack(fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.image_listbox = tk.Listbox(list_frame, selectmode=tk.EXTENDED, yscrollcommand=scrollbar.set)
        self.image_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.image_listbox.yview)

        self.image_listbox.bind("<<ListboxSelect>>", self._on_listbox_select)
        self.image_listbox.bind("<Double-Button-1>", self._on_listbox_double_click)

        # Image management buttons (move next to the list)
        btn_frame = ttk.Frame(list_panel)
        btn_frame.pack(fill=tk.X, pady=(6, 0))

        ttk.Button(btn_frame, text="Load from Module 1", command=self._load_from_module1).pack(fill=tk.X, pady=2)
        ttk.Button(btn_frame, text="Add External", command=self._add_external_image).pack(fill=tk.X, pady=2)
        ttk.Button(btn_frame, text="Delete Selected", command=self._delete_selected).pack(fill=tk.X, pady=2)
        ttk.Button(btn_frame, text="Move Up", command=self._move_up).pack(fill=tk.X, pady=2)
        ttk.Button(btn_frame, text="Move Down", command=self._move_down).pack(fill=tk.X, pady=2)
        ttk.Button(btn_frame, text="Export Selected", command=self._export_selected).pack(fill=tk.X, pady=2)
        
        # Viewer frame
        viewer_frame = ttk.Frame(right_panel)
        viewer_frame.pack(fill=tk.BOTH, expand=True)
        
        # Canvas for image display
        canvas_frame = ttk.Frame(viewer_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(canvas_frame, bg="gray", cursor="crosshair")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        self.viewer = ImageViewer(self.canvas)
        # Initialize tool bindings now that canvas/viewer exist
        if hasattr(self, "active_tool"):
            self._apply_active_tool()
        
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
            if getattr(self.settings, "auto_fit_on_load", False):
                self.viewer.fit_to_window()

    # ----------------------------
    # Tools palette
    # ----------------------------
    def _create_tools_panel(self, parent: ttk.Frame):
        """Create a vertical tools palette (left side)."""
        ttk.Label(parent, text="Tools").pack(anchor=tk.W, pady=(0, 6))

        self.active_tool = tk.StringVar(value="hand")

        def tool_btn(text: str, value: str):
            b = ttk.Radiobutton(
                parent,
                text=text,
                value=value,
                variable=self.active_tool,
                command=self._apply_active_tool,
            )
            b.pack(fill=tk.X, pady=2)

        tool_btn("Hand (Pan)", "hand")
        tool_btn("Zoom", "zoom")
        tool_btn("Crop", "crop")
        tool_btn("Brush", "brush")
        tool_btn("Eyedropper", "eyedropper")

        ttk.Separator(parent).pack(fill=tk.X, pady=8)

        # Brush controls
        ttk.Label(parent, text="Brush size").pack(anchor=tk.W)
        self.brush_size_var = tk.IntVar(value=10)
        ttk.Scale(
            parent,
            from_=1,
            to=50,
            orient=tk.HORIZONTAL,
            command=lambda v: self._set_brush_size(int(float(v))),
        ).pack(fill=tk.X, pady=(2, 6))

        ttk.Label(parent, text="Brush color").pack(anchor=tk.W)
        self._brush_color_preview = tk.Label(parent, text="      ", bg="#ff0000", relief=tk.SOLID, bd=1)
        self._brush_color_preview.pack(anchor=tk.W, pady=(2, 4))
        ttk.Button(parent, text="Clear paint", command=self._clear_paint_selected).pack(fill=tk.X, pady=(2, 0))

        ttk.Separator(parent).pack(fill=tk.X, pady=8)
        ttk.Button(parent, text="Settings…", command=self._open_settings_window).pack(fill=tk.X, pady=2)

        # Apply tool bindings now that canvas exists
        # (canvas is created later; _apply_active_tool is safe to call once viewer is ready)

    def _apply_active_tool(self):
        """Bind mouse actions according to the active tool."""
        if not hasattr(self, "canvas") or not hasattr(self, "viewer"):
            return

        tool = self.active_tool.get()

        # Clear any previous special binds (except wheel, which viewer manages)
        self.canvas.unbind("<Button-1>")
        self.canvas.unbind("<B1-Motion>")
        self.canvas.unbind("<ButtonRelease-1>")
        self.canvas.unbind("<Button-3>")

        if tool == "hand":
            self.canvas.config(cursor="fleur")
            self.viewer._bind_viewer_events()
            return

        if tool == "crop":
            self.canvas.config(cursor="crosshair")
            self._start_crop()
            return

        if tool == "zoom":
            self.canvas.config(cursor="plus")

            def zoom_in_click(event):
                self.viewer.zoom_in()

            def zoom_out_click(event):
                self.viewer.zoom_out()

            # Left click zooms in, right click zooms out
            self.canvas.bind("<Button-1>", lambda e: zoom_in_click(e))
            self.canvas.bind("<Button-3>", lambda e: zoom_out_click(e))
            # Keep wheel zoom from viewer
            self.canvas.bind("<MouseWheel>", self.viewer.on_wheel)
            self.canvas.bind("<Button-4>", self.viewer.on_wheel)
            self.canvas.bind("<Button-5>", self.viewer.on_wheel)
            return

        if tool == "eyedropper":
            self.canvas.config(cursor="tcross")

            def pick(event):
                self._eyedropper_pick(event.x, event.y)

            self.canvas.bind("<Button-1>", pick)
            # Keep wheel zoom from viewer
            self.canvas.bind("<MouseWheel>", self.viewer.on_wheel)
            self.canvas.bind("<Button-4>", self.viewer.on_wheel)
            self.canvas.bind("<Button-5>", self.viewer.on_wheel)
            return

        if tool == "brush":
            self.canvas.config(cursor="pencil")
            self._brush_last_pt = None

            def start(event):
                self._brush_last_pt = self._canvas_to_image_xy(event.x, event.y)

            def move(event):
                if self._brush_last_pt is None:
                    return
                p2 = self._canvas_to_image_xy(event.x, event.y)
                if p2 is None:
                    return
                self._paint_line_on_selected(self._brush_last_pt, p2)
                self._brush_last_pt = p2

            def end(_event):
                self._brush_last_pt = None

            self.canvas.bind("<Button-1>", start)
            self.canvas.bind("<B1-Motion>", move)
            self.canvas.bind("<ButtonRelease-1>", end)
            # Keep wheel zoom from viewer
            self.canvas.bind("<MouseWheel>", self.viewer.on_wheel)
            self.canvas.bind("<Button-4>", self.viewer.on_wheel)
            self.canvas.bind("<Button-5>", self.viewer.on_wheel)
            return

    # ----------------------------
    # Toolbar / Menus
    # ----------------------------
    def _create_toolbar(self, parent: ttk.Frame):
        """Create a top toolbar with grouped dropdown menus."""
        # File menu
        file_btn = ttk.Menubutton(parent, text="File")
        file_menu = tk.Menu(file_btn, tearoff=0)
        file_menu.add_command(label="Open PDF…", command=self._load_from_module1)
        file_menu.add_command(label="Add External Image…", command=self._add_external_image)
        file_menu.add_separator()
        file_menu.add_command(label="Export Selected…", command=self._export_selected)
        file_menu.add_command(label="Settings…", command=self._open_settings_window)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.destroy)
        file_btn["menu"] = file_menu
        file_btn.pack(side=tk.LEFT, padx=(0, 6))

        # Edit menu
        edit_btn = ttk.Menubutton(parent, text="Edit")
        edit_menu = tk.Menu(edit_btn, tearoff=0)
        edit_menu.add_command(label="Rotate -90°", command=lambda: self._rotate(-90))
        edit_menu.add_command(label="Rotate +90°", command=lambda: self._rotate(90))
        edit_menu.add_command(label="Rotate 180°", command=lambda: self._rotate(180))
        edit_menu.add_separator()
        edit_menu.add_command(label="Reset operations", command=self._reset_operations)
        edit_menu.add_command(label="Clear crop", command=self._clear_crop)
        edit_menu.add_separator()
        edit_menu.add_command(label="Delete selected", command=self._delete_selected)
        edit_btn["menu"] = edit_menu
        edit_btn.pack(side=tk.LEFT, padx=(0, 6))

        # Select menu
        select_btn = ttk.Menubutton(parent, text="Select")
        select_menu = tk.Menu(select_btn, tearoff=0)
        select_menu.add_command(label="All", command=self._select_all)
        select_menu.add_command(label="None", command=self._select_none)
        select_menu.add_command(label="Invert", command=self._select_invert)
        select_menu.add_separator()
        select_menu.add_command(label="Odd pages (1,3,5,...)", command=self._select_odd)
        select_menu.add_command(label="Even pages (2,4,6,...)", command=self._select_even)
        select_menu.add_separator()
        select_menu.add_command(label="Wizard…", command=self._open_select_wizard_dialog)
        select_btn["menu"] = select_menu
        select_btn.pack(side=tk.LEFT, padx=(0, 6))

        # Quick hint
        self._toolbar_hint_label = ttk.Label(
            parent,
            text="Tip: multi-select + edit applies to all selected. Pattern examples: 1-2, 1,3,5",
        )
        self._toolbar_hint_label.pack(side=tk.LEFT, padx=8)

    # ----------------------------
    # Brush / Eyedropper helpers
    # ----------------------------
    def _set_brush_size(self, size: int):
        size = max(1, min(200, int(size)))
        if hasattr(self, "brush_size_var"):
            self.brush_size_var.set(size)

    def _update_brush_preview(self, rgb: tuple[int, int, int]):
        if hasattr(self, "_brush_color_preview") and self._brush_color_preview is not None:
            self._brush_color_preview.config(bg=f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}")

    def _canvas_to_image_xy(self, canvas_x: float, canvas_y: float) -> Optional[tuple[float, float]]:
        """Map canvas coordinates to current displayed image coordinates."""
        if self.current_index < 0 or self.current_index >= len(self.processed_images):
            return None
        if self.current_index not in self.image_editors:
            return None
        img = self.image_editors[self.current_index].get_current_image()
        if img is None:
            return None

        img_w, img_h = img.size
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        scale = float(self.viewer.scale) if self.viewer.scale else 1.0
        center_x = canvas_w // 2 + int(self.viewer.offset_x)
        center_y = canvas_h // 2 + int(self.viewer.offset_y)

        ix = (img_w / 2.0) + ((canvas_x - center_x) / scale)
        iy = (img_h / 2.0) + ((canvas_y - center_y) / scale)

        ix = max(0.0, min(float(img_w - 1), ix))
        iy = max(0.0, min(float(img_h - 1), iy))
        return (ix, iy)

    def _ensure_editor(self, idx: int) -> ImageEditor:
        if idx not in self.image_editors:
            processed = self.processed_images[idx]
            editor = ImageEditor()
            editor.set_image(processed.pil_image)
            self.image_editors[idx] = editor
        return self.image_editors[idx]

    def _apply_brush_settings_to_editor(self, editor: ImageEditor):
        if hasattr(self, "brush_size_var"):
            editor.brush_size = int(self.brush_size_var.get())
        # brush_color is already stored per editor; keep UI preview in sync elsewhere

    def _paint_line_on_selected(self, p1: tuple[float, float], p2: tuple[float, float]):
        targets = self._get_selected_indices_or_current()
        if not targets:
            return

        for idx in targets:
            editor = self._ensure_editor(idx)
            self._apply_brush_settings_to_editor(editor)
            editor.paint_line(p1, p2)

        self._display_current_image()

    def _eyedropper_pick(self, canvas_x: float, canvas_y: float):
        p = self._canvas_to_image_xy(canvas_x, canvas_y)
        if p is None:
            return
        ix, iy = p
        editor = self._ensure_editor(self.current_index)
        base = editor.get_base_image()
        if base is None:
            base = editor.get_current_image()
        if base is None:
            return
        rgb = base.convert("RGB").getpixel((int(ix), int(iy)))
        editor.brush_color = (int(rgb[0]), int(rgb[1]), int(rgb[2]))
        self._update_brush_preview(editor.brush_color)
        # Switch to brush automatically after picking a color (common UX)
        if hasattr(self, "active_tool"):
            self.active_tool.set("brush")
            self._apply_active_tool()

    def _clear_paint_selected(self):
        targets = self._get_selected_indices_or_current()
        if not targets:
            return
        for idx in targets:
            if idx in self.image_editors:
                self.image_editors[idx].clear_paint()
        self._display_current_image()

    # ----------------------------
    # Settings / Persistence
    # ----------------------------
    def _default_keymap(self) -> Dict[str, str]:
        """
        Photoshop-inspired defaults (approximate).
        Friendly strings are converted to Tk sequences at bind time.
        """
        return {
            "open_pdf": "Ctrl+O",
            "export_selected": "Ctrl+Shift+E",
            "settings": "Tab",
            "settings_alt": "Ctrl+,",
            "select_all": "Ctrl+A",
            "select_none": "Ctrl+D",
            "select_invert": "Ctrl+Shift+I",
            "select_wizard": "Ctrl+Alt+S",
            "select_odd": "Ctrl+Alt+O",
            "select_even": "Ctrl+Alt+E",
            "prev_page": "Left",
            "next_page": "Right",
            "zoom_in": "Ctrl++",
            "zoom_out": "Ctrl+-",
            "fit_to_window": "Ctrl+0",
            "actual_size": "Ctrl+1",
            "rotate_left": "Ctrl+L",
            "rotate_right": "Ctrl+R",
            "rotate_180": "Ctrl+Shift+R",
            "reset_ops": "Ctrl+Alt+0",
            "delete_selected": "Delete",
            # Tool shortcuts (Photoshop-like)
            "tool_hand": "H",
            "tool_zoom": "Z",
            "tool_crop": "C",
            "tool_brush": "B",
            "tool_eyedropper": "I",

            # Crop helpers
            "start_crop": "C",
            "clear_crop": "Shift+C",
            "clear_paint": "Ctrl+Alt+Delete",
        }

    def _load_settings(self) -> UISettings:
        defaults = UISettings(
            default_dpi=300,
            auto_fit_on_load=True,
            show_toolbar_hint=True,
            confirm_delete=True,
            keymap=self._default_keymap(),
        )
        try:
            if os.path.exists(self.settings_path):
                with open(self.settings_path, "r", encoding="utf-8") as f:
                    data = json.load(f) or {}
                # Merge with defaults
                keymap = defaults.keymap.copy()
                keymap.update(data.get("keymap") or {})
                return UISettings(
                    default_dpi=int(data.get("default_dpi", defaults.default_dpi)),
                    auto_fit_on_load=bool(data.get("auto_fit_on_load", defaults.auto_fit_on_load)),
                    show_toolbar_hint=bool(data.get("show_toolbar_hint", defaults.show_toolbar_hint)),
                    confirm_delete=bool(data.get("confirm_delete", defaults.confirm_delete)),
                    keymap=keymap,
                )
        except Exception:
            pass
        return defaults

    def _save_settings(self):
        try:
            with open(self.settings_path, "w", encoding="utf-8") as f:
                json.dump(asdict(self.settings), f, ensure_ascii=False, indent=2)
        except Exception as e:
            messagebox.showerror("Settings", f"Failed to save settings: {e}")

    def _apply_settings_to_ui(self):
        # Toolbar hint visibility
        if hasattr(self, "_toolbar_hint_label") and self._toolbar_hint_label is not None:
            if self.settings.show_toolbar_hint:
                self._toolbar_hint_label.pack(side=tk.LEFT, padx=8)
            else:
                self._toolbar_hint_label.pack_forget()

    def _open_settings_window(self):
        """Open Settings window with tabs (General, Shortcuts)."""
        if self._settings_window is not None and self._settings_window.winfo_exists():
            self._settings_window.lift()
            return

        win = tk.Toplevel(self.root)
        win.title("Settings")
        win.transient(self.root)
        win.grab_set()
        self._settings_window = win

        container = ttk.Frame(win, padding="10")
        container.pack(fill=tk.BOTH, expand=True)

        nb = ttk.Notebook(container)
        nb.pack(fill=tk.BOTH, expand=True)

        # General tab
        general = ttk.Frame(nb, padding="10")
        nb.add(general, text="General")

        dpi_var = tk.IntVar(value=int(self.settings.default_dpi))
        auto_fit_var = tk.BooleanVar(value=bool(self.settings.auto_fit_on_load))
        hint_var = tk.BooleanVar(value=bool(self.settings.show_toolbar_hint))
        confirm_del_var = tk.BooleanVar(value=bool(self.settings.confirm_delete))

        row = 0
        ttk.Label(general, text="Default DPI (for Open PDF):").grid(row=row, column=0, sticky=tk.W, pady=4)
        ttk.Entry(general, textvariable=dpi_var, width=10).grid(row=row, column=1, sticky=tk.W, pady=4)
        row += 1

        ttk.Checkbutton(general, text="Auto fit to window after load", variable=auto_fit_var).grid(
            row=row, column=0, columnspan=2, sticky=tk.W, pady=4
        )
        row += 1

        ttk.Checkbutton(general, text="Show toolbar hint text", variable=hint_var).grid(
            row=row, column=0, columnspan=2, sticky=tk.W, pady=4
        )
        row += 1

        ttk.Checkbutton(general, text="Confirm before delete", variable=confirm_del_var).grid(
            row=row, column=0, columnspan=2, sticky=tk.W, pady=4
        )
        row += 1

        # Shortcuts tab
        shortcuts = ttk.Frame(nb, padding="10")
        nb.add(shortcuts, text="Shortcuts")

        ttk.Label(shortcuts, text="Edit shortcuts (examples: Ctrl+O, Ctrl+Shift+I, Tab, Delete, Left)").pack(
            anchor=tk.W, pady=(0, 8)
        )

        # Build shortcut editor grid
        key_vars: Dict[str, tk.StringVar] = {}
        actions = [
            ("open_pdf", "Open PDF"),
            ("export_selected", "Export Selected"),
            ("settings", "Settings (Tab)"),
            ("settings_alt", "Settings (Ctrl+,)"),
            ("tool_hand", "Tool: Hand"),
            ("tool_zoom", "Tool: Zoom"),
            ("tool_crop", "Tool: Crop"),
            ("tool_brush", "Tool: Brush"),
            ("tool_eyedropper", "Tool: Eyedropper"),
            ("select_wizard", "Select Wizard"),
            ("select_all", "Select All"),
            ("select_none", "Select None (Deselect)"),
            ("select_invert", "Select Invert"),
            ("select_odd", "Select Odd"),
            ("select_even", "Select Even"),
            ("prev_page", "Previous Page"),
            ("next_page", "Next Page"),
            ("zoom_in", "Zoom In"),
            ("zoom_out", "Zoom Out"),
            ("fit_to_window", "Fit to Window"),
            ("actual_size", "Actual Size"),
            ("rotate_left", "Rotate -90"),
            ("rotate_right", "Rotate +90"),
            ("rotate_180", "Rotate 180"),
            ("reset_ops", "Reset operations"),
            ("delete_selected", "Delete selected"),
            ("start_crop", "Start crop"),
            ("clear_crop", "Clear crop"),
            ("clear_paint", "Clear paint"),
        ]

        grid = ttk.Frame(shortcuts)
        grid.pack(fill=tk.BOTH, expand=True)
        grid.columnconfigure(1, weight=1)

        for r, (action_id, label) in enumerate(actions):
            ttk.Label(grid, text=label).grid(row=r, column=0, sticky=tk.W, pady=2, padx=(0, 8))
            key_vars[action_id] = tk.StringVar(value=str((self.settings.keymap or {}).get(action_id, "")))
            ttk.Entry(grid, textvariable=key_vars[action_id]).grid(row=r, column=1, sticky=(tk.W, tk.E), pady=2)

        def apply_settings():
            # Persist general
            self.settings.default_dpi = max(50, int(dpi_var.get()))
            self.settings.auto_fit_on_load = bool(auto_fit_var.get())
            self.settings.show_toolbar_hint = bool(hint_var.get())
            self.settings.confirm_delete = bool(confirm_del_var.get())

            # Persist shortcuts
            if self.settings.keymap is None:
                self.settings.keymap = {}
            for k, v in key_vars.items():
                self.settings.keymap[k] = v.get().strip()

            self._save_settings()
            self._apply_settings_to_ui()
            self._bind_shortcuts()
            messagebox.showinfo("Settings", "Saved.")

        btns = ttk.Frame(container)
        btns.pack(fill=tk.X, pady=(10, 0))
        ttk.Button(btns, text="Save", command=apply_settings).pack(side=tk.RIGHT)
        ttk.Button(btns, text="Close", command=win.destroy).pack(side=tk.RIGHT, padx=(0, 8))

        win.bind("<Escape>", lambda _e: win.destroy())

    # ----------------------------
    # Selection helpers
    # ----------------------------
    def _get_selected_indices_or_current(self) -> List[int]:
        """Return selected indices; if none, return [current_index] if valid."""
        selected = list(self.image_listbox.curselection())
        if selected:
            return selected
        if 0 <= self.current_index < len(self.processed_images):
            return [self.current_index]
        return []

    def _apply_selection_indices(self, indices: List[int]):
        """Apply a set of indices to the listbox selection."""
        self.image_listbox.selection_clear(0, tk.END)
        for i in sorted(set(indices)):
            if 0 <= i < len(self.processed_images):
                self.image_listbox.selection_set(i)
        if indices:
            first = sorted(set(indices))[0]
            self.image_listbox.see(first)
            # Keep a stable "current" page for viewing
            self.current_index = first
            self._display_current_image()

    def _select_all(self):
        self._apply_selection_indices(list(range(len(self.processed_images))))

    def _select_none(self):
        self.image_listbox.selection_clear(0, tk.END)

    def _select_invert(self):
        current = set(self.image_listbox.curselection())
        all_idx = set(range(len(self.processed_images)))
        self._apply_selection_indices(sorted(all_idx - current))

    def _select_odd(self):
        # odd page numbers => indices 0,2,4,...
        self._apply_selection_indices([i for i in range(len(self.processed_images)) if (i + 1) % 2 == 1])

    def _select_even(self):
        # even page numbers => indices 1,3,5,...
        self._apply_selection_indices([i for i in range(len(self.processed_images)) if (i + 1) % 2 == 0])

    def _parse_selection_pattern(self, pattern: str) -> List[int]:
        """
        Parse selection patterns:
        - "1-2" selects page 1..2
        - "1,3,5" selects pages 1,3,5
        - You can mix: "1-3,8,10-12"

        Returns 0-based indices.
        """
        pattern = (pattern or "").strip()
        if not pattern:
            return []

        max_page = len(self.processed_images)
        selected: set[int] = set()

        # Split on commas
        parts = [p.strip() for p in pattern.split(",") if p.strip()]
        for part in parts:
            # Range
            if "-" in part:
                m = re.fullmatch(r"\s*(\d+)\s*-\s*(\d+)\s*", part)
                if not m:
                    raise ValueError(f"Invalid range: '{part}'")
                a = int(m.group(1))
                b = int(m.group(2))
                if a > b:
                    a, b = b, a
                for page in range(a, b + 1):
                    if 1 <= page <= max_page:
                        selected.add(page - 1)
            else:
                if not re.fullmatch(r"\d+", part):
                    raise ValueError(f"Invalid page number: '{part}'")
                page = int(part)
                if 1 <= page <= max_page:
                    selected.add(page - 1)

        return sorted(selected)

    def _apply_selection_pattern(self):
        try:
            indices = self._parse_selection_pattern(self.select_pattern_var.get())
            if not indices:
                messagebox.showwarning("Select", "No pages matched the pattern.")
                return
            self._apply_selection_indices(indices)
        except Exception as e:
            messagebox.showerror("Select", str(e))

    def _open_select_wizard_dialog(self):
        """Open a wizard dialog for multi-select patterns (menu-accessible)."""
        if not self.processed_images:
            messagebox.showwarning("Select", "No images loaded.")
            return

        win = tk.Toplevel(self.root)
        win.title("Select Wizard")
        win.transient(self.root)
        win.grab_set()

        frm = ttk.Frame(win, padding="10")
        frm.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frm, text="Pattern (examples: 1-2, 1,3,5, 1-3,8,10-12):").pack(anchor=tk.W)
        var = tk.StringVar(value=self.select_pattern_var.get())
        ent = ttk.Entry(frm, textvariable=var, width=40)
        ent.pack(fill=tk.X, pady=(4, 8))
        ent.focus_set()

        btn_row1 = ttk.Frame(frm)
        btn_row1.pack(fill=tk.X)
        ttk.Button(btn_row1, text="All", command=lambda: self._select_all()).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 4))
        ttk.Button(btn_row1, text="None", command=lambda: self._select_none()).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=4)
        ttk.Button(btn_row1, text="Invert", command=lambda: self._select_invert()).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(4, 0))

        btn_row2 = ttk.Frame(frm)
        btn_row2.pack(fill=tk.X, pady=(6, 0))
        ttk.Button(btn_row2, text="Odd", command=lambda: self._select_odd()).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 4))
        ttk.Button(btn_row2, text="Even", command=lambda: self._select_even()).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(4, 0))

        def apply_and_close():
            self.select_pattern_var.set(var.get())
            self._apply_selection_pattern()
            win.destroy()

        btn_row3 = ttk.Frame(frm)
        btn_row3.pack(fill=tk.X, pady=(10, 0))
        ttk.Button(btn_row3, text="Apply", command=apply_and_close).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 4))
        ttk.Button(btn_row3, text="Close", command=win.destroy).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(4, 0))

        ent.bind("<Return>", lambda _e: apply_and_close())
    
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
            images = processor.process_pdf(pdf_path, dpi=int(self.settings.default_dpi))
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

        if getattr(self.settings, "confirm_delete", True):
            if not messagebox.askyesno("Delete", f"Delete {len(selected)} selected image(s)?"):
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
            # Keep current_index stable when multi-selecting; if current is not
            # in selection, move it to the first selected.
            if self.current_index not in selected:
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
        targets = self._get_selected_indices_or_current()
        if not targets:
            return

        for idx in targets:
            if idx not in self.image_editors:
                processed = self.processed_images[idx]
                editor = ImageEditor()
                editor.set_image(processed.pil_image)
                self.image_editors[idx] = editor
            self.image_editors[idx].rotate(angle)

        # Refresh display for the current page
        self._display_current_image()
    
    def _on_brightness_change(self, value):
        """Handle brightness slider change."""
        targets = self._get_selected_indices_or_current()
        if not targets:
            return

        factor = float(value)
        for idx in targets:
            if idx not in self.image_editors:
                processed = self.processed_images[idx]
                editor = ImageEditor()
                editor.set_image(processed.pil_image)
                self.image_editors[idx] = editor
            self.image_editors[idx].set_brightness(factor)

        self._display_current_image()
        self._update_brightness_label()
    
    def _update_brightness_label(self):
        """Update brightness label."""
        self.brightness_label.config(text=f"{self.brightness_var.get():.2f}")
    
    def _on_contrast_change(self, value):
        """Handle contrast slider change."""
        targets = self._get_selected_indices_or_current()
        if not targets:
            return

        factor = float(value)
        for idx in targets:
            if idx not in self.image_editors:
                processed = self.processed_images[idx]
                editor = ImageEditor()
                editor.set_image(processed.pil_image)
                self.image_editors[idx] = editor
            self.image_editors[idx].set_contrast(factor)

        self._display_current_image()
        self._update_contrast_label()
    
    def _update_contrast_label(self):
        """Update contrast label."""
        self.contrast_label.config(text=f"{self.contrast_var.get():.2f}")
    
    def _reset_operations(self):
        """Reset all operations on current image."""
        targets = self._get_selected_indices_or_current()
        if not targets:
            return

        for idx in targets:
            if idx in self.image_editors:
                self.image_editors[idx].reset()
            else:
                # If no editor exists yet, nothing to reset.
                pass

        self.brightness_var.set(1.0)
        self.contrast_var.set(1.0)
        self._display_current_image()
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
                x_left = int(img_w // 2 + rel_x1)
                y_top = int(img_h // 2 + rel_y1)
                x_right = int(img_w // 2 + rel_x2)
                y_bottom = int(img_h // 2 + rel_y2)

                # Normalize drag direction + clamp
                left = max(0, min(img_w, min(x_left, x_right)))
                right = max(0, min(img_w, max(x_left, x_right)))
                top = max(0, min(img_h, min(y_top, y_bottom)))
                bottom = max(0, min(img_h, max(y_top, y_bottom)))

                # Ensure valid crop box
                if right > left and bottom > top:
                    # Apply to all selected (or current if none selected)
                    targets = self._get_selected_indices_or_current()

                    # Use normalized box so it can be applied across different image sizes
                    fx1 = left / float(img_w)
                    fx2 = right / float(img_w)
                    fy1 = top / float(img_h)
                    fy2 = bottom / float(img_h)

                    for idx in targets:
                        if idx not in self.image_editors:
                            processed = self.processed_images[idx]
                            editor_t = ImageEditor()
                            editor_t.set_image(processed.pil_image)
                            self.image_editors[idx] = editor_t
                        editor = self.image_editors[idx]

                        base = editor.get_current_image()
                        if base is None:
                            continue
                        w2, h2 = base.size
                        crop_box = (
                            max(0, min(w2, int(fx1 * w2))),
                            max(0, min(h2, int(fy1 * h2))),
                            max(0, min(w2, int(fx2 * w2))),
                            max(0, min(h2, int(fy2 * h2))),
                        )
                        if crop_box[2] > crop_box[0] and crop_box[3] > crop_box[1]:
                            editor.set_crop(crop_box)

                    self._display_current_image()
        
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
        targets = self._get_selected_indices_or_current()
        if not targets:
            return

        for idx in targets:
            if idx in self.image_editors:
                self.image_editors[idx].set_crop(None)

        self._display_current_image()

    # ----------------------------
    # Shortcut binding
    # ----------------------------
    def _tk_sequence_from_friendly(self, s: str) -> Optional[str]:
        """
        Convert friendly strings like:
          - Ctrl+O
          - Ctrl+Shift+I
          - Tab
          - Ctrl+,
          - Ctrl++
        into Tk sequences like <Control-o>.
        """
        if not s:
            return None
        s = s.strip()
        if not s:
            return None
        if s.startswith("<") and s.endswith(">"):
            return s

        # Special-case "++" (common way users write Ctrl++). In Tk, the keysym is "plus".
        # Example: "Ctrl++" -> "Ctrl+Plus"
        if s.endswith("++"):
            s = s[:-1] + "Plus"
        # Also allow "Ctrl+" as shorthand
        if s.endswith("+"):
            s = s + "Plus"

        parts = [p.strip() for p in s.split("+") if p.strip()]
        if not parts:
            return None

        mods = []
        key = parts[-1]
        mod_parts = parts[:-1]

        mod_map = {
            "ctrl": "Control",
            "control": "Control",
            "shift": "Shift",
            "alt": "Alt",
        }
        for m in mod_parts:
            mm = mod_map.get(m.lower())
            if mm:
                mods.append(mm)

        key_map = {
            "tab": "Tab",
            "delete": "Delete",
            "backspace": "BackSpace",
            "esc": "Escape",
            "escape": "Escape",
            "left": "Left",
            "right": "Right",
            "up": "Up",
            "down": "Down",
            ",": "comma",
            ".": "period",
            "+": "plus",
            "plus": "plus",
            "-": "minus",
        }

        key = key_map.get(key.lower(), key)
        if len(key) == 1:
            key = key.lower()

        seq = "<" + "-".join(mods + [key]) + ">"
        return seq

    def _should_ignore_shortcut(self, event: tk.Event) -> bool:
        w = getattr(event, "widget", None)
        if w is None:
            return False
        # Avoid stealing keystrokes while typing in inputs
        if isinstance(w, (tk.Entry, ttk.Entry, tk.Text)):
            return True
        return False

    def _bind_shortcuts(self):
        # Unbind previous shortcuts
        try:
            for action_id, seq in list(self._shortcut_bind_ids.items()):
                if seq:
                    self.root.unbind_all(seq)
        except Exception:
            pass
        self._shortcut_bind_ids = {}

        keymap = self.settings.keymap or {}

        def bind(action_id: str, friendly: str, handler, *, break_default: bool = True):
            seq = self._tk_sequence_from_friendly(friendly)
            if not seq:
                return

            def _wrapped(event):
                # Special case: Tab should still work for focus traversal inside text inputs
                if action_id in ("settings", "settings_alt"):
                    if self._should_ignore_shortcut(event):
                        return None
                else:
                    if self._should_ignore_shortcut(event):
                        return None

                handler()
                return "break" if break_default else None

            try:
                self.root.bind_all(seq, _wrapped)
            except tk.TclError as e:
                # Don't prevent app startup due to a bad shortcut string.
                print(f"[WARN] Failed to bind shortcut {action_id}='{friendly}' -> {seq}: {e}")
                return
            self._shortcut_bind_ids[action_id] = seq

        # Actions
        bind("open_pdf", keymap.get("open_pdf", ""), self._load_from_module1)
        bind("export_selected", keymap.get("export_selected", ""), self._export_selected)
        bind("settings", keymap.get("settings", ""), self._open_settings_window)
        bind("settings_alt", keymap.get("settings_alt", ""), self._open_settings_window)

        bind("select_all", keymap.get("select_all", ""), self._select_all)
        bind("select_none", keymap.get("select_none", ""), self._select_none)
        bind("select_invert", keymap.get("select_invert", ""), self._select_invert)
        bind("select_wizard", keymap.get("select_wizard", ""), self._open_select_wizard_dialog)
        bind("select_odd", keymap.get("select_odd", ""), self._select_odd)
        bind("select_even", keymap.get("select_even", ""), self._select_even)

        bind("prev_page", keymap.get("prev_page", ""), self._prev_image)
        bind("next_page", keymap.get("next_page", ""), self._next_image)

        bind("zoom_in", keymap.get("zoom_in", ""), lambda: self.viewer.zoom_in())
        bind("zoom_out", keymap.get("zoom_out", ""), lambda: self.viewer.zoom_out())
        bind("fit_to_window", keymap.get("fit_to_window", ""), lambda: self.viewer.fit_to_window())
        bind("actual_size", keymap.get("actual_size", ""), lambda: self.viewer.reset_view())

        bind("rotate_left", keymap.get("rotate_left", ""), lambda: self._rotate(-90))
        bind("rotate_right", keymap.get("rotate_right", ""), lambda: self._rotate(90))
        bind("rotate_180", keymap.get("rotate_180", ""), lambda: self._rotate(180))
        bind("reset_ops", keymap.get("reset_ops", ""), self._reset_operations)

        bind("delete_selected", keymap.get("delete_selected", ""), self._delete_selected)
        # Tool selection
        bind("tool_hand", keymap.get("tool_hand", ""), lambda: self._set_tool("hand"))
        bind("tool_zoom", keymap.get("tool_zoom", ""), lambda: self._set_tool("zoom"))
        bind("tool_crop", keymap.get("tool_crop", ""), lambda: self._set_tool("crop"))
        bind("tool_brush", keymap.get("tool_brush", ""), lambda: self._set_tool("brush"))
        bind("tool_eyedropper", keymap.get("tool_eyedropper", ""), lambda: self._set_tool("eyedropper"))

        # Legacy crop shortcut (kept for compatibility)
        bind("start_crop", keymap.get("start_crop", ""), lambda: self._set_tool("crop"))
        bind("clear_crop", keymap.get("clear_crop", ""), self._clear_crop)
        bind("clear_paint", keymap.get("clear_paint", ""), self._clear_paint_selected)

    def _set_tool(self, tool: str):
        if hasattr(self, "active_tool"):
            self.active_tool.set(tool)
            self._apply_active_tool()


def main():
    """Main entry point for the UI module."""
    root = tk.Tk()
    app = ImageManagementUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
