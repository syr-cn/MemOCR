from concurrent.futures import ProcessPoolExecutor
from io import BytesIO
from typing import Dict, List, Tuple, Optional
import os

from PIL import Image
from recurrent.impls.markdown_render_v2 import markdown_to_image
import sys

# ==============================================================================
# Worker functions (must be top-level for multiprocessing)
# ==============================================================================

def _init_worker():
    """Initialize worker process. No-op since markdown_to_image doesn't need initialization."""
    pass

def _render_markdown_task(text: str) -> Tuple[str, bytes]:
    # redirect output to /dev/null to avoid logging to stdout
    with open(os.devnull, 'w') as f:
        sys.stdout = f
        # sys.stderr = f # do not discard stderr
        img = markdown_to_image(text)
    buf = BytesIO()
    
    # REMOVED optimize=True to save CPU time. 
    # The default PNG compression is usually good enough for IPC 
    # without the heavy "optimization" pass.
    img.save(buf, format="PNG") 
    
    return text, buf.getvalue()


class MPMarkdownRenderer:
    def __init__(
        self,
        font_path: str = "DejaVuSans.ttf",  # Not used by markdown_to_image, kept for compatibility
        bold_font_path: Optional[str] = None,  # Not used by markdown_to_image, kept for compatibility
        font_size: int = 10,  # Not used by markdown_to_image, kept for compatibility
        img_width: int = 800,  # Not used by markdown_to_image, kept for compatibility
        bg_color: str = "white",  # Not used by markdown_to_image, kept for compatibility
        text_color: str = "black",  # Not used by markdown_to_image, kept for compatibility
        line_spacing: float = 1.2,  # Not used by markdown_to_image, kept for compatibility
        max_workers: Optional[int] = None,
    ):
        self.font_path = font_path
        self.bold_font_path = bold_font_path
        self.font_size = font_size
        self.img_width = img_width
        self.bg_color = bg_color
        self.text_color = text_color
        self.line_spacing = line_spacing
        self.max_workers = max_workers or int(os.cpu_count() * 0.9) or 1
        print(f'MPMarkdownRenderer initialized with {self.max_workers} workers')

    def render(self, texts: List[str]) -> Tuple[List[Image.Image], Dict[str, Image.Image]]:
        unique_texts = list(set(texts))

        with ProcessPoolExecutor(
            max_workers=self.max_workers,
            initializer=_init_worker,
        ) as ex:
            results = list(ex.map(_render_markdown_task, unique_texts))

        cache_dict = {}
        for t, png in results:
            img = Image.open(BytesIO(png)).convert("RGB")
            cache_dict[t] = img

        return [cache_dict[t] for t in texts], cache_dict
