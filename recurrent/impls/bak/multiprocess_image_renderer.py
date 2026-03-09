# mp_text_renderer.py
from concurrent.futures import ProcessPoolExecutor
from io import BytesIO
from typing import Dict, Iterable, List, Tuple, Optional
import os

from PIL import Image, ImageDraw, ImageFont
from array import array
from bisect import bisect_right

# ---- import your function (must be top-level importable) ----
# from your_module import text_to_image_optimized
# Paste-in if needed:
def text_to_image_optimized_fast(
    text: str,
    font: ImageFont.FreeTypeFont,
    dummy_draw: ImageDraw.ImageDraw,
    width: int = 800,
    bg_color: str = "white",
    text_color: str = "black",
    padding: Tuple[int, int, int, int] = (6, 6, 6, 6),
    line_spacing: float = 1.25,
    render_mode_L_then_convert: bool = True,
) -> Image.Image:
    """
    Faster version with support for hard breaks:
      - '\\n'   -> force a line break
      - '\\n\\n' -> paragraph break (inserts an empty line)
    Avoids repeated textlength on growing strings, caches widths, and splits long words via
    binary search on cumulative advances.
    """
    # --- helpers & locals (bind for speed) ---
    getlength = getattr(font, "getlength", None)
    if getlength is None:
        # fallback to draw.textlength
        def measure(s: str) -> float:
            return dummy_draw.textlength(s, font=font)
    else:
        def measure(s: str) -> float:
            return getlength(s)

    left, top, right, bottom = padding
    usable_width = width - (left + right)

    # Handle empty input quickly
    if not text:
        h = top + bottom
        img = Image.new("RGB", (width, h), color=bg_color)
        return img

    # --- caches ---
    word_w_cache: Dict[str, float] = {}
    char_w_cache: Dict[str, float] = {}

    def width_word(w: str) -> float:
        ww = word_w_cache.get(w)
        if ww is None:
            ww = measure(w)
            word_w_cache[w] = ww
        return ww

    # Cache a single space width
    space_w = word_w_cache.get(" ")
    if space_w is None:
        space_w = measure(" ")
        word_w_cache[" "] = space_w

    from array import array as _array
    from bisect import bisect_right as _bisect_right

    def cumulative_char_advances(w: str) -> _array:
        """
        Return cumulative advance widths for characters in w.
        Uses per-char advances (fast, close-enough without shaping).
        """
        arr = _array("f", [0.0])
        aw = 0.0
        for ch in w:
            cw = char_w_cache.get(ch)
            if cw is None:
                cw = measure(ch)
                char_w_cache[ch] = cw
            aw += cw
            arr.append(aw)
        return arr  # len = len(w) + 1

    # --- line breaking with support for \n and \n\n ---
    lines: List[str] = []
    current: List[str] = []
    cur_width = 0.0

    def flush_current():
        nonlocal current, cur_width
        lines.append("".join(current))
        current = []
        cur_width = 0.0

    paragraphs = text.split("\n\n")  # paragraph = blocks separated by blank line
    for p_idx, paragraph in enumerate(paragraphs):
        # Within a paragraph, single '\n' forces a line break (not a blank line)
        # We inject explicit "\n" tokens so we can see them in the main loop.
        # .split() keeps the "\n" tokens separate because of the padding spaces.
        tokens = paragraph.replace("\n", " \n ").split()

        ti = 0
        while ti < len(tokens):
            tok = tokens[ti]

            # Forced line break
            if tok == "\n":
                flush_current()
                ti += 1
                continue

            # Normal word handling
            ww = width_word(tok)
            add_width = (space_w if current else 0.0) + ww

            if cur_width + add_width <= usable_width:
                if current:
                    current.append(" ")
                current.append(tok)
                cur_width += add_width
                ti += 1
                continue

            # doesn't fit: flush current if any
            if current:
                flush_current()

            # If the word itself fits on an empty line, place it
            if ww <= usable_width:
                current = [tok]
                cur_width = ww
                ti += 1
                continue

            # Word longer than line: split efficiently
            csum = cumulative_char_advances(tok)
            start = 0
            while start < len(tok):
                remain = usable_width
                base = csum[start]
                target = base + remain
                idx = _bisect_right(csum, target) - 1
                if idx <= start:
                    idx = start + 1  # ensure progress
                piece = tok[start:idx]
                lines.append(piece)
                start = idx

            ti += 1

        # End of paragraph: flush any pending text on the current line
        if current:
            flush_current()

        # Paragraph break (between paragraphs only): add a blank line
        if p_idx < len(paragraphs) - 1:
            lines.append("")  # empty line to represent the blank paragraph gap

    # --- size & render ---
    ascent, descent = font.getmetrics()
    base_h = ascent + descent
    n = len(lines)

    if n == 0:
        total_h = top + bottom
    else:
        # total height = paddings + base_h per line + spacing between lines
        total_h = top + bottom + n * base_h + (n - 1) * int(base_h * (line_spacing - 1.0))

    mode = "L" if render_mode_L_then_convert else "RGB"
    img = Image.new(mode, (width, total_h), color=255 if mode == "L" else bg_color)
    draw = ImageDraw.Draw(img)
    x, y = left, top
    line_px = int(base_h * line_spacing)

    if mode == "L":
        for line in lines:
            # empty strings draw nothing but still advance y (blank lines)
            if line:
                draw.text((x, y), line, font=font, fill=0)
            y += line_px
        img = img.convert("RGB")
        if bg_color != "white" or text_color != "black":
            # Keep default for perf; for arbitrary colors, add a tint/composite pass here if needed.
            pass
    else:
        for line in lines:
            if line:
                draw.text((x, y), line, font=font, fill=text_color)
            y += line_px

    return img
# -------------------- worker side (top-level!) --------------------

_FONT: Optional[ImageFont.FreeTypeFont] = None
_DUMMY_DRAW: Optional[ImageDraw.ImageDraw] = None
_CFG: Dict = {}

def _init_worker(
    font_path: str,
    font_size: int,
    width: int,
    bg_color: str,
    text_color: str,
    line_spacing: float,
):
    """Runs once per process to build non-picklable objects locally."""
    global _FONT, _DUMMY_DRAW, _CFG
    _FONT = ImageFont.truetype(font_path, size=font_size)
    # Use a tiny scratch canvas for measurement
    scratch = Image.new("L", (8, 8))
    _DUMMY_DRAW = ImageDraw.Draw(scratch)
    _CFG = dict(width=width, bg_color=bg_color, text_color=text_color, line_spacing=line_spacing)

def _render_to_png_bytes(text: str) -> Tuple[str, bytes]:
    """Render a single text to PNG bytes. Returns (text, png_bytes)."""
    img = text_to_image_optimized_fast(
        text=text,
        font=_FONT,                  # set in _init_worker
        dummy_draw=_DUMMY_DRAW,      # set in _init_worker
        width=_CFG["width"],
        bg_color=_CFG["bg_color"],
        text_color=_CFG["text_color"],
        line_spacing=_CFG["line_spacing"],
    )
    buf = BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return text, buf.getvalue()

# -------------------- public class --------------------

class MPTextRenderer:
    """
    Multi-process text->image renderer.

    You fill the init blanks per your environment (font path/size, width, etc.).
    Workers rebuild font/draw internally so nothing non-picklable crosses processes.
    """

    def __init__(
        self,
        *,
        # You said to “leave these in __init__” — fill them if you want to also use
        # your single-process path elsewhere. They are NOT sent to worker processes.
        font=None,                 # type: Optional[ImageFont.FreeTypeFont]
        dummy_draw=None,           # type: Optional[ImageDraw.ImageDraw]
        img_width: int = 800,

        # Required for multiprocess safety:
        font_path: str = "DejaVuSans.ttf",
        font_size: int = 10,

        # Rendering options:
        bg_color: str = "white",
        text_color: str = "black",
        line_spacing: float = 1,

        # Pool options:
        max_workers: Optional[int] = None,
        chunksize: int = 16,
    ):
        # Your local (main-process) copies, for convenience elsewhere
        self.font = font                # you fill if you want
        self.dummy_draw = dummy_draw    # you fill if you want
        self.img_width = img_width      # you fill as needed

        # Config actually used by worker processes
        self.font_path = font_path
        self.font_size = font_size
        self.bg_color = bg_color
        self.text_color = text_color
        self.line_spacing = line_spacing

        self.max_workers = max_workers or int(os.cpu_count() * 0.8) or 1
        self.chunksize = max(1, chunksize)

    def _unique_preserve_order(self, texts: Iterable[str]) -> List[str]:
        seen, out = set(), []
        for t in texts:
            if t not in seen:
                seen.add(t)
                out.append(t)
        return out

    def render(self, texts: List[str]) -> Tuple[List[Image.Image], Dict[str, Image.Image]]:
        """
        Render a list of texts.
        Returns:
            images_in_order: List[PIL.Image] aligned to input order
            cache_dict: Dict[text, PIL.Image] for the unique set
        """
        if not texts:
            return [], {}

        unique_texts = self._unique_preserve_order(texts)

        with ProcessPoolExecutor(
            max_workers=self.max_workers,
            initializer=_init_worker,
            initargs=(
                self.font_path,
                self.font_size,
                self.img_width,
                self.bg_color,
                self.text_color,
                self.line_spacing,
            ),
        ) as ex:
            results = list(ex.map(_render_to_png_bytes, unique_texts, chunksize=self.chunksize))

        # Rehydrate PIL images and build dict
        cache_dict: Dict[str, Image.Image] = {}
        for t, png in results:
            img = Image.open(BytesIO(png)).convert("RGB")  # force load & detach from BytesIO
            cache_dict[t] = img

        images_in_order = [cache_dict[t] for t in texts]
        return images_in_order, cache_dict
