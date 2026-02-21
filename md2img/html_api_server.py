"""
HTML to Image HTTP API Service
High QPS support using ThreadPoolExecutor and Thread-Local Playwright instances.
Port: 8002
"""

import io
import threading
import time
import asyncio
import random
import logging
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Optional
from PIL import Image
from playwright.sync_api import sync_playwright, Browser, Page
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
import uvicorn

# Thread-local storage for independent browser instances per thread
_thread_local = threading.local()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("html_renderer")

class HtmlRequest(BaseModel):
    """HTML Request Model"""
    content: str
    format: Optional[str] = "png"  # supports png, jpeg
    width: Optional[int] = 800     # Optional viewport width control

def get_thread_local_browser() -> Browser:
    """
    Get browser instance for current thread.
    Initialize if not exists.
    """
    if (
        not hasattr(_thread_local, "playwright_instance")
        or _thread_local.playwright_instance is None
    ):
        _thread_local.playwright_context_manager = sync_playwright()
        _thread_local.playwright_instance = (
            _thread_local.playwright_context_manager.__enter__()
        )
        _thread_local.browser = _thread_local.playwright_instance.chromium.launch(
            headless=True,
            args=[
                "--disable-dev-shm-usage",
                "--disable-gpu",
                "--no-sandbox",
            ],
        )

    return _thread_local.browser

# Global Executor
executor: Optional[ThreadPoolExecutor] = None

def get_executor() -> ThreadPoolExecutor:
    """Singleton Executor"""
    global executor
    if executor is None:
        # Adjust max_workers based on your server's CPU/Memory
        executor = ThreadPoolExecutor(max_workers=10, thread_name_prefix="pw_render")
    return executor

def _html_to_image_sync(response_text: str, image_format: str = "png", viewport_width: int = 800) -> bytes:
    """
    Synchronous function to render HTML to image via Playwright.
    """
    start_time = time.time()

    # Wrap raw HTML in a structured container to ensure CSS application and screenshot bounding
    html_content = response_text.split("```html")[1].split("```")[0].strip()
    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            body {{
                background-color: transparent;
                margin: 0;
                padding: 0;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            }}
            .wrapper {{
                padding: 20px;
                display: inline-block; /* Allows the div to shrink-wrap content */
                min-width: 100px;
                background-color: white; 
            }}
            /* Basic default styles to make raw HTML look decent */
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
            th {{ padding-top: 12px; padding-bottom: 12px; text-align: left; background-color: #04AA6D; color: white; }}
        </style>
    </head>
    <body>
        <div class="wrapper">
            {html_content}
        </div>
    </body>
    </html>
    """
    # full_html = html_content.replace("```html", "").replace("```", "").strip()

    browser = get_thread_local_browser()
    page: Optional[Page] = None

    try:
        page = browser.new_page(viewport={"width": viewport_width, "height": 1000})
        page.set_content(full_html)
        
        # Wait for hydration
        page.wait_for_load_state("domcontentloaded", timeout=5000)

        # Locate the wrapper to take a screenshot of ONLY the content
        element_handle = page.locator(".wrapper")
        
        # Fallback to full page if element issue occurs
        if element_handle.count() > 0:
            screenshot_bytes = element_handle.screenshot(type=image_format, omit_background=True)
        else:
            screenshot_bytes = page.screenshot(full_page=True, type=image_format)

        # Verify output with PIL (Optional: could add cropping here if needed)
        # pil_image = Image.open(io.BytesIO(screenshot_bytes))
        
        process_time = time.time() - start_time
        if random.random() < 0.1: # Log 10% of requests
            logger.info(f"Render time: {process_time:.3f}s, Size: {len(html_content)} chars")

        return screenshot_bytes

    except Exception as e:
        logger.error(f"Playwright error: {e}")
        raise e
    finally:
        if page:
            page.close()

async def html_to_image(html_content: str, image_format: str = "png", width: int = 800) -> bytes:
    loop = asyncio.get_event_loop()
    exec_pool = get_executor()
    return await loop.run_in_executor(
        exec_pool, _html_to_image_sync, html_content, image_format, width
    )

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting HTML to Image Service on Port 8002...")
    get_executor()
    yield
    print("Shutting down service...")
    global executor
    if executor:
        executor.shutdown(wait=True)

app = FastAPI(title="HTML to Image API", version="1.0.0", lifespan=lifespan)

@app.post("/render")
async def render_html(request: HtmlRequest):
    if not request.content:
        raise HTTPException(status_code=400, detail="HTML content cannot be empty")

    if request.format not in ["png", "jpeg"]:
        raise HTTPException(status_code=400, detail="Format must be png or jpeg")

    try:
        start_time = time.time()
        image_bytes = await html_to_image(request.content, request.format, request.width)
        total_time = time.time() - start_time
        
        return Response(
            content=image_bytes,
            media_type=f"image/{request.format}",
            headers={
                "X-Processing-Time": f"{total_time:.3f}",
            },
        )
    except Exception as e:
        logger.error(f"Render failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9001, log_level="info", access_log=False)