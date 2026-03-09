"""
Markdown 转图片 HTTP API 服务
支持高 QPS 请求，使用浏览器实例池进行性能优化
"""

import markdown
import io
import queue
import threading
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Optional
from PIL import Image, ImageChops
from playwright.sync_api import sync_playwright, Browser, Page
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
import uvicorn
import random
# 线程本地存储，为每个线程维护独立的浏览器实例
_thread_local = threading.local()
import logging

logger = logging.getLogger(__name__)


class MarkdownRequest(BaseModel):
    """Markdown 请求模型"""

    content: str
    format: Optional[str] = "png"  # 支持 png, jpeg


def get_thread_local_browser() -> Browser:
    """
    获取当前线程的浏览器实例（使用线程本地存储）
    每个线程都有自己的 Playwright 上下文和浏览器实例
    """
    if (
        not hasattr(_thread_local, "playwright_instance")
        or _thread_local.playwright_instance is None
    ):
        # 为当前线程初始化 Playwright
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


def cleanup_thread_local_browser():
    """清理当前线程的浏览器实例"""
    if hasattr(_thread_local, "browser") and _thread_local.browser:
        try:
            _thread_local.browser.close()
        except:
            pass
        _thread_local.browser = None

    if (
        hasattr(_thread_local, "playwright_context_manager")
        and _thread_local.playwright_context_manager
    ):
        try:
            _thread_local.playwright_context_manager.__exit__(None, None, None)
        except:
            pass
        _thread_local.playwright_context_manager = None
        _thread_local.playwright_instance = None


# 全局执行器
executor: Optional[ThreadPoolExecutor] = None


def get_executor() -> ThreadPoolExecutor:
    """获取线程池执行器（单例模式）"""
    global executor
    if executor is None:
        executor = ThreadPoolExecutor(max_workers=20, thread_name_prefix="playwright")
    return executor


def trim_whitespace(im: Image.Image) -> Image.Image:
    """
    自动裁剪图片，移除背景色边框

    Args:
        im: PIL Image 对象

    Returns:
        裁剪后的 PIL Image 对象
    """
    if im.mode == "RGBA":
        bg_color = (255, 255, 255, 0)
        bg = Image.new("RGBA", im.size, bg_color)
        diff = ImageChops.difference(im, bg)
        bbox = diff.getbbox()
        if not bbox and im.getbbox():
            bbox = im.getbbox()
    else:
        bg = Image.new(im.mode, im.size, im.getpixel((0, 0)))
        diff = ImageChops.difference(im, bg)
        bbox = diff.getbbox()

    if not bbox:
        return im

    padding = 10
    bbox = (
        max(bbox[0] - padding, 0),
        max(bbox[1] - padding, 0),
        min(bbox[2] + padding, im.width),
        min(bbox[3] + padding, im.height),
    )

    if bbox:
        return im.crop(bbox)
    return im


def _markdown_to_image_sync(markdown_text: str, image_format: str = "png") -> bytes:
    """
    将 Markdown 文本转换为图片

    Args:
        markdown_text: Markdown 文本内容
        image_format: 图片格式 (png 或 jpeg)

    Returns:
        图片的字节数据
    """
    start_time = time.time()

    # 清理和转换 Markdown
    markdown_text = markdown_text.strip().strip("`")
    html_content = markdown.markdown(markdown_text, extensions=["extra"])

    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            body {{
                background-color: white;
                margin: 2em;
                padding: 2em;
            }}
            .content {{
                padding: 0px;
                display: inline-block;
                white-space: pre-wrap;
            }}
            h1, h2, h3, h4, h5, h6, p, ul, ol, blockquote {{
                margin-top: 0.5em;
                margin-bottom: 0.5em;
                padding: 0;
            }}
            ul, ol {{
                margin-left: 20px;
            }}
            code {{
                background-color: #f4f4f4;
                padding: 2px 4px;
                border-radius: 3px;
                font-family: 'Courier New', monospace;
            }}
            pre {{
                background-color: #f4f4f4;
                padding: 10px;
                border-radius: 5px;
                overflow-x: auto;
            }}
        </style>
    </head>
    <body>
        <div class="content">
            {html_content}
        </div>
    </body>
    </html>
    """

    # 获取当前线程的浏览器实例（线程本地存储）
    browser = get_thread_local_browser()
    page: Optional[Page] = None

    try:
        # 创建新页面
        page = browser.new_page()

        # 设置内容
        page.set_content(full_html)

        # 等待内容加载（使用更短的超时时间以提升性能）
        page.wait_for_load_state("domcontentloaded", timeout=5000)

        # 获取内容元素的边界框
        element_handle = page.locator(".content")
        box = element_handle.bounding_box()

        if box and box["width"] >= 10 and box["height"] >= 10:
            clip_area = {
                "x": box["x"],
                "y": box["y"],
                "width": box["width"],
                "height": box["height"],
            }
            screenshot_bytes = page.screenshot(
                clip=clip_area, omit_background=True, type=image_format
            )
        else:
            screenshot_bytes = page.screenshot(
                full_page=True, omit_background=True, type=image_format
            )

        # 转换为 PIL Image 并裁剪
        # pil_image = Image.open(io.BytesIO(screenshot_bytes))
        # pil_image = trim_whitespace(pil_image)

        # 转换回字节
        output = io.BytesIO(screenshot_bytes)
        # pil_image.save(output, format=image_format.upper())
        output.seek(0)

        process_time = time.time() - start_time
        if random.random() < 0.01:
            logger.info(f"处理时间: {process_time:.3f}秒，文本长度: {len(markdown_text)}")

        return output.getvalue()

    finally:
        # 关闭页面（浏览器实例在线程本地，不需要返回池中）
        if page:
            page.close()


async def markdown_to_image(markdown_text: str, image_format: str = "png") -> bytes:
    """
    将 Markdown 文本转换为图片（异步包装）

    Args:
        markdown_text: Markdown 文本内容
        image_format: 图片格式 (png 或 jpeg)

    Returns:
        图片的字节数据
    """
    # 使用线程池执行同步操作，避免跨线程问题
    loop = asyncio.get_event_loop()
    executor = get_executor()
    return await loop.run_in_executor(
        executor, _markdown_to_image_sync, markdown_text, image_format
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时
    print("正在启动 Markdown to Image API 服务...")
    get_executor()
    print("服务启动完成，准备接收请求")
    yield
    # 关闭时
    print("正在关闭服务...")
    global executor
    if executor:
        # 关闭执行器，这会触发所有线程的清理
        executor.shutdown(wait=True)
    print("服务已关闭")


app = FastAPI(title="Markdown to Image API", version="1.0.0", lifespan=lifespan)


@app.get("/")
async def root():
    """根路径，返回 API 信息"""
    return {
        "service": "Markdown to Image API",
        "version": "1.0.0",
        "endpoints": {
            "/render": "POST - 将 Markdown 转换为图片",
            "/health": "GET - 健康检查",
        },
    }


@app.get("/health")
async def health_check():
    """健康检查端点"""
    global executor
    return {
        "status": "healthy",
        "executor_initialized": executor is not None,
    }


@app.post("/render")
async def render_markdown(request: MarkdownRequest):
    """
    将 Markdown 文本渲染为图片

    Args:
        request: 包含 Markdown 内容的请求对象

    Returns:
        图片响应
    """
    if not request.content or not request.content.strip():
        raise HTTPException(status_code=400, detail="Markdown 内容不能为空")

    if request.format not in ["png", "jpeg"]:
        raise HTTPException(
            status_code=400, detail="不支持的图片格式，仅支持 png 和 jpeg"
        )

    try:
        start_time = time.time()
        image_bytes = await markdown_to_image(request.content, request.format)
        total_time = time.time() - start_time

        if random.random() < 0.01:
            logger.info(f"[Render Success] 处理时间: {total_time:.3f}秒，文本长度: {len(request.content)}")

        return Response(
            content=image_bytes,
            media_type=f"image/{request.format}",
            headers={
                "X-Processing-Time": f"{total_time:.3f}",
                "X-Image-Size": str(len(image_bytes)),
            },
        )
    except Exception as e:
        logger.error(f"[Render Error] 渲染错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"渲染失败: {str(e)}")



if __name__ == "__main__":
    # 启动服务器
    uvicorn.run(app, host="0.0.0.0", port=9000, log_level="info", access_log=True)
