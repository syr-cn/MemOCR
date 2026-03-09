import concurrent.futures
import io
from PIL import Image
from typing import List, Optional, Tuple
import time
import requests
import statistics

from test_markdown_api import test_single_request

API_BASE_URL = "http://33.235.223.64:8000"

def test_single_request(markdown_content: str) -> Tuple[float, int, bool]:
    """
    测试单个请求
    
    Returns:
        (处理时间, 响应大小, 是否成功)
    """
    start_time = time.time()
    try:
        response = requests.post(
            f"{API_BASE_URL}/render",
            json={"content": markdown_content, "format": "png"},
            timeout=30
        )
        elapsed_time = time.time() - start_time
        
        if response.status_code == 200:
            return response.content, elapsed_time, len(response.content), True
        else:
            print(f"请求失败，状态码: {response.status_code}, 响应: {response.text}")
            return None, elapsed_time, 0, False
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"请求异常: {str(e)}")
        return None, elapsed_time, 0, False


def test_concurrent_requests(
    markdown_content: str,
    num_requests: int,
    max_workers: int = 10
) -> dict:
    """
    测试并发请求
    
    Args:
        markdown_content: Markdown 内容
        num_requests: 请求总数
        max_workers: 最大并发数
        
    Returns:
        测试结果统计
    """
    print(f"\n开始并发测试: {num_requests} 个请求, 最大并发数: {max_workers}")
    
    times = []
    images = []
    sizes = []
    success_count = 0
    
    start_total = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(test_single_request, markdown_content)
            for _ in range(num_requests)
        ]
        
        for future in concurrent.futures.as_completed(futures):
            image_bytes, elapsed, size, success = future.result()
            times.append(elapsed)
            if success:
                images.append(Image.open(io.BytesIO(image_bytes)))
                sizes.append(size)
                success_count += 1
    
    total_time = time.time() - start_total
    with open("images.png", "wb") as f:
        images[0].save(f, format="PNG")
    
    if times:
        return {
            "total_requests": num_requests,
            "successful_requests": success_count,
            "failed_requests": num_requests - success_count,
            "total_time": total_time,
            "qps": num_requests / total_time if total_time > 0 else 0,
            "avg_time": statistics.mean(times),
            "median_time": statistics.median(times),
            "min_time": min(times),
            "max_time": max(times),
            "p95_time": statistics.quantiles(times, n=20)[18] if len(times) > 1 else times[0],
            "p99_time": statistics.quantiles(times, n=100)[98] if len(times) > 1 else times[0],
            "avg_image_size": statistics.mean(sizes) if sizes else 0,
        }
    else:
        return {"error": "所有请求都失败了"}


def generate_images_concurrently(
    markdown_list: List[str],
    max_workers: int = 10
) -> List[Optional[Image.Image]]:
    """
    Takes a list of markdown strings and returns a list of generated images using concurrent requests.
    
    Args:
        markdown_list: A list of strings to be converted to images.
        max_workers: The number of threads to use for parallel requests.
        
    Returns:
        A list of PIL.Image objects (or None if a request failed), 
        matching the order of the input list.
    """
    results = [None] * len(markdown_list)

    # Helper wrapper to keep track of the index so we can maintain order
    def process_item(index, content):
        # We assume 'test_single_request' is your existing function that handles the API call.
        # It likely returns: (image_bytes, elapsed_time, size, success_boolean)
        try:
            # Replace 'test_single_request' with your actual request function name if different
            image_bytes, _, _, success = test_single_request(content)
            
            if success and image_bytes:
                image = Image.open(io.BytesIO(image_bytes))
                image.load() # Force load the image data into memory while the file is 'open'
                return index, image
            else:
                return index, None
        except Exception as e:
            print(f"Error processing item {index}: {e}")
            return index, None

    print(f"Starting batch generation for {len(markdown_list)} items with {max_workers} workers...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks with their original index
        futures = [
            executor.submit(process_item, i, content) 
            for i, content in enumerate(markdown_list)
        ]
        
        # Process as they complete to show progress (optional), or just wait for all
        for future in concurrent.futures.as_completed(futures):
            idx, img = future.result()
            results[idx] = img

    return results

dummy_markdown = """
 # Memory Update: Government Position of Corliss Archer’s Portrayal

## Key Information Extracted from Article

- The film **Kiss and Tell (1945)** stars **Shirley Temple** as **Corliss Archer**.
- Shirley Temple was a **child actress** who portrayed the character in the film.
- The film was released in **1945** and is an American comedy.
- The article does not mention Shirley Temple holding any government position.

## Previous Memory (Retained)

- The woman who portrayed Corliss Archer in the film *Kiss and Tell* is **Shirley Temple**.
- No prior memory indicated her government position.

## New Information Added

- Shirley Temple was a **child actress**, not a government official.
- She was **not** known to have held any government position.
- The article confirms her role in the 1945 film, but provides no information about her political career or any government office held.

## Conclusion

The woman who portrayed Corliss Archer in the film *Kiss and Tell* — **Shirley Temple** — **did not hold any government position**. She was a child actress and did not serve in any political office.

This information is now fully documented in the memory.
"""

if __name__ == "__main__":
    texts = [
        "# Header 1\nSome content.",
        "## Header 2\nDifferent content.",
        "**Bold text** example.",
        dummy_markdown,
    ]

    # Get the list of images
    images = generate_images_concurrently(texts, max_workers=5)

    # Save them locally
    for i, img in enumerate(images):
        if img:
            img.save(f"markdown_output_{i}.png")
        else:
            print(f"Image {i} failed to generate.")