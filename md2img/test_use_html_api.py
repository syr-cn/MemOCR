



import concurrent.futures
import io
from PIL import Image
from typing import List, Optional, Tuple
import time
import requests
import statistics

API_BASE_URL = "http://33.235.223.64:8002"

def test_single_request(html_content: str, width: int = 800) -> Tuple[float, int, bool]:
    """
    测试单个请求
    
    Returns:
        (处理时间, 响应大小, 是否成功)
    """
    start_time = time.time()
    try:
        response = requests.post(
            f"{API_BASE_URL}/render",
            json={"content": html_content, "format": "png", "width": width},
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
    html_content: str,
    num_requests: int,
    max_workers: int = 10
) -> dict:
    """
    测试并发请求
    
    Args:
        html_content: HTML 内容
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
            executor.submit(test_single_request, html_content)
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
    html_list: List[str],
    max_workers: int = 10
) -> List[Optional[Image.Image]]:
    """
    Takes a list of markdown strings and returns a list of generated images using concurrent requests.
    
    Args:
        html_list: A list of strings to be converted to images.
        max_workers: The number of threads to use for parallel requests.
        
    Returns:
        A list of PIL.Image objects (or None if a request failed), 
        matching the order of the input list.
    """
    results = [None] * len(html_list)

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

    print(f"Starting batch generation for {len(html_list)} items with {max_workers} workers...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks with their original index
        futures = [
            executor.submit(process_item, i, content) 
            for i, content in enumerate(html_list)
        ]
        
        # Process as they complete to show progress (optional), or just wait for all
        for future in concurrent.futures.as_completed(futures):
            idx, img = future.result()
            results[idx] = img

    return results

HTML_CASES = [
    {
        "name": "01_basic_text",
        "html": """
            <h1 style="color: #333;">Hello World</h1>
            <p>This is a basic HTML rendering test.</p>
            <p>It supports <b>bold</b>, <i>italic</i>, and <span style="color: red;">inline styles</span>.</p>
        """
    },
    {
        "name": "02_data_table",
        "html": """
            <h2>Employee Data</h2>
            <table>
              <tr>
                <th>Firstname</th>
                <th>Lastname</th>
                <th>Savings</th>
              </tr>
              <tr>
                <td>Peter</td>
                <td>Griffin</td>
                <td>$100</td>
              </tr>
              <tr>
                <td>Lois</td>
                <td>Griffin</td>
                <td>$150</td>
              </tr>
              <tr>
                <td>Joe</td>
                <td>Swanson</td>
                <td>$300</td>
              </tr>
            </table>
        """
    },
    {
        "name": "03_code_snippet",
        "html": """
            <div style="background: #2d2d2d; color: #ccc; padding: 15px; border-radius: 8px; font-family: monospace;">
                <div><span style="color: #cc99cd;">def</span> <span style="color: #f8c555;">hello</span>():</div>
                <div style="padding-left: 20px;">print(<span style="color: #7ec699;">"Hello Python!"</span>)</div>
            </div>
        """
    },
    {
        "name": "04_dashboard_card",
        "html": """
            <div style="
                border: 1px solid #e0e0e0; 
                border-radius: 12px; 
                box-shadow: 0 4px 6px rgba(0,0,0,0.1); 
                padding: 20px; 
                width: 300px; 
                font-family: sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
            ">
                <div style="font-size: 14px; opacity: 0.8;">Total Revenue</div>
                <div style="font-size: 32px; font-weight: bold; margin: 10px 0;">$45,231.89</div>
                <div style="display: flex; align-items: center; font-size: 14px;">
                    <span style="background: rgba(255,255,255,0.2); padding: 2px 8px; border-radius: 10px;">+20.1%</span>
                    <span style="margin-left: 10px; opacity: 0.8;">from last month</span>
                </div>
            </div>
        """
    },
    {
        "name": "05_complex_grid",
        "html": """
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; width: 400px;">
                <div style="background: #ffadad; padding: 20px; text-align: center; border-radius: 5px;">Box 1</div>
                <div style="background: #ffd6a5; padding: 20px; text-align: center; border-radius: 5px;">Box 2</div>
                <div style="background: #fdffb6; padding: 20px; text-align: center; border-radius: 5px; grid-column: span 2;">Wide Box 3</div>
            </div>
        """
    }
]

if __name__ == "__main__":
    # Get the list of images
    test_names = [case["name"] for case in HTML_CASES]
    test_htmls = [case["html"] for case in HTML_CASES]
    images = generate_images_concurrently(test_htmls, max_workers=5)

    # Save them locally
    for name, img in zip(test_names, images):
        if img:
            img.save(f"output_html_{name}.png")
        else:
            print(f"Image {name} failed to generate.")