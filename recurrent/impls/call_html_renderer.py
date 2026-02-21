import concurrent.futures
import io
from PIL import Image
from typing import List, Optional, Tuple
import time
import requests
import statistics
import random
import numpy as np
import os

API_BASE_IP = os.environ.get("RENDER_SERVER_IP")
API_BASE_PORT = os.environ.get("RENDER_SERVER_PORT", 9001)
API_BASE_URL = f"http://{API_BASE_IP}:{API_BASE_PORT}"
os.environ['NO_PROXY'] = f"localhost,127.0.0.1,{API_BASE_IP}"

EMPTY_TEXT = "<h1 style='color: #333;'>NO MEMORY FOUND</h1>"

def retry_on_error(max_retries: int = 3, delay: int = 1):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for i in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    print(f"Error in retry_wrapper: {str(e)}")
                    time.sleep(delay)
            print(f"[Warning] All requests failed, maximum retries: {max_retries}")
            return None
        return wrapper
    return decorator

@retry_on_error(max_retries=3, delay=1)
def test_single_request(html_content: str, width: int = 800) -> Optional[bytes]:
    """
    Test a single request.
    
    Returns:
        Optional[bytes]: The image bytes if successful, None otherwise.
    """
    start_time = time.time()
    try:
        response = requests.post(
            f"{API_BASE_URL}/render",
            json={"content": html_content, "format": "png", "width": width},
            timeout=30
        )
        elapsed_time = time.time() - start_time

        response.raise_for_status()
        return response.content
    except Exception as e:
        print(f"Error in test_single_request. Input: \n```{html_content}``\nError: {str(e)}")
        raise e



def test_concurrent_requests(
    html_content: str,
    num_requests: int = 10,
    width: int = 800,
    max_workers: int = 10
) -> dict:
    """
    Test concurrent requests.
    
    Args:
        html_content: HTML content.
        width: Width.
        num_requests: Total number of requests.
        max_workers: Maximum number of concurrent workers.
        
    Returns:
        A list of PIL.Image objects (or None if a request failed).
    """
    images = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(test_single_request, html_content, width)
            for _ in range(num_requests)
        ]
        
        for future in concurrent.futures.as_completed(futures):
            image_bytes = future.result()
            if image_bytes:
                images.append(Image.open(io.BytesIO(image_bytes)))
    
    return images


def add_gaussian_noise(image: Image.Image, sigma: float = 20.0) -> Image.Image:
    """
    Adds Gaussian noise to a PIL image.
    
    Args:
        image: Input PIL Image.
        sigma: Standard deviation of the Gaussian noise (noise level).
    
    Returns:
        Noisy PIL Image.
    """
    # Convert to numpy array
    img_arr = np.array(image)
    
    # Generate noise
    noise = np.random.normal(0, sigma, img_arr.shape)
    
    # Add noise and clip to valid range [0, 255]
    noisy_arr = np.clip(img_arr + noise, 0, 255).astype(np.uint8)
    
    return Image.fromarray(noisy_arr)


def subsample_image(image: Image.Image, ratio: float = 0.5) -> Image.Image:
    """
    Simulates low-resolution by downsampling and then upsampling back to original size.
    
    Args:
        image: Input PIL Image.
        ratio: Downsample ratio (e.g., 0.5 means half size).
    
    Returns:
        Degraded PIL Image.
    """
    if ratio >= 1.0 or ratio <= 0:
        return image
        
    orig_w, orig_h = image.size
    new_w = int(orig_w * ratio)
    new_h = int(orig_h * ratio)
    
    # Downsample (using Bilinear for better quality reduction)
    small_img = image.resize((new_w, new_h), Image.BILINEAR)
    
    # Upsample back (using Nearest or Bilinear to simulate artifacts)
    # NEAREST makes it look more "pixelated", BICUBIC/BILINEAR makes it blurry.
    # We'll use NEAREST to make the subsampling effect obvious.
    restored_img = small_img.resize((orig_w, orig_h), Image.NEAREST)
    
    return restored_img


# --- Main Batch Function ---
def batch_generate_images(
    html_list: List[str],
    width: int = 800,
    max_workers: int = 10
) -> List[Optional[Image.Image]]:
    """
    Takes a list of html strings and returns a list of raw generated PIL images 
    using concurrent requests.
    
    Args:
        html_list: A list of strings to be converted to images.
        width: Viewport width.
        max_workers: The number of threads to use for parallel requests.
        
    Returns:
        A list of PIL.Image objects (or None if a request failed).
    """
    results = [None for _ in range(len(html_list))]

    def process_item(index, content):
        image_bytes = test_single_request(content, width)
        
        if image_bytes:
            try:
                image = Image.open(io.BytesIO(image_bytes))
                image.load()  # Force load data into memory so file pointer isn't needed
                return index, image
            except Exception as e:
                print(f"Error loading image index {index}: {e}")
                return index, None
        else:
            return index, None

    print(f"Starting batch generation for {len(html_list)} items with {max_workers} workers...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_item, i, content) 
            for i, content in enumerate(html_list)
        ]
        
        for future in concurrent.futures.as_completed(futures):
            idx, img = future.result()
            results[idx] = img

    return results


def batch_process_images(
    images: List[Optional[Image.Image]],
    max_workers: int = 10
) -> List[Optional[Image.Image]]:
    """
    Takes a list of PIL images and applies optional Noise and Subsampling 
    augmentations based on environment variables.
    
    Environment Variables:
        GLOBAL_PERTURB_SWITCH (int): Master switch (1 for on, 0 for off).
        AUG_NOISE_PROB (float): Probability to add noise (0.0 to 1.0).
        AUG_NOISE_SIGMA (float): Sigma for Gaussian noise.
        AUG_SUBSAMPLE_PROB (float): Probability to subsample (0.0 to 1.0).
        AUG_SUBSAMPLE_RATIO (float): Ratio for subsampling.
    """
    processed_results = [None for _ in range(len(images))]

    def apply_transformations(idx: int, img: Image.Image) -> Image.Image:
        # filter out too wierd images
        if min(img.size) < 10 or max(img.size)/min(img.size) > 100:
            return idx, None
        return idx, img
            
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(apply_transformations, idx, img) 
            for idx, img in enumerate(images) if img is not None
        ]
        
        for future in concurrent.futures.as_completed(futures):
            idx, img = future.result()
            processed_results[idx] = img

    return processed_results

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
    html_list = [case["html"] for case in HTML_CASES]
    test_names = [case["name"] for case in HTML_CASES]
    start_time = time.time()
    images = batch_generate_images(html_list, width=800, max_workers=5)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    for name, img in zip(test_names, images):
        if img:
            img.save(f"output_html_{name}.png")
        else:
            print(f"Image {name} failed to generate.")