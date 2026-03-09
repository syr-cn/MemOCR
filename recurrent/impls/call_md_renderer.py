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
API_BASE_PORT = os.environ.get("RENDER_SERVER_PORT", 9000)
API_BASE_URL = f"http://{API_BASE_IP}:{API_BASE_PORT}"
os.environ['NO_PROXY'] = f"localhost,127.0.0.1,{API_BASE_IP}"

def retry_on_error(max_retries: int = 3, delay: int = 1):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for i in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    print(f"Error in retry_wrapper: {str(e)}")
                    time.sleep(delay)
            raise Exception(f"All requests failed, maximum retries: {max_retries}")
        return wrapper
    return decorator

@retry_on_error(max_retries=3, delay=1)
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

        response.raise_for_status()
        return response.content, elapsed_time, len(response.content), True
    except Exception as e:
        print(f"Error in test_single_request. Input: \n```{markdown_content}``\nError: {str(e)}")
        raise e



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
    markdown_list: List[str],
    width: int = 800,
    max_workers: int = 10
) -> List[Optional[Image.Image]]:
    """
    Takes a list of markdown strings and returns a list of raw generated PIL images 
    using concurrent requests.
    
    Args:
        markdown_list: A list of strings to be converted to images.
        width: Viewport width.
        max_workers: The number of threads to use for parallel requests.
        
    Returns:
        A list of PIL.Image objects (or None if a request failed).
    """
    results = [None for _ in range(len(markdown_list))]

    def process_item(index, content):
        image_bytes, _, _, success = test_single_request(content)
        
        if success and image_bytes:
            try:
                image = Image.open(io.BytesIO(image_bytes))
                image.load()  # Force load data into memory so file pointer isn't needed
                return index, image
            except Exception as e:
                print(f"Error loading image index {index}: {e}")
                return index, None
        else:
            return index, None

    print(f"Starting batch generation for {len(markdown_list)} items with {max_workers} workers...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_item, i, content) 
            for i, content in enumerate(markdown_list)
        ]
        
        for future in concurrent.futures.as_completed(futures):
            idx, img = future.result()
            results[idx] = img

    return results

def batch_filter_images(
    images: List[Optional[Image.Image]],
) -> List[Optional[Image.Image]]:
    """
    Takes a list of PIL images and filters out images that are too small or too large.
    """
    filtered_images = []
    for img in images:
        if img is None:
            filtered_images.append(None)
            continue
        if min(img.size) < 10 or max(img.size)/min(img.size) > 100:
            filtered_images.append(None)
            continue
        filtered_images.append(img)
    assert len(filtered_images) == len(images), f"len(filtered_images): {len(filtered_images)} != len(images): {len(images)}"
    return filtered_images

def batch_process_images(
    images: List[Optional[Image.Image]],
    max_workers: int = 10
) -> List[Optional[Image.Image]]:
    """
    Takes a list of PIL images and applies optional Noise and Subsampling 
    augmentations based on environment variables.
    
    Environment Variables:
        AUG_NOISE_PROB (float): Probability to add noise (0.0 to 1.0).
        AUG_NOISE_SIGMA (float): Sigma for Gaussian noise.
        AUG_SUBSAMPLE_PROB (float): Probability to subsample (0.0 to 1.0).
        AUG_SUBSAMPLE_RATIO (float): Ratio for subsampling.
    """
    # 1. Read configuration from Environment Variables
    raise NotImplementedError("Not implemented yet.")
    try:
        GLOBAL_PERTURB_SWITCH = int(os.environ.get("GLOBAL_PERTURB_SWITCH", "0"))

        noise_prob = float(os.environ.get("AUG_NOISE_PROB", "0.0"))
        noise_sigma = float(os.environ.get("AUG_NOISE_SIGMA", "25.0"))
        
        subsample_prob = float(os.environ.get("AUG_SUBSAMPLE_PROB", "0.0"))
        subsample_ratio = float(os.environ.get("AUG_SUBSAMPLE_RATIO", "0.5"))
    except ValueError:
        print("Warning: Invalid environment variable format for augmentations. Defaults used.")
        noise_prob, noise_sigma = 0.0, 25.0
        subsample_prob, subsample_ratio = 0.0, 0.5
        GLOBAL_PERTURB_SWITCH = 0

    if GLOBAL_PERTURB_SWITCH == 1:
        enable_noise_flag = noise_prob > 0 and random.random() < noise_prob
        enable_subsample_flag = subsample_prob > 0 and random.random() < subsample_prob
    else:
        enable_noise_flag = False
        enable_subsample_flag = False

    print(f"Processing batch of {len(images)} images.")
    print(f"Augmentation Config -> Noise: {enable_noise_flag} (sigma={noise_sigma}) | Subsample: {enable_subsample_flag} (ratio={subsample_ratio})")

    processed_results = [None for _ in range(len(images))]

    def apply_transformations(idx: int, img: Image.Image) -> Image.Image:
        if enable_noise_flag:
            img = add_gaussian_noise(img, sigma=noise_sigma)
        if enable_subsample_flag:
            img = subsample_image(img, ratio=subsample_ratio)
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

if __name__ == "__main__":
    texts = [
        "# Header 1\nSome content.",
        "## Header 2\nDifferent content.",
        "**Bold text** example."
    ]

    # Get the list of images
    start_time = time.time()
    images = batch_generate_images(texts, max_workers=5)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")

    # Save them locally
    for i, img in enumerate(images):
        if img:
            img.save(f"output_{i}.png")
        else:
            print(f"Image {i} failed to generate.")