"""
Markdown API 性能测试脚本
测试高 QPS 场景下的 API 性能
"""
import requests
import time
import statistics
import concurrent.futures
from typing import List, Tuple
import json
from PIL import Image
import io

# API_BASE_URL = "http://localhost:8000"
API_BASE_URL = "http://33.235.246.42:8000"


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


def test_health_check():
    """测试健康检查端点"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"健康检查通过:")
            print(f"  - 浏览器池大小: {data.get('browser_pool_size', 'N/A')}")
            print(f"  - 可用浏览器数: {data.get('available_browsers', 'N/A')}")
            return True
        else:
            print(f"健康检查失败，状态码: {response.status_code}")
            return False
    except Exception as e:
        print(f"健康检查异常: {str(e)}")
        return False


def print_test_results(results: dict):
    """打印测试结果"""
    print("\n" + "="*60)
    print("测试结果统计")
    print("="*60)
    print(f"总请求数: {results.get('total_requests', 0)}")
    print(f"成功请求数: {results.get('successful_requests', 0)}")
    print(f"失败请求数: {results.get('failed_requests', 0)}")
    print(f"总耗时: {results.get('total_time', 0):.3f} 秒")
    print(f"QPS (每秒请求数): {results.get('qps', 0):.2f}")
    print(f"\n响应时间统计:")
    print(f"  平均: {results.get('avg_time', 0):.3f} 秒")
    print(f"  中位数: {results.get('median_time', 0):.3f} 秒")
    print(f"  最小值: {results.get('min_time', 0):.3f} 秒")
    print(f"  最大值: {results.get('max_time', 0):.3f} 秒")
    print(f"  P95: {results.get('p95_time', 0):.3f} 秒")
    print(f"  P99: {results.get('p99_time', 0):.3f} 秒")
    print(f"\n平均图片大小: {results.get('avg_image_size', 0):.0f} 字节")
    print("="*60)


def main():
    """主测试函数"""
    print("Markdown to Image API 性能测试")
    print("="*60)
    
    # 测试健康检查
    print("\n1. 测试健康检查端点...")
    if not test_health_check():
        print("健康检查失败，请确保 API 服务正在运行")
        print("启动命令: python3 markdown_api_server.py")
        return
    
    # 准备测试用的 Markdown 内容
    simple_md = """
# 简单 Markdown 测试

这是一个**简单**的 Markdown 测试内容。

## 特点
- 快速渲染
- 高质量输出
"""
    
    complex_md = """
# Complex Markdown Test

This is a Markdown document with multiple elements.

## Title Examples

### Third-level Title

#### Fourth-level Title

**Bold Text** and *Italic Text*

## List Examples

### Unordered List
- Item 1
- Item 2
  - Sub-item 2.1
  - Sub-item 2.2
- Item 3

### Ordered List
1. First Item
2. Second Item
3. Third Item

## Code Examples

Inline Code: `print("Hello, World!")`

Code Block:

```python
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
```

## Quote Examples

> This is a quoted text.
> It can contain multiple lines.

## Link and Image Examples

[Link Text](https://example.com)

---

**Test Completed**
"""
    
    # 测试场景
    test_scenarios = [
        {
            "name": "Simple Content - Single Request",
            "content": simple_md,
            "num_requests": 1,
            "max_workers": 1
        },
        {
            "name": "Simple Content - 10 Concurrent Requests",
            "content": simple_md,
            "num_requests": 10,
            "max_workers": 10
        },
        {
            "name": "Simple Content - 50 Concurrent Requests",
            "content": simple_md,
            "num_requests": 50,
            "max_workers": 20
        },
        {
            "name": "Complex Content - Single Request",
            "content": complex_md,
            "num_requests": 1,
            "max_workers": 1
        },
        {
            "name": "Complex Content - 10 Concurrent Requests",
            "content": complex_md,
            "num_requests": 10,
            "max_workers": 10
        },
        {
            "name": "Complex Content - 50 Concurrent Requests",
            "content": complex_md,
            "num_requests": 50,
            "max_workers": 20
        },
        {
            "name": "Complex Content - 512 Concurrent Requests",
            "content": complex_md,
            "num_requests": 512,
            "max_workers": 20
        },
    ]
    
    all_results = []
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n{'='*60}")
        print(f"Test Scenario {i}: {scenario['name']}")
        print(f"{'='*60}")
        
        results = test_concurrent_requests(
            scenario['content'],
            scenario['num_requests'],
            scenario['max_workers']
        )
        
        results['scenario'] = scenario['name']
        all_results.append(results)
        print_test_results(results)
        
        # Wait for the next test
        if i < len(test_scenarios):
            print("\nWaiting 2 seconds for the next test...")
            time.sleep(2)
    
    # 总结
    print("\n" + "="*60)
    print("所有测试场景总结")
    print("="*60)
    for result in all_results:
        print(f"\n{result['scenario']}:")
        print(f"  QPS: {result.get('qps', 0):.2f}")
        print(f"  平均响应时间: {result.get('avg_time', 0):.3f} 秒")
        print(f"  P95 响应时间: {result.get('p95_time', 0):.3f} 秒")
    
    # 性能评估
    print("\n" + "="*60)
    print("性能评估")
    print("="*60)
    
    # 找出最佳 QPS
    best_qps = max(r.get('qps', 0) for r in all_results)
    best_scenario = next(r for r in all_results if r.get('qps', 0) == best_qps)
    print(f"最高 QPS: {best_qps:.2f} ({best_scenario['scenario']})")
    
    # 找出平均响应时间
    avg_times = [r.get('avg_time', 0) for r in all_results if r.get('successful_requests', 0) > 0]
    if avg_times:
        overall_avg = statistics.mean(avg_times)
        print(f"总体平均响应时间: {overall_avg:.3f} 秒")
    
    print("\n性能建议:")
    if best_qps >= 20:
        print("✓ 系统可以处理高 QPS 请求（≥20 QPS）")
    elif best_qps >= 10:
        print("⚠ 系统可以处理中等 QPS 请求（10-20 QPS），建议优化")
    else:
        print("✗ 系统 QPS 较低（<10 QPS），需要优化")
    
    if avg_times and overall_avg < 1.0:
        print("✓ 平均响应时间良好（<1秒）")
    elif avg_times and overall_avg < 2.0:
        print("⚠ 平均响应时间可接受（1-2秒）")
    else:
        print("✗ 平均响应时间较长（>2秒），需要优化")


if __name__ == "__main__":
    main()

