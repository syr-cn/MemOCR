# Markdown to Image API 服务

高性能的 Markdown 转图片 HTTP API 服务，支持高 QPS 请求。

## 功能特点

- ✅ HTTP RESTful API 接口
- ✅ 浏览器实例池管理，支持并发请求
- ✅ 自动扩展浏览器实例（最大 20 个）
- ✅ 性能优化，支持高 QPS
- ✅ 健康检查端点
- ✅ 支持 PNG 和 JPEG 格式

## 安装依赖

```bash
pip install -r requirements_api.txt
playwright install
```

## 启动服务

```bash
python3 markdown_api_server.py
```

服务将在 `http://localhost:9000` 启动。

## API 端点

### 1. 根路径
```
GET /
```
返回 API 信息

### 2. 健康检查
```
GET /health
```
返回服务健康状态和浏览器池信息

### 3. Markdown 渲染
```
POST /render
Content-Type: application/json

{
    "content": "# Markdown 内容",
    "format": "png"  // 可选: "png" 或 "jpeg"，默认 "png"
}
```

返回: 图片二进制数据（Content-Type: image/png 或 image/jpeg）

响应头包含:
- `X-Processing-Time`: 处理时间（秒）
- `X-Image-Size`: 图片大小（字节）

## 性能测试

运行性能测试脚本：

```bash
# 确保服务已启动
python3 markdown_api_server.py

# 在另一个终端运行测试
python3 test_markdown_api.py
```

## 配置说明

在 `markdown_api_server.py` 中可以调整以下参数：

- `pool_size`: 初始浏览器池大小（默认 5）
- `max_size`: 最大浏览器实例数（默认 20）

## 性能优化

1. **浏览器实例池**: 复用浏览器实例，避免频繁创建/销毁
2. **队列管理**: 使用线程安全的队列管理浏览器实例
3. **自动扩展**: 根据负载自动创建新的浏览器实例
4. **快速加载**: 使用 `domcontentloaded` 而非 `networkidle` 以提升速度
5. **连接复用**: 浏览器实例在请求间复用

## 预期性能

根据测试，预期性能指标：

- **单请求响应时间**: 0.5-1.5 秒
- **并发 QPS**: 10-30 QPS（取决于内容复杂度）
- **P95 响应时间**: < 2 秒
- **P99 响应时间**: < 3 秒

## 注意事项

1. 首次启动需要下载浏览器，可能需要一些时间
2. 浏览器实例会占用内存，根据服务器配置调整 `max_size`
3. 建议在生产环境中使用进程管理器（如 systemd、supervisor）管理服务

