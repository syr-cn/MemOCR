#!/bin/bash

# Markdown API 服务启动脚本

echo "正在启动 Markdown to Image API 服务..."
echo "服务将在 http://localhost:8000 启动"
echo "按 Ctrl+C 停止服务"
echo ""

python3 markdown_api_server.py 2>&1 | tee api_server.log

echo "服务已停止"
exit 0