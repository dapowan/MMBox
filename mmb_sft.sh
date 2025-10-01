#!/bin/bash

# 创建日志目录
mkdir -p log
mkdir -p output

# 后台运行 mmb_sft.py，日志写入 log/output.log
nohup python mmb_sft.py > log/output.log 2>&1 &

echo "✅ mmb_sft.py is running in background. Logs: log/output.log"
