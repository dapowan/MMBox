#!/bin/bash

# 创建日志目录
mkdir -p log
mkdir -p output

# 后台运行 evaluate.py，日志写入 log/output.log
nohup python evaluate.py > log/eva_output.log 2>&1 &

echo "✅ evaluate.py is running in background. Logs: log/eva_output.log"
